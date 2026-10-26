"""Build patient-separated slice-feature CSVs one patient at a time."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_loading import load_patient
from config import settings
from feature_extraction.slice_features import (
    FEATURE_NAMES,
    extract_slice_features,
)

SplitName = Literal["train", "validation", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS: dict[SplitName, float] = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
DEFAULT_SEED = 42
DEFAULT_NORMALIZATION_STRATEGY = "zscore"
MODEL_FEATURE_COLUMNS = tuple(
    f"{modality}_{statistic}"
    for modality in settings.MRI_MODALITIES
    for statistic in FEATURE_NAMES
)
CSV_COLUMNS = ("patient_id", "slice_index", "target", *MODEL_FEATURE_COLUMNS)
NORMALIZATION_POLICY = (
    "Per-patient, per-modality z-score using normalize_mri(strategy='zscore') "
    "with an explicit all-False background mask; all finite voxels, including "
    "zero-valued voxels, participate. Non-finite inputs are excluded from "
    "normalization statistics and set to zero by normalize_mri; original "
    "finite-voxel masks are used during feature extraction."
)
TARGET_POLICY = (
    "1 if any segmentation voxel in the axial slice is nonzero; otherwise 0. "
    "This indicates mask-positive versus zero-mask slices only, not healthy "
    "versus diseased patients, and does not assign subregion meanings to "
    "segmentation labels 1, 2, or 3."
)


class DatasetBuildError(RuntimeError):
    """Raised when patient data cannot be safely converted into split files."""


def _normalization_module() -> ModuleType:
    """Load normalization by file path to avoid preprocessing.py collision."""
    module_name = "phase2_preprocessing_normalization"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    module_path = PROJECT_ROOT / "preprocessing" / "normalization.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load normalization module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def feature_columns() -> tuple[str, ...]:
    """Return only the ordered model-input feature columns."""
    return MODEL_FEATURE_COLUMNS


def _validated_ratios(
    split_ratios: Mapping[str, float] | None,
) -> dict[SplitName, float]:
    ratios = (
        dict(DEFAULT_SPLIT_RATIOS)
        if split_ratios is None
        else dict(split_ratios)
    )
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(
            "split_ratios must contain exactly 'train', 'validation', and 'test'"
        )

    validated: dict[SplitName, float] = {}
    for name in SPLIT_NAMES:
        value = ratios[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Split ratio for {name!r} must be a finite number")
        ratio = float(value)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Split ratio for {name!r} must be greater than zero")
        validated[name] = ratio

    total = sum(validated.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0; received {total}")
    return validated


def _allocate_counts(
    patient_count: int,
    ratios: Mapping[SplitName, float],
) -> dict[SplitName, int]:
    """Apportion patients after reserving one patient for every partition."""
    if patient_count < len(SPLIT_NAMES):
        raise ValueError(
            "At least three patients are required to create nonempty "
            "train, validation, and test partitions"
        )

    remaining = patient_count - len(SPLIT_NAMES)
    ratio_total = sum(ratios.values())
    exact = {
        name: remaining * ratios[name] / ratio_total
        for name in SPLIT_NAMES
    }
    extra = {name: math.floor(exact[name]) for name in SPLIT_NAMES}
    unassigned = remaining - sum(extra.values())
    remainder_order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(exact[name] - extra[name]), SPLIT_NAMES.index(name)),
    )
    for name in remainder_order[:unassigned]:
        extra[name] += 1

    counts = {name: extra[name] + 1 for name in SPLIT_NAMES}
    if sum(counts.values()) != patient_count or any(count == 0 for count in counts.values()):
        raise ValueError(
            "Split configuration cannot produce three nonempty partitions"
        )
    return counts


def split_patients(
    patient_ids: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
    split_ratios: Mapping[str, float] | None = None,
) -> dict[SplitName, tuple[str, ...]]:
    """Return reproducible, disjoint patient assignments for all three splits."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    ids = list(patient_ids)
    if any(not isinstance(patient_id, str) or not patient_id for patient_id in ids):
        raise ValueError("patient IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("patient IDs must be unique")

    ratios = _validated_ratios(split_ratios)
    counts = _allocate_counts(len(ids), ratios)
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)

    assignments: dict[SplitName, tuple[str, ...]] = {}
    start = 0
    for name in SPLIT_NAMES:
        stop = start + counts[name]
        assignments[name] = tuple(shuffled[start:stop])
        start = stop
    return assignments


def _enumerate_patients(dataset_root: Path) -> list[str]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Configured extracted dataset directory does not exist: {dataset_root}"
        )
    patient_ids = sorted(
        entry.name
        for entry in dataset_root.iterdir()
        if entry.is_dir() and entry.name.startswith("BraTS-GLI-")
    )
    if not patient_ids:
        raise ValueError(f"No BraTS-GLI patient folders found in {dataset_root}")
    return patient_ids


def _validate_patient(patient_id: str, patient: Any) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not getattr(patient, "dimensions_compatible", False):
        raise ValueError("MRI modalities and segmentation do not have matching 3D shapes")
    if not getattr(patient, "segmentation_labels_valid", False):
        raise ValueError("segmentation contains invalid or unsupported labels")

    volumes = getattr(patient, "volumes", None)
    if not isinstance(volumes, Mapping):
        raise ValueError("loader did not provide a volume mapping")
    missing = [name for name in settings.MRI_MODALITIES if name not in volumes]
    if settings.SEGMENTATION_MASK_SUFFIX not in volumes:
        missing.append(settings.SEGMENTATION_MASK_SUFFIX)
    if missing:
        raise ValueError(f"loader omitted required volumes: {', '.join(missing)}")

    modalities = {
        name: np.asarray(volumes[name])
        for name in settings.MRI_MODALITIES
    }
    segmentation = np.asarray(volumes[settings.SEGMENTATION_MASK_SUFFIX])
    reference_shape = modalities[settings.MRI_MODALITIES[0]].shape
    if len(reference_shape) != 3 or any(
        volume.shape != reference_shape for volume in modalities.values()
    ):
        raise ValueError("MRI modality volumes must have matching 3D shapes")
    if segmentation.shape != reference_shape:
        raise ValueError(
            f"segmentation shape {segmentation.shape} does not match "
            f"MRI shape {reference_shape}"
        )
    return modalities, segmentation


def build_dataset(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    split_ratios: Mapping[str, float] | None = None,
    dataset_root: str | Path | None = None,
    patient_loader: Callable[[str], Any] | None = None,
) -> Path:
    """Create split CSVs and a manifest, processing one patient at a time.

    The output directory must not already exist. Files are written to a
    temporary sibling directory and published only after every patient has
    been processed successfully; failures remove the temporary output and
    identify the patient being processed.

    `dataset_root` and `patient_loader` are injectable for isolated tests. When
    omitted, the configured dataset root and existing load_patient interface
    are used. The normalization policy includes all finite intensities,
    including zeros, and feature extraction excludes pixels that were
    non-finite in the original modality.
    """
    ratios = _validated_ratios(split_ratios)
    root = Path(settings.DATASET_ROOT) if dataset_root is None else Path(dataset_root)
    patient_ids = _enumerate_patients(root)
    assignments = split_patients(
        patient_ids,
        seed=seed,
        split_ratios=ratios,
    )
    normalize_mri = _normalization_module().normalize_mri

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.building-",
            dir=destination.parent,
        )
    )
    loader = load_patient if patient_loader is None else patient_loader
    loader = load_patient if patient_loader is None else patient_loader
    label_counts: dict[SplitName, Counter[int]] = {
        name: Counter({0: 0, 1: 0}) for name in SPLIT_NAMES
    }

    try:
        with (
            (staging_dir / "train.csv").open(
                "w", newline="", encoding="utf-8"
            ) as train_file,
            (staging_dir / "validation.csv").open(
                "w", newline="", encoding="utf-8"
            ) as validation_file,
            (staging_dir / "test.csv").open(
                "w", newline="", encoding="utf-8"
            ) as test_file,
        ):
            files = {
                "train": train_file,
                "validation": validation_file,
                "test": test_file,
            }
            writers = {
                name: csv.DictWriter(files[name], fieldnames=CSV_COLUMNS)
                for name in SPLIT_NAMES
            }
            for writer in writers.values():
                writer.writeheader()

            for split_name in SPLIT_NAMES:
                for patient_id in assignments[split_name]:
                    try:
                        patient = loader(patient_id)
                        modalities, segmentation = _validate_patient(
                            patient_id, patient
                        )
                        shape = modalities[settings.MRI_MODALITIES[0]].shape
                        background_mask = np.zeros(shape, dtype=bool)
                        finite_masks = {
                            modality: np.isfinite(volume)
                            for modality, volume in modalities.items()
                        }
                        normalized = {
                            modality: normalize_mri(
                                modalities[modality],
                                strategy=DEFAULT_NORMALIZATION_STRATEGY,
                                background_mask=background_mask,
                            )
                            for modality in settings.MRI_MODALITIES
                        }
                        slice_labels = np.any(segmentation != 0, axis=(0, 1))

                        for slice_index, positive in enumerate(slice_labels):
                            target = int(positive)
                            features = extract_slice_features(
                                normalized,
                                slice_index,
                                foreground_masks=finite_masks,
                            )
                            if features.shape != (len(MODEL_FEATURE_COLUMNS),):
                                raise ValueError(
                                    "feature extractor returned unexpected shape "
                                    f"{features.shape}"
                                )
                            row: dict[str, object] = {
                                "patient_id": patient_id,
                                "slice_index": slice_index,
                                "target": target,
                            }
                            row.update(
                                zip(MODEL_FEATURE_COLUMNS, features.tolist())
                            )
                            writers[split_name].writerow(row)
                            label_counts[split_name][target] += 1
                    except Exception as exc:
                        raise DatasetBuildError(
                            f"Failed to process patient {patient_id!r} "
                            f"assigned to {split_name}: {exc}"
                        ) from exc
                    finally:
                        if "patient" in locals():
                            del patient
                        if "modalities" in locals():
                            del modalities
                        if "segmentation" in locals():
                            del segmentation
                        if "normalized" in locals():
                            del normalized
                        if "finite_masks" in locals():
                            del finite_masks
                        if "background_mask" in locals():
                            del background_mask

        manifest = {
            "random_seed": seed,
            "split_ratios": ratios,
            "patient_assignments": {
                name: list(assignments[name]) for name in SPLIT_NAMES
            },
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "csv_columns": list(CSV_COLUMNS),
            "normalization_strategy": DEFAULT_NORMALIZATION_STRATEGY,
            "normalization_background_policy": NORMALIZATION_POLICY,
            "target_policy": TARGET_POLICY,
            "per_split_label_counts": {
                name: {
                    "zero_mask": label_counts[name][0],
                    "mask_positive": label_counts[name][1],
                }
                for name in SPLIT_NAMES
            },
        }
        with (staging_dir / "manifest.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")

        os.rename(staging_dir, destination)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="New directory for CSV splits and manifest")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        output = build_dataset(args.output_dir, seed=args.seed)
    except (DatasetBuildError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Dataset build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset splits and manifest written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
