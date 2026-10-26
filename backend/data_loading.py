"""Load and validate one BraTS-GLI patient's NIfTI volumes at a time."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


EXPECTED_SEGMENTATION_LABELS = frozenset({0, 1, 2, 3})
NIFTI_EXTENSIONS = (".nii.gz", ".nii")


@dataclass(frozen=True)
class LoadedPatient:
    """Volumes and validation results for one loaded patient."""

    patient_id: str
    volumes: Mapping[str, np.ndarray]
    source_paths: Mapping[str, Path]
    shape_mismatches: Mapping[str, tuple[int, ...]]
    non_3d_volumes: tuple[str, ...]
    segmentation_labels: tuple[int | float, ...]
    invalid_segmentation_labels: tuple[int | float, ...]
    segmentation_has_non_finite_values: bool

    @property
    def dimensions_compatible(self) -> bool:
        """Whether all five volumes have the same three-dimensional shape."""
        return not self.shape_mismatches and not self.non_3d_volumes

    @property
    def segmentation_labels_valid(self) -> bool:
        """Whether the segmentation contains only finite BraTS-GLI labels."""
        return (
            not self.invalid_segmentation_labels
            and not self.segmentation_has_non_finite_values
        )


def _resolve_volume_path(patient_dir: Path, suffix: str) -> Path:
    """Find exactly one NIfTI file for a patient and modality suffix."""
    candidates = [
        patient_dir / f"{patient_dir.name}-{suffix}{extension}"
        for extension in NIFTI_EXTENSIONS
    ]
    matches = [path for path in candidates if path.exists()]

    if not matches:
        expected = " or ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing expected NIfTI volume: {expected}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple NIfTI files found for {patient_dir.name} "
            f"suffix {suffix!r}: {', '.join(str(path) for path in matches)}"
        )
    if not matches[0].is_file():
        raise OSError(f"Expected a regular NIfTI file but found: {matches[0]}")
    return matches[0]


def load_patient(patient_id: str) -> LoadedPatient:
    """Load one patient's modalities and mask without resampling or reshaping.

    Only the requested patient's five arrays are materialized. Shape and label
    problems are returned for explicit reporting rather than silently fixed.
    """
    try:
        import nibabel as nib
        from nibabel.filebasedimages import ImageFileError
    except ImportError as exc:
        raise ImportError(
            "MRI loading requires nibabel, which is not installed in this "
            "Python environment. Ask for approval before installing it."
        ) from exc

    if (
        not patient_id
        or Path(patient_id).name != patient_id
        or patient_id in {".", ".."}
    ):
        raise ValueError("patient_id must be a single patient-folder name")

    dataset_root = Path(settings.DATASET_ROOT)
    patient_dir = dataset_root / patient_id
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    suffixes = (*settings.MRI_MODALITIES, settings.SEGMENTATION_MASK_SUFFIX)
    source_paths: dict[str, Path] = {}
    missing_files: list[str] = []
    for suffix in suffixes:
        try:
            source_paths[suffix] = _resolve_volume_path(patient_dir, suffix)
        except FileNotFoundError as exc:
            missing_files.append(str(exc))
    if missing_files:
        raise FileNotFoundError(
            "Required patient volume file(s) are missing:\n"
            + "\n".join(missing_files)
        )

    volumes: dict[str, np.ndarray] = {}
    for suffix, path in source_paths.items():
        try:
            image = nib.load(str(path))
            volumes[suffix] = np.asanyarray(image.dataobj)
        except (ImageFileError, OSError, ValueError, EOFError) as exc:
            raise OSError(f"Unable to read NIfTI volume {path}: {exc}") from exc

    reference_modality = settings.MRI_MODALITIES[0]
    reference_shape = volumes[reference_modality].shape
    shape_mismatches = {
        suffix: volume.shape
        for suffix, volume in volumes.items()
        if volume.shape != reference_shape
    }
    non_3d_volumes = tuple(
        suffix for suffix, volume in volumes.items() if volume.ndim != 3
    )

    segmentation = volumes[settings.SEGMENTATION_MASK_SUFFIX]
    labels_array = np.unique(segmentation)
    has_non_finite_values = not bool(np.isfinite(segmentation).all())
    segmentation_labels = tuple(
        value.item() for value in labels_array
    )
    invalid_labels = tuple(
        label
        for label in segmentation_labels
        if not isinstance(label, (int, float, np.integer, np.floating))
        or not np.isfinite(label)
        or not float(label).is_integer()
        or int(label) not in EXPECTED_SEGMENTATION_LABELS
    )

    return LoadedPatient(
        patient_id=patient_id,
        volumes=volumes,
        source_paths=source_paths,
        shape_mismatches=shape_mismatches,
        non_3d_volumes=non_3d_volumes,
        segmentation_labels=segmentation_labels,
        invalid_segmentation_labels=invalid_labels,
        segmentation_has_non_finite_values=has_non_finite_values,
    )


def print_report(patient: LoadedPatient) -> None:
    """Print volume metadata and explicit shape/label validation results."""
    for suffix, volume in patient.volumes.items():
        print(
            f"{suffix}: shape={volume.shape}, dtype={volume.dtype}, "
            f"file={patient.source_paths[suffix]}"
        )

    if patient.dimensions_compatible:
        print("PASS: all five volumes have matching 3D dimensions")
    else:
        reference_shape = patient.volumes[settings.MRI_MODALITIES[0]].shape
        print(
            f"FAIL: expected matching 3D dimensions; "
            f"{settings.MRI_MODALITIES[0]} reference shape={reference_shape}"
        )
        for suffix, shape in patient.shape_mismatches.items():
            print(f"FAIL: {suffix} shape={shape} does not match reference")
        for suffix in patient.non_3d_volumes:
            print(f"FAIL: {suffix} is not 3D")

    print(f"Segmentation labels: {list(patient.segmentation_labels)}")
    if patient.segmentation_labels_valid:
        print("PASS: segmentation labels are finite BraTS-GLI values (0, 1, 2, 3)")
    else:
        print("FAIL: segmentation contains invalid or non-finite label values")
        if patient.invalid_segmentation_labels:
            print(
                "Invalid labels: "
                f"{list(patient.invalid_segmentation_labels)}"
            )
        if patient.segmentation_has_non_finite_values:
            print("Invalid values: segmentation contains NaN or infinity")


def main(argv: list[str] | None = None) -> int:
    """Load and report one patient selected by folder name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patient_id",
        help="Patient folder name under the configured extracted dataset root",
    )
    args = parser.parse_args(argv)

    try:
        patient = load_patient(args.patient_id)
    except (ImportError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print_report(patient)
    if not patient.dimensions_compatible or not patient.segmentation_labels_valid:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
