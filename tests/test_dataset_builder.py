"""Synthetic tests for patient-level slice dataset preparation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.dataset_builder import (
    CSV_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    DatasetBuildError,
    build_dataset,
    split_patients,
)
from config import settings


MODALITIES = settings.MRI_MODALITIES
RATIOS = {"train": 0.5, "validation": 0.25, "test": 0.25}


def make_patient(patient_id: str, target_mode: str = "mixed") -> SimpleNamespace:
    shape = (2, 2, 2)
    volumes = {
        modality: np.arange(1, 9, dtype=np.int16).reshape(shape)
        + modality_index * 10
        for modality_index, modality in enumerate(MODALITIES)
    }
    segmentation = np.zeros(shape, dtype=np.uint8)
    if target_mode == "positive_only":
        segmentation[0, 0, :] = 1
    elif target_mode == "mixed":
        segmentation[0, 0, 1] = 3
    elif target_mode != "negative_only":
        raise ValueError(f"Unknown target mode: {target_mode}")
    volumes[settings.SEGMENTATION_MASK_SUFFIX] = segmentation
    return SimpleNamespace(
        patient_id=patient_id,
        volumes=volumes,
        dimensions_compatible=True,
        segmentation_labels_valid=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class DatasetBuilderTests(unittest.TestCase):
    def test_patient_assignments_are_deterministic_for_seed(self) -> None:
        patients = [f"BraTS-GLI-{index:05d}-000" for index in range(12)]
        first = split_patients(patients, seed=123, split_ratios=RATIOS)
        second = split_patients(patients, seed=123, split_ratios=RATIOS)
        self.assertEqual(first, second)

    def test_partitions_are_disjoint_and_assign_each_patient_once(self) -> None:
        patients = [f"patient-{index}" for index in range(9)]
        partitions = split_patients(patients, seed=5, split_ratios=RATIOS)
        assigned = [
            patient_id
            for partition in partitions.values()
            for patient_id in partition
        ]
        self.assertEqual(len(assigned), len(patients))
        self.assertEqual(set(assigned), set(patients))
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertTrue(all(partitions[name] for name in partitions))

    def test_rejects_too_few_patients_and_invalid_ratios(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least three patients"):
            split_patients(["p1", "p2"], seed=1)
        for invalid in (
            {"train": 0.7, "validation": 0.2, "test": 0.2},
            {"train": 1.0, "validation": 0.0, "test": 0.0},
            {"train": 0.7, "validation": 0.3},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    split_patients(["p1", "p2", "p3"], split_ratios=invalid)

    def test_builds_ordered_csv_with_aligned_rows_and_labels(self) -> None:
        patient_modes = {
            "BraTS-GLI-00001-000": "mixed",
            "BraTS-GLI-00002-000": "negative_only",
            "BraTS-GLI-00003-000": "positive_only",
            "BraTS-GLI-00004-000": "mixed",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "patients"
            root.mkdir()
            for patient_id in patient_modes:
                (root / patient_id).mkdir()
            output = Path(temporary) / "output"

            build_dataset(
                output,
                seed=17,
                split_ratios=RATIOS,
                dataset_root=root,
                patient_loader=lambda patient_id: make_patient(
                    patient_id, patient_modes[patient_id]
                ),
            )

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["csv_columns"], list(CSV_COLUMNS))
            self.assertEqual(
                manifest["feature_columns"], list(MODEL_FEATURE_COLUMNS)
            )
            self.assertEqual(
                CSV_COLUMNS[:3], ("patient_id", "slice_index", "target")
            )
            self.assertEqual(len(MODEL_FEATURE_COLUMNS), 16)
            self.assertFalse(
                {"patient_id", "slice_index", "target"}
                & set(manifest["feature_columns"])
            )

            all_rows: list[dict[str, str]] = []
            for split_name in ("train", "validation", "test"):
                rows = read_csv(output / f"{split_name}.csv")
                self.assertEqual(
                    list(rows[0]) if rows else list(CSV_COLUMNS),
                    list(CSV_COLUMNS),
                )
                assigned_patients = set(
                    manifest["patient_assignments"][split_name]
                )
                self.assertTrue(
                    all(row["patient_id"] in assigned_patients for row in rows)
                )
                all_rows.extend(rows)

            self.assertEqual(len(all_rows), 8)
            rows_by_patient: dict[str, list[dict[str, str]]] = {}
            for row in all_rows:
                rows_by_patient.setdefault(row["patient_id"], []).append(row)
            for patient_id, rows in rows_by_patient.items():
                self.assertEqual(
                    [int(row["slice_index"]) for row in rows], [0, 1]
                )
                mode = patient_modes[patient_id]
                targets = [int(row["target"]) for row in rows]
                if mode == "positive_only":
                    self.assertEqual(targets, [1, 1])
                elif mode == "negative_only":
                    self.assertEqual(targets, [0, 0])
                else:
                    self.assertEqual(targets, [0, 1])
                raw = np.arange(1, 9, dtype=np.float64).reshape((2, 2, 2))
                normalized = (raw - raw.mean()) / raw.std()
                for row in rows:
                    slice_index = int(row["slice_index"])
                    slice_values = normalized[:, :, slice_index]
                    lower, upper = np.percentile(
                        slice_values, (25, 75), method="linear"
                    )
                    expected_modality_features = (
                        slice_values.mean(),
                        slice_values.std(),
                        np.median(slice_values),
                        upper - lower,
                    )
                    expected_features = np.tile(
                        expected_modality_features, len(MODALITIES)
                    )
                    actual_features = np.array(
                        [float(row[column]) for column in MODEL_FEATURE_COLUMNS]
                    )
                    np.testing.assert_allclose(
                        actual_features, expected_features, atol=1e-7
                    )

            expected_counts = {
                split_name: {
                    "zero_mask": sum(int(row["target"]) == 0 for row in read_csv(output / f"{split_name}.csv")),
                    "mask_positive": sum(int(row["target"]) == 1 for row in read_csv(output / f"{split_name}.csv")),
                }
                for split_name in ("train", "validation", "test")
            }
            self.assertEqual(
                manifest["per_split_label_counts"], expected_counts
            )

    def test_missing_or_invalid_patient_fails_without_publishing_output(self) -> None:
        patients = ["BraTS-GLI-00001-000", "BraTS-GLI-00002-000", "BraTS-GLI-00003-000"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "patients"
            root.mkdir()
            for patient_id in patients:
                (root / patient_id).mkdir()
            output = Path(temporary) / "output"

            def missing_loader(patient_id: str) -> SimpleNamespace:
                if patient_id == patients[0]:
                    raise FileNotFoundError("missing t1c file")
                return make_patient(patient_id)

            with self.assertRaisesRegex(DatasetBuildError, "missing t1c file"):
                build_dataset(
                    output,
                    seed=1,
                    dataset_root=root,
                    patient_loader=missing_loader,
                )
            self.assertFalse(output.exists())

            def invalid_loader(patient_id: str) -> SimpleNamespace:
                patient = make_patient(patient_id)
                if patient_id == patients[0]:
                    patient.segmentation_labels_valid = False
                return patient

            with self.assertRaisesRegex(
                DatasetBuildError, "invalid or unsupported labels"
            ):
                build_dataset(
                    output,
                    seed=1,
                    dataset_root=root,
                    patient_loader=invalid_loader,
                )
            self.assertFalse(output.exists())

    def test_patient_with_only_one_class_remains_whole_in_its_split(self) -> None:
        patient_modes = {
            "BraTS-GLI-00001-000": "positive_only",
            "BraTS-GLI-00002-000": "negative_only",
            "BraTS-GLI-00003-000": "mixed",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "patients"
            root.mkdir()
            for patient_id in patient_modes:
                (root / patient_id).mkdir()
            output = Path(temporary) / "output"
            build_dataset(
                output,
                seed=2,
                dataset_root=root,
                patient_loader=lambda patient_id: make_patient(
                    patient_id, patient_modes[patient_id]
                ),
            )

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            patient_to_split = {
                patient_id: split_name
                for split_name, ids in manifest["patient_assignments"].items()
                for patient_id in ids
            }
            for patient_id, mode in patient_modes.items():
                rows = read_csv(
                    output / f"{patient_to_split[patient_id]}.csv"
                )
                patient_rows = [
                    row for row in rows if row["patient_id"] == patient_id
                ]
                self.assertEqual(len(patient_rows), 2)
                expected_target = 1 if mode == "positive_only" else 0
                if mode != "mixed":
                    self.assertEqual(
                        [int(row["target"]) for row in patient_rows],
                        [expected_target, expected_target],
                    )


if __name__ == "__main__":
    unittest.main()
