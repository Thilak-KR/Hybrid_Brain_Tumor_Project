"""Verify Phase 2 project configuration and dataset paths without modifying data."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import settings
except ImportError:
    print("FAIL: Import config.settings from the project root")
    raise SystemExit(1)


results: list[bool] = []


def report(label: str, passed: bool, details: str) -> None:
    """Print one check result and record it for the final status."""
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {label} - {details}")


configured_project_root = Path(settings.PROJECT_ROOT)
report(
    "Project root",
    configured_project_root == PROJECT_ROOT and configured_project_root.is_dir(),
    str(configured_project_root),
)

archive_path = Path(settings.DATASET_ARCHIVE_PATH)
report(
    "ZIP archive",
    archive_path.is_file(),
    str(archive_path),
)

dataset_root = Path(settings.DATASET_ROOT)
dataset_exists = dataset_root.is_dir()
report(
    "Extracted dataset directory",
    dataset_exists,
    str(dataset_root),
)

results_dir = Path(settings.RESULTS_DIR)
report(
    "Results directory",
    results_dir.is_dir(),
    str(results_dir),
)

models_dir = Path(settings.MODELS_DIR)
report(
    "Models directory",
    models_dir.is_dir(),
    str(models_dir),
)

patient_dirs = (
    sorted(
        path
        for path in dataset_root.glob("BraTS-GLI-*")
        if path.is_dir()
    )
    if dataset_exists
    else []
)
patient_dir = patient_dirs[0] if patient_dirs else None
report(
    "Patient folder",
    patient_dir is not None,
    str(patient_dir) if patient_dir is not None else "No BraTS-GLI patient folder found",
)

expected_suffixes = (*settings.MRI_MODALITIES, settings.SEGMENTATION_MASK_SUFFIX)
for suffix in expected_suffixes:
    expected_file = (
        patient_dir / f"{patient_dir.name}-{suffix}.nii.gz"
        if patient_dir is not None
        else None
    )
    report(
        f"Patient {suffix} file",
        expected_file is not None and expected_file.is_file(),
        (
            str(expected_file)
            if expected_file is not None
            else "Cannot check because no patient folder was found"
        ),
    )

overall_passed = all(results)
print("Overall: " + ("PASS" if overall_passed else "FAIL"))
raise SystemExit(0 if overall_passed else 1)
