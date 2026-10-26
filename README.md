# Hybrid Quantum-Classical Brain Tumor Detection

## Purpose and status

This project provides MRI slice feature extraction, patient-separated dataset
preparation, and classical machine-learning baselines. The current implemented
pipeline is classical; the `models/quantum/` directory is reserved for future
work. It is an experimental development project, not a clinical system.

The current target is **slice-level mask presence**: a slice is labeled 1 when
its segmentation mask contains any nonzero voxel and 0 otherwise. This is not
clinical diagnosis, patient-level tumor detection, or tumor segmentation.

## Project structure

- `backend/data_loading.py` — loads and validates a patient's four MRI
  modalities and segmentation volume.
- `backend/dataset_builder.py` — assigns patients to train, validation, and
  test splits, then writes slice-level feature CSVs and a manifest.
- `feature_extraction/slice_features.py` — extracts four intensity statistics
  per modality (16 features per slice).
- `preprocessing/normalization.py` — normalization used by the dataset builder.
- `models/classical/train_baselines.py` — trains logistic-regression and random
  forest baselines and reports validation metrics.
- `tests/` — unit tests for feature extraction, normalization, dataset building,
  and the classical baselines.
- `scripts/` and top-level Python utilities — dataset download, verification,
  inspection, visualization, and preprocessing utilities.
- `config/settings.py` — MRI modality names and local data paths.

## Windows PowerShell setup

From the project root, create and activate a virtual environment, then install
the listed project dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `py -3.11` is unavailable, use the path to your installed Python 3.11
executable in place of `py -3.11`. If PowerShell blocks activation, use the
environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

When the environment is not activated, use
`.\.venv\Scripts\python.exe` in place of `python` in the commands below.

Run the test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

## Dataset access and local layout

MRI data is not included in this repository. Access the BraTS-GLI training data
through its authorized distribution (the download utility uses Synapse and
requires the appropriate account/access). Do not commit downloaded data or
credentials.

`config/settings.py` currently points to an archive at
`D:\BrainTumorDataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`
and extracted patient folders under
`D:\BrainTumorDataset\extracted\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`.
The extracted directory is expected to contain patient folders named like
`BraTS-GLI-<patient-id>`; each folder must contain the four configured MRI
modalities and its segmentation NIfTI file. These are absolute local paths, so
collaborators may need a local configuration adjustment before using
data-dependent commands. This repository setup does not change that source
configuration.

## Classical workflow

Run commands from the project root after configuring the dataset location and
obtaining the data. The dataset builder reads the configured patient volumes
one patient at a time and creates patient-level train, validation, and test
splits. Its CLI takes a **new output directory** and optionally a random seed:

```powershell
python -m backend.dataset_builder results\phase2_full_dataset_seed42_reproduction --seed 42
```

The output directory must not already exist; use another new directory name if
that path is already present. At least three patients are needed for
non-empty patient-separated train, validation, and test splits. The builder
writes `train.csv`, `validation.csv`, `test.csv`, and `manifest.json`.

Train the classical baselines using the generated dataset:

```powershell
python -m models.classical.train_baselines `
  --dataset-dir results\phase2_full_dataset_seed42_reproduction `
  --results-root results `
  --seed 42
```

The baseline command fits logistic regression and random forest on `train.csv`,
evaluates them on `validation.csv`, and stores model files plus a validation
report in a new timestamped directory under `results`. It does not load the
test split. Keep test data reserved for a separately planned final evaluation.
The dataset builder and baseline trainer implement these command-line
arguments and outputs.

## Data and generated files

MRI datasets, generated CSV datasets and manifests, trained model artifacts,
and experiment results are excluded from Git. They must be obtained or generated
locally and should remain outside commits.

The dependency list is intentionally unpinned: no dependency versions have
been declared or verified as a reproducible compatibility set for collaborator
machines.
