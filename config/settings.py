from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ARCHIVE_PATH = Path(
    r"D:\BrainTumorDataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
)
DATASET_ROOT = Path(
    r"D:\BrainTumorDataset\extracted\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

MRI_MODALITIES = ("t1c", "t1n", "t2w", "t2f")
SEGMENTATION_MASK_SUFFIX = "seg"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
