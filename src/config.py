"""Project-wide configuration values."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

IMAGE_SIZE = (48, 48)
INPUT_SHAPE = (48, 48, 1)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]
MODEL_FILENAME = "emotion_model.h5"
MODEL_PATH = PROJECT_ROOT / MODEL_FILENAME
BEST_CHECKPOINT_PATH = MODELS_DIR / "best_emotion_model.keras"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

