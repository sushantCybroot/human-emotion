"""Shared helper utilities."""

from pathlib import Path

import cv2
import numpy as np

from src.config import CLASS_NAMES, IMAGE_SIZE, PLOTS_DIR, REPORTS_DIR


def ensure_directories():
    """Create output directories if they do not exist."""
    for directory in [PLOTS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def preprocess_image_array(image_array: np.ndarray) -> np.ndarray:
    """Normalize and reshape image array for prediction."""
    resized = cv2.resize(image_array, IMAGE_SIZE)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """Load an image from disk, convert to grayscale, and prepare for inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return preprocess_image_array(grayscale)


def plot_training_history(history):
    """Save training accuracy and loss curves."""
    import matplotlib.pyplot as plt

    ensure_directories()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_and_save_reports(model, test_generator):
    """Evaluate the trained model and save reports."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    ensure_directories()

    predictions = model.predict(test_generator, verbose=1)
    predicted_indices = np.argmax(predictions, axis=1)
    true_indices = test_generator.classes

    report = classification_report(
        true_indices,
        predicted_indices,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    report_path = REPORTS_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(true_indices, predicted_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return report, cm


def predict_label(model, processed_image: np.ndarray):
    """Return predicted emotion label and confidence."""
    predictions = model.predict(processed_image, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    return CLASS_NAMES[predicted_index], confidence
