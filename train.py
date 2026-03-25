"""Train a CNN for human emotion detection using FER2013."""

from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from src.config import BEST_CHECKPOINT_PATH, CLASS_NAMES, MODEL_PATH, MODELS_DIR, TEST_DIR, TRAIN_DIR, VALIDATION_DIR
from src.data_loader import create_data_generators
from src.model import build_emotion_cnn
from src.utils import ensure_directories, evaluate_and_save_reports, plot_training_history


def validate_dataset_structure():
    """Validate the expected dataset folder structure."""
    required_dirs = [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]
    missing_dirs = [str(directory) for directory in required_dirs if not directory.exists()]

    if missing_dirs:
        raise FileNotFoundError(
            "Missing dataset directories:\n"
            + "\n".join(missing_dirs)
            + "\n\nExpected structure:\n"
            + "dataset/train/<class_name>\n"
            + "dataset/validation/<class_name>\n"
            + "dataset/test/<class_name>"
        )

    for split_dir in required_dirs:
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")


def build_class_weights(train_generator):
    """Compute balanced class weights from the training split."""
    class_indices = train_generator.classes
    unique_classes = np.unique(class_indices)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=class_indices,
    )
    return {int(class_id): float(weight) for class_id, weight in zip(unique_classes, weights)}


def main():
    """Train, evaluate, and save the final model."""
    validate_dataset_structure()
    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_generator, validation_generator, test_generator = create_data_generators()
    model = build_emotion_cnn(num_classes=len(CLASS_NAMES))
    class_weights = build_class_weights(train_generator)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=BEST_CHECKPOINT_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("Using class weights:")
    for class_id, weight in class_weights.items():
        print(f"  {CLASS_NAMES[class_id]}: {weight:.4f}")

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    plot_training_history(history)

    if Path(BEST_CHECKPOINT_PATH).exists():
        model = load_model(BEST_CHECKPOINT_PATH)

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    report, _ = evaluate_and_save_reports(model, test_generator)

    model.save(MODEL_PATH)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print(f"\nSaved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
