"""Predict the emotion of a single face image."""

import argparse

from tensorflow.keras.models import load_model

from src.config import MODEL_PATH
from src.utils import load_and_preprocess_image, predict_label


def parse_args():
    parser = argparse.ArgumentParser(description="Predict emotion from a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to the trained model file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model)
    processed_image = load_and_preprocess_image(args.image)
    label, confidence = predict_label(model, processed_image)

    print(f"Predicted Emotion: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()

