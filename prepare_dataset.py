"""Download FER2013 from Kaggle and prepare train/validation/test folders."""

import argparse
import random
import shutil
from pathlib import Path

import kagglehub

from src.config import CLASS_NAMES, DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare the FER2013 dataset.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of training images to copy into the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic validation sampling.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing dataset/train, dataset/validation, and dataset/test folders first.",
    )
    return parser.parse_args()


def reset_target_dirs(force: bool):
    targets = [DATASET_DIR / "train", DATASET_DIR / "validation", DATASET_DIR / "test"]
    if force:
        for target in targets:
            if target.exists():
                shutil.rmtree(target)

    for split in targets:
        split.mkdir(parents=True, exist_ok=True)
        for class_name in CLASS_NAMES:
            (split / class_name).mkdir(parents=True, exist_ok=True)


def copy_files(file_paths, destination_dir: Path):
    for file_path in file_paths:
        shutil.copy2(file_path, destination_dir / file_path.name)


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_path = Path(kagglehub.dataset_download("msambare/fer2013"))
    print(f"Downloaded dataset to: {dataset_path}")

    source_train = dataset_path / "train"
    source_test = dataset_path / "test"
    target_train = DATASET_DIR / "train"
    target_validation = DATASET_DIR / "validation"
    target_test = DATASET_DIR / "test"

    reset_target_dirs(force=args.force)

    for class_name in CLASS_NAMES:
        train_files = sorted((source_train / class_name).glob("*"))
        test_files = sorted((source_test / class_name).glob("*"))

        if not train_files:
            raise FileNotFoundError(f"No training files found for class: {class_name}")
        if not test_files:
            raise FileNotFoundError(f"No test files found for class: {class_name}")

        validation_count = max(1, int(len(train_files) * args.validation_split))
        validation_files = set(random.sample(train_files, validation_count))
        final_train_files = [file_path for file_path in train_files if file_path not in validation_files]

        copy_files(final_train_files, target_train / class_name)
        copy_files(validation_files, target_validation / class_name)
        copy_files(test_files, target_test / class_name)

        print(
            f"{class_name}: "
            f"train={len(final_train_files)}, "
            f"validation={len(validation_files)}, "
            f"test={len(test_files)}"
        )

    print(f"Prepared dataset under: {DATASET_DIR}")


if __name__ == "__main__":
    main()
