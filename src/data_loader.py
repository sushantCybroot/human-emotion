"""Dataset loading and preprocessing utilities."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import BATCH_SIZE, CLASS_NAMES, IMAGE_SIZE, TEST_DIR, TRAIN_DIR, VALIDATION_DIR


def create_data_generators():
    """Create train, validation, and test generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )

    validation_generator = eval_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    test_generator = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_generator, validation_generator, test_generator

