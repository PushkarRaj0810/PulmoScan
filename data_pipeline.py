
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_DIR = Path("chest_xray")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(split_dir):
    paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in VALID_EXTENSIONS
        )

        paths.extend(str(p) for p in files)
        labels.extend([label] * len(files))

    if not paths:
        raise ValueError(f"No images found in {split_dir}")

    return np.array(paths), np.array(labels, dtype=np.int32)


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])

        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)

        return image, tf.cast(tf.reshape(label, [1]), tf.float32)

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE)

    if training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.03, 0.03),
        ])

        ds = ds.map(
            lambda images, labels: (
                augmentation(images, training=True),
                labels,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)


def print_distribution(name, labels):
    counts = np.bincount(labels, minlength=2)

    print(f"\n{name}: {len(labels)} images")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {counts[class_id]}")


def load_datasets():
    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            "Expected chest_xray/train and chest_xray/test."
        )

    # Collect the original training images.
    all_train_paths, all_train_labels = collect_images(train_dir)

    # Split while preserving each class's proportion.
    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            all_train_paths,
            all_train_labels,
            test_size=0.2,
            random_state=SEED,
            stratify=all_train_labels,
        )
    )

    # Load the separate, untouched test set.
    test_paths, test_labels = collect_images(test_dir)

    print("Class mapping:", dict(enumerate(CLASS_NAMES)))
    print_distribution("TRAIN", train_labels)
    print_distribution("VALIDATION", val_labels)
    print_distribution("TEST", test_labels)

    train_ds = make_dataset(
        train_paths, train_labels, training=True
    )
    val_ds = make_dataset(
        val_paths, val_labels, training=False
    )
    test_ds = make_dataset(
        test_paths, test_labels, training=False
    )

    return train_ds, val_ds, test_ds, CLASS_NAMES.copy()
