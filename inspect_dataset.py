
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

# Dataset directory
DATA_DIR = Path("chest_xray")

# Check dataset structure
if not DATA_DIR.exists():
    raise FileNotFoundError(
        "chest_xray folder not found. "
        "Place it inside the Medical Diagnosis project folder."
    )

print("TensorFlow version:", tf.__version__)

# Count images in each split and class
for split in ["train", "val", "test"]:
    print(f"\n{split.upper()} DATASET")

    for label in ["NORMAL", "PNEUMONIA"]:
        folder = DATA_DIR / split / label

        if folder.exists():
            images = [
                file for file in folder.iterdir()
                if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            print(f"{label}: {len(images)} images")
        else:
            print(f"{label}: folder not found")

# Display sample images
train_dir = DATA_DIR / "train"

normal_dir = train_dir / "NORMAL"
pneumonia_dir = train_dir / "PNEUMONIA"

normal_images = list(normal_dir.glob("*"))
pneumonia_images = list(pneumonia_dir.glob("*"))

if normal_images and pneumonia_images:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, image_path, label in [
        (axes[0], normal_images[0], "NORMAL"),
        (axes[1], pneumonia_images[0], "PNEUMONIA"),
    ]:
        image = tf.keras.utils.load_img(image_path)
        ax.imshow(image, cmap="gray")
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("Could not find sample images in both classes.")
