
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)
MODEL_PATH = Path("models/best_resnet50.keras")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # The trained model performs ResNet50 preprocessing internally.
    probability = float(model.predict(image_array, verbose=0)[0][0])

    predicted_class = (
        "PNEUMONIA" if probability >= 0.5 else "NORMAL"
    )

    print(f"\nImage: {image_path.name}")
    print(f"Predicted class: {predicted_class}")
    print(f"Model score for PNEUMONIA: {probability:.2%}")
    print("\nThis is an experimental model prediction, not a medical diagnosis.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: py predict.py "path/to/chest_xray.jpg"')
        raise SystemExit(1)

    predict_image(sys.argv[1])
