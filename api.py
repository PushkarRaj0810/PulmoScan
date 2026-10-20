
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

MODEL_PATH = Path("models/best_resnet50.keras")
IMAGE_SIZE = (224, 224)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="PulmoScan API",
    description="Experimental chest X-ray classification service.",
    version="1.0.0",
)


@app.on_event("startup")
def load_model():
    """Load the trained model once when the API starts."""
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    app.state.model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/")
def home():
    return {
        "app": "PulmoScan",
        "message": "Chest X-ray classification API is running.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept a JPG/PNG image and return the model prediction."""
    allowed_types = {"image/jpeg", "image/png"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG or PNG image.",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB size limit.",
        )

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        with Image.open(BytesIO(contents)) as uploaded_image:
            uploaded_image.verify()

        with Image.open(BytesIO(contents)) as uploaded_image:
            image = uploaded_image.convert("RGB").resize(IMAGE_SIZE)

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    try:
        probability = float(
            app.state.model.predict(image_array, verbose=0)[0][0]
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The model could not process this image.",
        )

    prediction = "PNEUMONIA" if probability >= 0.5 else "NORMAL"

    return {
        "model": "ResNet50",
        "prediction": prediction,
        "pneumonia_score": round(probability, 6),
        "disclaimer": (
            "Experimental output only; not a medical diagnosis."
        ),
    }
