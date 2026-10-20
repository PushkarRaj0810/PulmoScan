
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_resnet50.keras"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "gradcam"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


class GradCAMError(RuntimeError):
    """Grad-CAM failure that can retain a prediction already computed."""

    def __init__(self, message, score=None):
        super().__init__(message)
        self.score = score


def prepare_xray(image):
    """Prepare a PIL image with the same preprocessing used by api.py."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)[np.newaxis, ...]
    return image, image_array


def load_xray(image_path):
    """Load an X-ray using the same resizing path as api.py."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")

    try:
        with Image.open(image_path) as source:
            image, image_array = prepare_xray(source)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read image {image_path}: {exc}") from exc

    return image, image_array


def build_gradcam_models(model):
    """
    Build models for Grad-CAM while preserving the original
    preprocessing layers of the complete classifier.
    """
    try:
        backbone = model.get_layer("resnet50")
    except ValueError as exc:
        raise ValueError(
            "Could not find the nested 'resnet50' model. "
            "Check the loaded model architecture."
        ) from exc

    # Final convolutional feature map in ResNet50.
    target_layer = backbone.get_layer("conv5_block3_out")
    if len(target_layer.output.shape) != 4:
        raise RuntimeError(
            "Expected conv5_block3_out to produce a 4D feature map, "
            f"got shape {target_layer.output.shape}."
        )

    # Find the tensor that enters ResNet50 in the outer model graph.
    nodes = backbone._inbound_nodes

    if not nodes:
        raise RuntimeError("The ResNet50 backbone has no inbound nodes.")

    inbound_tensors = tf.nest.flatten(nodes[0].input_tensors)

    if len(inbound_tensors) != 1:
        raise RuntimeError(
            "Expected one input tensor entering the ResNet50 backbone."
        )

    backbone_input_tensor = inbound_tensors[0]

    # Preserve the complete preprocessing pipeline.
    preprocess_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=backbone_input_tensor,
    )

    # Return the final convolutional activations and backbone output.
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[
            target_layer.output,
            backbone.output,
        ],
    )

    # Classifier layers after the backbone.
    backbone_index = model.layers.index(backbone)
    head_layers = model.layers[backbone_index + 1:]

    if not head_layers:
        raise RuntimeError("No classifier layers were found after ResNet50.")

    return preprocess_model, backbone_grad_model, head_layers


def generate_gradcam(model, image_array, gradcam_models=None):
    """Generate a Grad-CAM heatmap for the predicted binary class."""
    image_array = np.asarray(image_array, dtype=np.float32)
    if image_array.shape != (1, *IMAGE_SIZE, 3):
        raise ValueError(
            "Expected one RGB image with shape "
            f"(1, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}, 3), "
            f"got {image_array.shape}."
        )
    if not np.isfinite(image_array).all():
        raise ValueError("Image contains non-finite pixel values.")

    if gradcam_models is None:
        gradcam_models = build_gradcam_models(model)
    preprocess_model, backbone_grad_model, head_layers = gradcam_models

    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Apply the same preprocessing as the original model.
        processed_image = preprocess_model(
            image_tensor,
            training=False,
        )

        # Extract convolutional activations.
        conv_outputs, features = backbone_grad_model(
            processed_image,
            training=False,
        )

        # Run the original classification head.
        predictions = features

        for layer in head_layers:
            predictions = layer(predictions, training=False)

        # Binary classifier output: one sigmoid score per image.
        score = predictions[:, 0]
        class_score = tf.where(score >= 0.5, score, 1.0 - score)

    gradients = tape.gradient(class_score, conv_outputs)

    score_value = float(score[0].numpy())
    if gradients is None:
        raise GradCAMError(
            "Gradients are disconnected. Grad-CAM could not be generated.",
            score=score_value,
        )
    if not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):
        raise GradCAMError(
            "Grad-CAM gradients contain non-finite values.",
            score=score_value,
        )

    # Average gradients across height and width.
    weights = tf.reduce_mean(gradients, axis=(1, 2))

    # Weighted combination of the convolutional feature maps.
    heatmap = tf.reduce_sum(
        conv_outputs[0] * weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    # Keep positive contributions for the class predicted by the model.
    heatmap = tf.maximum(heatmap, 0)
    heatmap_max = tf.reduce_max(heatmap)
    if not bool(tf.math.is_finite(heatmap_max).numpy()):
        raise GradCAMError(
            "Grad-CAM heatmap contains non-finite values.",
            score=score_value,
        )
    if float(heatmap_max.numpy()) <= 0:
        raise GradCAMError(
            "Grad-CAM heatmap is empty; the predicted class has no "
            "positive activation at the selected feature map.",
            score=score_value,
        )
    heatmap = heatmap / heatmap_max

    return heatmap.numpy(), score_value


def create_visualizations(original_image, heatmap):
    """Create the colored heatmap and overlay as in-memory PIL images."""
    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    ).resize(
        original_image.size,
        Image.Resampling.BILINEAR,
    )
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    colored_heatmap = (
        plt.get_cmap("jet")(heatmap_array)[..., :3] * 255
    ).astype(np.uint8)
    original_array = np.asarray(
        original_image.convert("RGB"),
        dtype=np.float32,
    )
    overlay = np.clip(
        0.55 * original_array
        + 0.45 * colored_heatmap.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(colored_heatmap), Image.fromarray(overlay)


def save_visualization(
    original_image,
    heatmap,
    score,
    output_path,
):
    """Save the original image, heatmap, and overlay."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap_image, overlay_image = create_visualizations(
        original_image,
        heatmap,
    )

    predicted_class = CLASS_NAMES[int(score >= 0.5)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original X-ray")

    axes[1].imshow(heatmap_image)
    axes[1].set_title(f"Grad-CAM for {predicted_class}")

    axes[2].imshow(overlay_image)
    axes[2].set_title("Grad-CAM Overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"Prediction: {predicted_class} | "
        f"PNEUMONIA score: {score:.4f}",
        fontsize=13,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPrediction: {predicted_class}")
    print(f"Model score: {score:.4f}")
    print(f"Grad-CAM saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM for a chest X-ray."
    )

    parser.add_argument(
        "image",
        help="Path to the chest X-ray image",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH.resolve()}"
            )

        print("Loading PulmoScan model...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model {MODEL_PATH.resolve()}: {exc}"
            ) from exc

        print("Loading X-ray...")
        original_image, image_array = load_xray(image_path)

        print("Generating Grad-CAM...")
        heatmap, score = generate_gradcam(model, image_array)

        # Verify that the Grad-CAM classifier head matches the complete model.
        reference_score = float(model.predict(image_array, verbose=0)[0][0])
        score_delta = abs(score - reference_score)
        if not np.isclose(score, reference_score, rtol=1e-5, atol=1e-6):
            raise RuntimeError(
                "Grad-CAM prediction differs from the complete model: "
                f"{score:.8f} vs {reference_score:.8f} "
                f"(delta {score_delta:.2e})."
            )

        output_path = (
            Path(args.output)
            if args.output
            else OUTPUT_DIR / f"{image_path.stem}_gradcam.png"
        )

        save_visualization(
            original_image,
            heatmap,
            score,
            output_path,
        )
        print(
            "Prediction parity verified: "
            f"Grad-CAM={score:.8f}, model={reference_score:.8f}, "
            f"delta={score_delta:.2e}"
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
