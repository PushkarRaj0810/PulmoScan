
from pathlib import Path
import hashlib
from io import BytesIO

import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

import gradcam

# ---------------- Configuration ----------------
LOGO_PATH = Path("pulmoscan_logo.jpg")
MAX_FILE_SIZE = 10 * 1024 * 1024

st.set_page_config(
    page_title="PulmoScan | Chest X-ray",
    page_icon="🩻",
    layout="wide",
)

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
    :root {
        --ink: #eaf4f7;
        --muted: #b5cbd5;
        --teal: #22b8c7;
        --teal-dark: #087f8c;
        --line: #31525f;
        --surface: #173944;
        --canvas: #102b35;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%,
                rgba(34,184,199,.16), transparent 34rem),
            var(--canvas);
        color: var(--ink);
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }

    [data-testid="stMainBlockContainer"] {
        max-width: 1450px;
        padding: 1rem 1.5rem 2rem;
    }

    h1, h2, h3, p, label {
        color: var(--ink);
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    h1, h2, h3 {
        margin-top: .35rem;
        margin-bottom: .35rem;
    }

    [data-testid="stCaptionContainer"] {
        margin-top: -.15rem;
    }

    .brand-name {
        color: var(--ink);
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -.04em;
        line-height: 1.15;
        margin-bottom: .1rem;
    }

    .brand-caption {
        color: var(--muted);
        font-size: .82rem;
    }

    .eyebrow {
        color: var(--teal);
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .14em;
        margin-top: .5rem;
        text-transform: uppercase;
    }

    .hero-title {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.1rem);
        font-weight: 800;
        letter-spacing: -.05em;
        line-height: 1.12;
        margin: .2rem 0;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: .75rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(23,57,68,.88);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 12px 32px rgba(0,0,0,.12);
        padding: .65rem .8rem;
    }

    [data-testid="stFileUploader"] section {
        background: #193e49;
        border: 1px dashed #507784;
        border-radius: 14px;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section * {
        color: var(--ink);
    }

    [data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 3rem;
    }

    [data-testid="stButton"] button[kind="primary"] {
        background: var(--teal-dark);
        border: 1px solid var(--teal-dark);
        color: white;
    }

    [data-testid="stButton"] button[kind="primary"]:hover {
        background: #096b76;
        border-color: #096b76;
    }

    .section-kicker {
        color: var(--teal);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
    }

    .empty-state {
        background: #153641;
        border: 1px dashed #426773;
        border-radius: 14px;
        color: var(--muted);
        padding: 1.1rem .8rem;
        text-align: center;
    }

    .empty-icon {
        color: var(--teal);
        font-size: 2rem;
        margin-bottom: .25rem;
    }

    .empty-title {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: .2rem;
    }

    .result-summary {
        align-items: center;
        background: #102f3a;
        border: 1px solid #315461;
        border-radius: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: .35rem 1.5rem;
        justify-content: space-between;
        margin: .45rem 0 .25rem;
        padding: .6rem .9rem;
    }

    .result-summary-label,
    .result-summary-score-label {
        color: var(--muted);
        display: block;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
    }

    .result-class {
        font-size: 1.35rem;
        font-weight: 850;
        letter-spacing: -.04em;
        line-height: 1.2;
        margin-top: .1rem;
    }

    .result-class.normal { color: #58d6a1; }
    .result-class.pneumonia { color: #ff9b91; }

    .score-card-value {
        color: var(--ink);
        display: block;
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.2;
        margin-top: .1rem;
    }

    .score-caption {
        color: var(--muted);
        font-size: .82rem;
        line-height: 1.6;
        margin: .3rem 0 .55rem;
    }

    .result-notice {
        background: #45391e;
        border: 1px solid #8c7134;
        border-radius: 12px;
        color: #ffe8b0;
        font-size: .85rem;
        line-height: 1.6;
        margin-top: .55rem;
        padding: .65rem .8rem;
    }

    .result-notice strong { color: #fff0c9; }

    .footer-copy {
        color: #8faab5;
        font-size: .78rem;
        line-height: 1.6;
        margin-top: .75rem;
        text-align: center;
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: .75rem .65rem 1.25rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: .5rem .55rem;
        }

        .result-summary {
            justify-content: flex-start;
        }

        .hero-title {
            font-size: clamp(1.8rem, 8vw, 2.5rem);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    """Load the classifier once per Streamlit server process."""
    if not gradcam.MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {gradcam.MODEL_PATH.resolve()}"
        )
    return tf.keras.models.load_model(gradcam.MODEL_PATH)


@st.cache_resource
def load_gradcam_models():
    """Cache the graph pieces used for gradients alongside the classifier."""
    return gradcam.build_gradcam_models(load_model())


# ---------------- Header ----------------
logo_col, brand_col = st.columns([1, 5], vertical_alignment="center")

with logo_col:
    if LOGO_PATH.is_file():
        st.image(str(LOGO_PATH), width=82)
    else:
        st.markdown("## 🩻")

with brand_col:
    st.markdown(
        '<div class="brand-name">PulmoScan</div>'
        '<div class="brand-caption">Chest X-ray research demo</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="eyebrow">Chest X-ray image exploration</div>
    <h1 class="hero-title">A clearer view of your chest X-ray.</h1>
    <p class="hero-copy">
        Upload a chest X-ray to explore the output of our experimental
        ResNet50 model. This project is intended for education and research.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------- Main Interface ----------------
upload_col, preview_col = st.columns([1.35, 1], gap="medium")

with upload_col:
    with st.container(border=True):
        st.markdown(
            '<div class="section-kicker">Step 1 · Add an image</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Upload a chest X-ray")
        uploaded_file = st.file_uploader(
            "Choose a JPG or PNG image",
            type=["jpg", "jpeg", "png"],
            key="xray_upload",
        )

        image = None
        analysis_image = None
        image_array = None
        file_id = None

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_id = hashlib.sha256(file_bytes).hexdigest()

            if len(file_bytes) > MAX_FILE_SIZE:
                st.error("Please upload an image smaller than 10 MB.")
            else:
                try:
                    with Image.open(BytesIO(file_bytes)) as source:
                        source.verify()

                    with Image.open(BytesIO(file_bytes)) as source:
                        image = source.convert("RGB")

                    analysis_image, image_array = gradcam.prepare_xray(image)
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    st.error(f"This file is not a valid image: {error}")
                    image = None
                    image_array = None

with preview_col:
    with st.container(border=True):
        st.markdown(
            '<div class="section-kicker">Image preview</div>',
            unsafe_allow_html=True,
        )
        if image is not None:
            st.markdown(
                f'<div class="empty-title">{uploaded_file.name}</div>',
                unsafe_allow_html=True,
            )
            st.image(image, width="stretch")
            st.caption(f"{image.width} × {image.height} px")
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-icon">+</div>
                    <div class="empty-title">Your X-ray preview</div>
                    Choose an image to get started.
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="section-kicker">Step 2 · Review output</div>',
    unsafe_allow_html=True,
)
st.subheader("Model analysis")

if image is None or image_array is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">Ready when you are</div>
            Upload an X-ray to view the model output and visualization.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    result = st.session_state.get("pulmoscan_result")
    if not result or result["file_id"] != file_id:
        with st.spinner("Running model analysis and generating Grad-CAM..."):
            try:
                model = load_model()
            except (
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                tf.errors.OpError,
            ) as error:
                st.error(f"PulmoScan could not load its model: {error}")
                model = None

            if model is not None:
                heatmap = None
                heatmap_image = None
                overlay_image = None
                gradcam_error = None
                score = None

                try:
                    heatmap, score = gradcam.generate_gradcam(
                        model,
                        image_array,
                        gradcam_models=load_gradcam_models(),
                    )
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    tf.errors.OpError,
                ) as error:
                    gradcam_error = str(error)
                    if isinstance(error, gradcam.GradCAMError):
                        score = error.score

                if score is None:
                    try:
                        score = float(
                            model.predict(image_array, verbose=0)[0][0]
                        )
                    except (
                        IndexError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        tf.errors.OpError,
                    ) as error:
                        st.error(
                            "PulmoScan could not analyze this image: "
                            f"{error}"
                        )
                        score = None

                if score is not None and heatmap is not None:
                    try:
                        heatmap_image, overlay_image = (
                            gradcam.create_visualizations(
                                analysis_image,
                                heatmap,
                            )
                        )
                    except (
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as error:
                        gradcam_error = (
                            "The prediction is available, but the "
                            f"visualization could not be prepared: {error}"
                        )

                if score is not None:
                    st.session_state["pulmoscan_result"] = {
                        "file_id": file_id,
                        "score": score,
                        "heatmap_image": heatmap_image,
                        "overlay_image": overlay_image,
                        "gradcam_error": gradcam_error,
                    }
                    result = st.session_state["pulmoscan_result"]

    if result and result["file_id"] == file_id:
        score = result["score"]
        prediction = gradcam.CLASS_NAMES[int(score >= 0.5)]
        style = "pneumonia" if prediction == "PNEUMONIA" else "normal"
        st.markdown(
            f"""
            <div class="result-summary">
                <div>
                    <span class="result-summary-label">Model prediction</span>
                    <div class="result-class {style}">{prediction}</div>
                </div>
                <div>
                    <span class="result-summary-score-label">
                        PNEUMONIA model score
                    </span>
                    <strong class="score-card-value">{score:.1%}</strong>
                </div>
            </div>
            <p class="score-caption">
                The score is the model's sigmoid output and is not necessarily
                a calibrated probability of disease.
            </p>
            <div class="result-notice">
                <strong>⚠ Experimental AI — not a medical diagnosis.</strong>
                This output is not clinically validated and must not replace
                advice from a qualified healthcare professional.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["gradcam_error"]:
            st.warning(
                "The prediction is available, but Grad-CAM could not "
                f"be displayed: {result['gradcam_error']}"
            )

        original_col, heatmap_col, overlay_col = st.columns(3, gap="medium")
        with original_col:
            st.markdown("**Original X-ray**")
            st.image(image, caption=uploaded_file.name, width="stretch")
        with heatmap_col:
            st.markdown(f"**Grad-CAM · {prediction}**")
            if result["heatmap_image"] is not None:
                st.image(
                    result["heatmap_image"],
                    caption="Model visualization, not medical evidence",
                    width="stretch",
                )
            else:
                st.info("The Grad-CAM heatmap is unavailable.")
        with overlay_col:
            st.markdown("**Overlay**")
            if result["overlay_image"] is not None:
                st.image(
                    result["overlay_image"],
                    caption="Model visualization, not medical evidence",
                    width="stretch",
                )
            else:
                st.info("The Grad-CAM overlay is unavailable.")

st.markdown(
    """
    <div class="footer-copy">
        PulmoScan · Experimental research demo · Educational use only
    </div>
    """,
    unsafe_allow_html=True,
)
