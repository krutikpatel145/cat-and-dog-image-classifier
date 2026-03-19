import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐾",
    layout="centered"
)

# ── Load model (cached so it only loads once) ──────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf

    # Try optimized model first, then fall back to main model
    model_paths = [
        "optimized_models/dog_cat_model_optimized.h5",
        "dog_cat_model.h5",
    ]
    for path in model_paths:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)

    st.error("❌ No model file found. Please ensure `dog_cat_model.h5` is in the repo.")
    st.stop()

# ── Prediction helper ──────────────────────────────────────────
def predict(model, image: Image.Image):
    IMG_SIZE = 150
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)          # shape: (1, 150, 150, 3)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "🐶 Dog" if prob >= 0.5 else "🐱 Cat"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence

# ── UI ─────────────────────────────────────────────────────────
st.title("🐾 Dog vs Cat Classifier")
st.markdown("Upload a photo and the model will tell you whether it's a **dog** or a **cat**.")

model = load_model()

uploaded = st.file_uploader(
    "Choose an image…",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Classifying…"):
            label, confidence = predict(model, image)

        st.markdown("### Result")
        st.markdown(f"## {label}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")

        color = "green" if confidence >= 0.80 else "orange" if confidence >= 0.60 else "red"
        st.progress(int(confidence * 100))

        if confidence < 0.65:
            st.warning("⚠️ Low confidence — try a clearer photo with a single pet.")

else:
    st.info("👆 Upload a JPG or PNG image of a cat or dog to get started.")

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with TensorFlow & Streamlit · [GitHub](https://github.com/krutikpatel145/cat-and-dog-image-classifier)")
