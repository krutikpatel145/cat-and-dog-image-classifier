import io
import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image

POSSIBLE_MODEL_PATHS = [
    "optimized_models/model_optimized.h5",
    "dog_cat_model.h5",
    "dogs_vs_cats_model.h5",
    "dogs_vs_cats_simple_cnn.h5",
]

IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = ["Cat", "Dog"]


def resolve_model_path() -> str | None:
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


MODEL_PATH = resolve_model_path()
MODEL = keras.models.load_model(MODEL_PATH) if MODEL_PATH else None


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_image(img_array: np.ndarray) -> tuple[str, float, float]:
    raw = float(MODEL.predict(img_array, verbose=0)[0][0])
    idx = int(raw > 0.5)
    label = CLASS_NAMES[idx]
    confidence = raw if idx == 1 else (1.0 - raw)
    return label, float(confidence), raw


def main() -> None:
    st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
    st.title("Cat vs Dog Image Classifier")
    st.write("Upload an image and the model will predict whether it is a cat or a dog.")

    if MODEL is None:
        st.error(
            "Model file not found. Please upload your trained .h5 model in one of these paths: "
            + ", ".join(POSSIBLE_MODEL_PATHS)
        )
        st.stop()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(io.BytesIO(uploaded_file.read()))
        except Exception:
            st.error("Invalid image file. Please upload a valid JPG/PNG image.")
            return

        st.image(img, caption="Uploaded image", use_column_width=True)
        img_array = preprocess_image(img)
        label, confidence, raw_score = predict_image(img_array)

        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.4f}")
        st.write(f"Raw model score: {raw_score:.4f}")


if __name__ == "__main__":
    main()
