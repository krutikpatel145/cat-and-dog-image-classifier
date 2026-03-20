import io
import streamlit as st
from PIL import Image

from inference import POSSIBLE_MODEL_PATHS, load_model, predict_image, preprocess_image, resolve_model_path


MODEL_PATH = resolve_model_path()
MODEL, MODEL_ERROR = load_model(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
    st.title("Cat vs Dog Image Classifier")
    st.write("Upload an image and the model will predict whether it is a cat or a dog.")

    if MODEL is None:
        st.error(
            (MODEL_ERROR or "Model file not found.")
            + " Please upload your trained .h5 model in one of these paths: "
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
        label, confidence, raw_score, cat_prob, dog_prob = predict_image(MODEL, img_array)

        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.4f}")
        st.write(f"Raw model score: {raw_score:.4f}")
        st.write(f"Cat probability: {cat_prob:.4f}")
        st.write(f"Dog probability: {dog_prob:.4f}")


if __name__ == "__main__":
    main()
