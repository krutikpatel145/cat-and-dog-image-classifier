import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
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


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB


@app.get("/")
def index():
    return render_template("index.html", model_loaded=MODEL is not None, model_path=MODEL_PATH)


@app.post("/predict")
def predict():
    if MODEL is None:
        return jsonify({"error": "Model file not found. Please place a trained .h5 model in the project."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image'."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    try:
        content = file.read()
        img = Image.open(io.BytesIO(content))
    except Exception:
        return jsonify({"error": "Invalid image file."}), 400

    img_array = preprocess_image(img)
    label, confidence, raw_score = predict_image(img_array)

    return jsonify({"label": label, "confidence": confidence, "raw_score": raw_score})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
