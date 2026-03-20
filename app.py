import io
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from inference import load_model, predict_image, preprocess_image, resolve_model_path


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

MODEL_PATH = resolve_model_path()
MODEL, MODEL_ERROR = load_model(MODEL_PATH)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_e: RequestEntityTooLarge):
    return jsonify({"error": "File too large. Max size is 10MB."}), 413


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_loaded=MODEL is not None,
        model_path=MODEL_PATH,
        model_error=MODEL_ERROR,
    )


@app.post("/predict")
def predict():
    if MODEL is None:
        msg = MODEL_ERROR or "Model file not found. Please place a trained .h5 model in the project."
        return jsonify({"error": msg}), 500

    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image'."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    try:
        content = file.read()
        img = Image.open(io.BytesIO(content))
        img.load()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Invalid image file."}), 400

    img_array = preprocess_image(img)
    label, confidence, raw_score, cat_prob, dog_prob = predict_image(MODEL, img_array)

    return jsonify(
        {
            "label": label,
            "confidence": confidence,
            "raw_score": raw_score,
            "cat_prob": cat_prob,
            "dog_prob": dog_prob,
        }
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=5000, debug=debug)
