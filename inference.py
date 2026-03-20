import os

import numpy as np
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
MODEL_THRESHOLD = 0.5


def resolve_model_path() -> str | None:
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


def load_model(model_path: str | None) -> tuple[keras.Model | None, str | None]:
    if not model_path:
        return None, "No .h5 model found in the supported locations."

    try:
        model = keras.models.load_model(model_path)
        return model, None
    except Exception as e:  # pragma: no cover - depends on local model validity
        return None, f"Failed to load model from '{model_path}': {e}"


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_image(
    model: keras.Model,
    img_array: np.ndarray,
) -> tuple[str, float, float, float, float]:
    """
    Returns:
      - label: "Cat" or "Dog"
      - confidence: probability of the predicted label
      - raw_score: raw model output (not forced into [0,1])
      - cat_prob: probability of Cat in [0,1]
      - dog_prob: probability of Dog in [0,1]
    """

    raw_score = float(model.predict(img_array, verbose=0)[0][0])
    dog_prob = float(np.clip(raw_score, 0.0, 1.0))
    cat_prob = float(1.0 - dog_prob)

    label = CLASS_NAMES[int(dog_prob > MODEL_THRESHOLD)]
    confidence = dog_prob if label == "Dog" else cat_prob

    return label, float(confidence), float(raw_score), float(cat_prob), float(dog_prob)

