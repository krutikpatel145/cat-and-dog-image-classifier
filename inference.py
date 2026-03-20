import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["Cat", "Dog", "Not Detected"]

def resolve_model_path() -> str | None:
    # Kept for backward compatibility with app.py imports
    return "mobilenet_v2"

def load_model(model_path: str | None = None) -> tuple[keras.Model | None, str | None]:
    try:
        model = MobileNetV2(weights="imagenet")
        return model, None
    except Exception as e:
        return None, f"Failed to load MobileNetV2: {e}"

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(
    model: keras.Model,
    img_array: np.ndarray,
) -> tuple[str, float, float, float, float]:
    """
    Returns:
      - label: "Cat", "Dog", or "Not Detected"
      - confidence: probability of the predicted label
      - raw_score: float, probability of the top ImageNet class
      - cat_prob: probability if it's considered a cat
      - dog_prob: probability if it's considered a dog
    """
    preds = model.predict(img_array, verbose=0)[0]
    top_class_idx = int(np.argmax(preds))
    top_class_prob = float(preds[top_class_idx])

    # ImageNet Dog classes: 151 to 268 inclusive
    # ImageNet Cat classes: 281 to 285 inclusive (tabby, tiger cat, Persian cat, Siamese cat, Egyptian cat)
    
    if 151 <= top_class_idx <= 268:
        label = "Dog"
        confidence = top_class_prob
        dog_prob = top_class_prob
        cat_prob = 0.0
    elif 281 <= top_class_idx <= 285:
        label = "Cat"
        confidence = top_class_prob
        dog_prob = 0.0
        cat_prob = top_class_prob
    else:
        label = "Not Detected"
        confidence = top_class_prob
        dog_prob = 0.0
        cat_prob = 0.0

    return label, confidence, top_class_prob, cat_prob, dog_prob
