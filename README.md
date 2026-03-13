# Cat vs Dog Image Classifier (Flask)

A clean, modern web UI for a **Cat vs Dog** image classifier. Upload an image (drag & drop supported), preview it instantly, and get a prediction with a smooth loading state and a stylish result badge.

## Features

- **Modern minimalist UI**: centered card layout, subtle shadows, responsive design
- **Drag & drop upload** + file picker
- **Image preview** before prediction
- **Loading spinner** during inference
- **Result badge** + confidence bar
- **Flask backend** with a simple JSON prediction endpoint

## Project structure

```text
cat-and-dog-image-classifier/
  app.py
  requirements.txt
  templates/
    index.html
  static/
    app.js
```

## Requirements

- Python 3.10+ recommended
- A trained Keras/TensorFlow model file (`.h5`) in one of these locations:
  - `optimized_models/model_optimized.h5` (preferred)
  - `dog_cat_model.h5`
  - `dogs_vs_cats_model.h5`
  - `dogs_vs_cats_simple_cnn.h5`

> Note: `.gitignore` ignores most model files by default (`*.h5`), but **allows** optimized models under `optimized_models/`.

## Setup

Create and activate a virtual environment (recommended), then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
python app.py
```

Then open:

- `http://localhost:5000`

## How to use

1. Drag & drop (or choose) an image file (PNG/JPG/JPEG/BMP).
2. Confirm the preview looks correct.
3. Click **Predict**.
4. View the **Cat/Dog** badge and confidence details.

## API

### `POST /predict`

Accepts `multipart/form-data` with:

- `image`: the uploaded image file

Returns JSON:

```json
{
  "label": "Cat",
  "confidence": 0.93,
  "raw_score": 0.07
}
```

## Notes

- Maximum upload size is **10MB** (configured in `app.py`).
- If you see “Model not loaded” on the page, place a trained `.h5` in one of the supported locations and restart the app.

