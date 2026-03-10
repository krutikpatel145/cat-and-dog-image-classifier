"""
Dogs vs Cats Image Classifier - Web Interface
Simple and attractive Streamlit app for image classification

To run:
1. Install Streamlit: pip install streamlit
2. Run: streamlit run app.py
"""

import streamlit as st
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="🐱🐶 Dogs vs Cats Classifier",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for attractive design
st.markdown("""
    <style>
    .main {
        padding: 2rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #FF5252 0%, #26A69A 100%);
        transform: scale(1.02);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .confidence-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    h1 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Model settings - check for multiple possible model file names
# Priority: Optimized models first (smaller, faster), then original models
POSSIBLE_MODEL_PATHS = [
    "optimized_models/model_optimized.h5",  # Optimized model (recommended for deployment)
    "dog_cat_model.h5",  # Original model
    "dogs_vs_cats_model.h5",
    "dogs_vs_cats_simple_cnn.h5"
]

# Find which model file exists
MODEL_PATH = None
for path in POSSIBLE_MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

IMG_HEIGHT = 150
IMG_WIDTH = 150
class_names = ['Cats', 'Dogs']

# Title and description
st.markdown("<h1>🐱🐶 Dogs vs Cats Classifier</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI tell you if it\'s a Cat or Dog!</p>', unsafe_allow_html=True)

# Load model (cache it so it doesn't reload every time)
@st.cache_resource
def load_model():
    """Load the trained model"""
    if MODEL_PATH is None:
        st.error("❌ Model file not found!")
        st.info("""
        **Please train the model first by running:**
        ```
        python dogs_vs_cats_classifier.py
        ```
        
        This will create the model file needed for predictions.
        """)
        st.stop()
        return None
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        st.success(f"✅ Model loaded: {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
model = load_model()

if model is None:
    st.stop()

# Function to preprocess image
def preprocess_image(img):
    """Preprocess image for prediction"""
    # Resize image
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    # Convert to array
    img_array = image.img_to_array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize to 0-1
    img_array = img_array / 255.0
    return img_array

# Function to make prediction
def predict_image(img_array, model):
    """Make prediction on preprocessed image"""
    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_index = int(prediction > 0.5)
    predicted_class = class_names[predicted_index]
    confidence = prediction if predicted_index == 1 else (1 - prediction)
    return predicted_class, confidence, prediction

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload an image of a Cat or Dog",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Supported formats: JPG, JPEG, PNG, BMP"
)

# Display uploaded image and prediction
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, caption="📷 Your Uploaded Image", use_container_width=True)
    
    with col2:
        # Preprocess and predict
        with st.spinner("🔮 Analyzing image..."):
            img_array = preprocess_image(img)
            predicted_class, confidence, raw_score = predict_image(img_array, model)
        
        # Display prediction result
        st.markdown("---")
        
        # Prediction box with emoji
        if predicted_class == "Cats":
            emoji = "🐱"
            color = "#FF6B6B"
        else:
            emoji = "🐶"
            color = "#4ECDC4"
        
        st.markdown(f"""
        <div class="prediction-box" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
            <h2 style="font-size: 3rem; margin: 0;">{emoji}</h2>
            <h2 style="margin: 1rem 0;">It's a {predicted_class.upper()}!</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence score
        confidence_percent = confidence * 100
        
        # Confidence bar
        st.progress(confidence)
        st.markdown(f"**Confidence: {confidence_percent:.1f}%**")
        
        # Confidence message
        if confidence_percent > 80:
            st.success("✅ Very High Confidence - Model is very sure!")
        elif confidence_percent > 70:
            st.info("✅ High Confidence - Model is confident!")
        elif confidence_percent > 60:
            st.warning("⚠️ Moderate Confidence - Model is somewhat sure")
        else:
            st.error("❌ Low Confidence - Model is uncertain")
        
        # Additional info (expandable)
        with st.expander("ℹ️ More Details"):
            st.write(f"**Raw Prediction Score:** {raw_score:.4f}")
            st.write(f"**Class Index:** {0 if predicted_class == 'Cats' else 1}")
            st.write(f"**Image Size:** {img.size[0]} x {img.size[1]} pixels")
    
    # Try another image button
    st.markdown("---")
    if st.button("🔄 Try Another Image"):
        st.rerun()

else:
    # Show instructions when no image is uploaded
    st.info("👆 Please upload an image above to get started!")
    
    # Show example section
    st.markdown("---")
    st.markdown("### 📝 How to Use:")
    st.markdown("""
    1. **Click** the "Upload an image" area above
    2. **Select** an image file from your computer (JPG, PNG, etc.)
    3. **Wait** for the AI to analyze the image
    4. **See** the prediction result!
    """)
    
    st.markdown("### 💡 Tips:")
    st.markdown("""
    - Use clear, focused images for best results
    - The model works best with images showing the full face/body
    - Supported formats: JPG, JPEG, PNG, BMP
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>"
    "Made with ❤️ using TensorFlow, Keras & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
