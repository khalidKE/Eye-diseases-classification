"""
Streamlit Web Application for Eye Disease Classification

A user-friendly interface for:
- Uploading retinal images
- Getting disease predictions
- Viewing Grad-CAM visualizations
- Understanding model confidence
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

from src.predict import EyeDiseasePredictor, format_prediction_result, get_prediction_color
from src.gradcam import generate_gradcam_for_model
from src.config import CLASS_NAMES_DISPLAY, CLASS_COLORS, CLASS_NAMES

# Page configuration
st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
        color: #856404;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str):
    """Load and cache the prediction model."""
    return EyeDiseasePredictor(model_path)


def display_header():
    """Display application header."""
    st.title("👁️ Eye Disease Classification")
    st.markdown("""
    ### AI-Powered Retinal Image Analysis
    
    Upload a retinal fundus image to detect:
    - **Normal** - Healthy eye
    - **Diabetic Retinopathy** - Diabetes-related eye damage
    - **Cataract** - Clouding of the eye lens
    - **Glaucoma** - Optic nerve damage
    """)


def display_disclaimer():
    """Display medical disclaimer."""
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Medical Disclaimer</b><br>
    This tool is for educational and research purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the guidance of a qualified healthcare provider for any medical concerns.
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar controls."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_dir = Path("models")
        if model_dir.exists():
            model_files = list(model_dir.glob("**/*.h5"))
            if model_files:
                selected_model = st.selectbox(
                    "Select Model",
                    options=[str(f) for f in model_files],
                    format_func=lambda x: Path(x).name
                )
            else:
                st.error("No models found. Please train a model first.")
                selected_model = None
        else:
            st.error("Models directory not found.")
            selected_model = None
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Visualization options
        st.subheader("🔍 Visualization")
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        
        # About
        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown("""
        **Version:** 1.0.0
        
        **Technologies:**
        - TensorFlow/Keras
        - ResNet50 & EfficientNet
        - Grad-CAM Explainability
        
        **Dataset:**
        [Kaggle Eye Diseases](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
        """)
    
    return selected_model, confidence_threshold, show_gradcam


def process_uploaded_image(uploaded_file, temp_dir):
    """Save uploaded file to temp directory and return path."""
    # Save uploaded file
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def display_prediction_results(result):
    """Display prediction results in a formatted way."""
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    
    # Get display name and color
    display_name = result['predicted_class_display']
    color = get_prediction_color(predicted_class)
    
    # Main prediction box
    st.markdown(f"""
    <div class="prediction-box" style="background-color: {color}20; border-left: 4px solid {color};">
        <h3 style="color: {color}; margin: 0;">{display_name}</h3>
        <p style="font-size: 1.5rem; margin: 0;">Confidence: <b>{confidence*100:.2f}%</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # All probabilities
    st.subheader("📊 Class Probabilities")
    
    # Sort by probability
    sorted_probs = sorted(
        result['all_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for class_name, prob in sorted_probs:
        display_name = CLASS_NAMES_DISPLAY.get(class_name, class_name)
        color = CLASS_COLORS.get(class_name, "#3498db")
        
        # Progress bar
        st.markdown(f"**{display_name}**")
        st.progress(prob)
        st.markdown(f"<p style='text-align: right; color: {color};'>{prob*100:.2f}%</p>", 
                   unsafe_allow_html=True)


def display_image_comparison(original_path, gradcam_path):
    """Display original and Grad-CAM images side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        original_img = Image.open(original_path)
        st.image(original_img, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Grad-CAM Visualization")
        if os.path.exists(gradcam_path):
            gradcam_img = Image.open(gradcam_path)
            st.image(gradcam_img, use_container_width=True)
            st.caption("Red/yellow areas indicate regions the model focused on")
        else:
            st.error("Grad-CAM visualization not available")


def main():
    """Main application function."""
    display_header()
    display_disclaimer()
    
    # Sidebar settings
    model_path, confidence_threshold, show_gradcam = display_sidebar()
    
    if model_path is None:
        st.error("⚠️ Please ensure a trained model exists in the 'models' directory.")
        return
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            predictor = load_model(model_path)
            predictor.confidence_threshold = confidence_threshold
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # File upload
    st.markdown("---")
    st.subheader("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process uploaded image
            image_path = process_uploaded_image(uploaded_file, temp_dir)
            
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Uploaded Image")
                image = Image.open(image_path)
                st.image(image, use_container_width=True)
            
            with col2:
                # Make prediction
                st.subheader("🤖 AI Prediction")
                
                with st.spinner("Analyzing image..."):
                    try:
                        result = predictor.predict(image_path)
                        display_prediction_results(result)
                        
                        # Confidence warning
                        if not predictor.is_confident(result):
                            st.warning("⚠️Please consult a specialist.")
                    
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        return
            
            # Grad-CAM visualization
            if show_gradcam:
                st.markdown("---")
                
                gradcam_path = os.path.join(temp_dir, "gradcam.png")
                
                with st.spinner("Generating Grad-CAM visualization..."):
                    try:
                        generate_gradcam_for_model(
                            model_path=model_path,
                            image_path=image_path,
                            save_path=gradcam_path,
                            image_size=224
                        )
                        
                        display_image_comparison(image_path, gradcam_path)
                    
                    except Exception as e:
                        logger.warning(f"Grad-CAM generation failed: {e}")
                        # Show original image without Grad-CAM
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📷 Original Image")
                            image = Image.open(image_path)
                            st.image(image, use_container_width=True)
                        with col2:
                            st.subheader("🔥 Grad-CAM")
                            st.info("Grad-CAM visualization unavailable for this model architecture.\n\nThe AI prediction above is still accurate - only the heatmap overlay is disabled.")
                            st.caption("This is a known issue with TensorFlow 2.21+ Keras 3 models.")
            
            # Additional information
            st.markdown("---")
            st.subheader("ℹ️ About This Prediction")
            
            predicted_class = result['predicted_class']
            
            info_texts = {
                'normal': """
                **Normal Eye** - No signs of disease detected. The retina appears healthy
                with normal blood vessels and optic disc.
                """,
                'diabetic_retinopathy': """
                **Diabetic Retinopathy** - Diabetes-related damage to blood vessels in the retina.
                This is a leading cause of blindness. Early detection and management are crucial.
                """,
                'cataract': """
                **Cataract** - Clouding of the eye's natural lens, leading to blurry vision.
                This is common with aging and can be treated with surgery.
                """,
                'glaucoma': """
                **Glaucoma** - Damage to the optic nerve, often due to high eye pressure.
                This can cause vision loss and requires immediate medical attention.
                """
            }
            
            st.markdown(info_texts.get(predicted_class, ""))
            
            # Download button for results
            st.markdown("---")
            result_text = format_prediction_result(result, detailed=True)
            st.download_button(
                label="📥 Download Prediction Report",
                data=result_text,
                file_name="prediction_report.txt",
                mime="text/plain"
            )


if __name__ == "__main__":
    main()
