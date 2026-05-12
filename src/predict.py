"""
Prediction/Inference Module for Eye Disease Classification

This module handles:
- Single image prediction
- Batch prediction
- Confidence scoring
- Result formatting
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import cv2
from tensorflow import keras

from src.config import CLASS_NAMES, CLASS_NAMES_DISPLAY, CLASS_COLORS
from src.preprocessing import preprocess_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EyeDiseasePredictor:
    """
    Predictor class for eye disease classification.
    
    Handles model loading and prediction on new images.
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str] = None,
        image_size: Tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model (.h5 file)
            class_names: List of class names
            image_size: Target image size
            confidence_threshold: Minimum confidence for prediction
        """
        self.model_path = model_path
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        
        if class_names is None:
            self.class_names = CLASS_NAMES
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self) -> keras.Model:
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model from: {self.model_path}")
        model = keras.models.load_model(self.model_path)
        logger.info("Model loaded successfully")
        return model
    
    def preprocess(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Preprocessed image array
        """
        if isinstance(image, str):
            # Load from path
            img = preprocess_image(image, self.image_size, apply_crop=True, normalize=True)
        else:
            # Already an array
            img = cv2.resize(image, self.image_size)
            img = img.astype(np.float32) / 255.0
        
        if img is None:
            raise ValueError("Failed to preprocess image")
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(
        self,
        image: Union[str, np.ndarray],
        return_top_k: int = 3
    ) -> Dict:
        """
        Predict on a single image.
        
        Args:
            image: Image path or numpy array
            return_top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        processed_img = self.preprocess(image)
        
        # Predict
        predictions = self.model.predict(processed_img, verbose=0)
        probabilities = predictions[0]
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:return_top_k]
        
        results = {
            'predicted_class': self.class_names[top_indices[0]],
            'predicted_class_display': CLASS_NAMES_DISPLAY.get(
                self.class_names[top_indices[0]],
                self.class_names[top_indices[0]]
            ),
            'confidence': float(probabilities[top_indices[0]]),
            'all_probabilities': {
                self.class_names[i]: float(probabilities[i])
                for i in range(len(self.class_names))
            },
            'top_predictions': [
                {
                    'class': self.class_names[idx],
                    'class_display': CLASS_NAMES_DISPLAY.get(
                        self.class_names[idx],
                        self.class_names[idx]
                    ),
                    'probability': float(probabilities[idx]),
                    'percentage': f"{probabilities[idx]*100:.2f}%"
                }
                for idx in top_indices
            ]
        }
        
        return results
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict on multiple images.
        
        Args:
            images: List of image paths or arrays
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Preprocess batch
            batch_imgs = []
            for img in batch:
                try:
                    processed = self.preprocess(img)
                    batch_imgs.append(processed[0])  # Remove batch dim
                except Exception as e:
                    logger.error(f"Error preprocessing image: {e}")
                    continue
            
            if not batch_imgs:
                continue
            
            # Stack and predict
            batch_array = np.stack(batch_imgs, axis=0)
            predictions = self.model.predict(batch_array, verbose=0)
            
            # Process results
            for j, pred in enumerate(predictions):
                top_idx = np.argmax(pred)
                result = {
                    'image_index': i + j,
                    'predicted_class': self.class_names[top_idx],
                    'predicted_class_display': CLASS_NAMES_DISPLAY.get(
                        self.class_names[top_idx],
                        self.class_names[top_idx]
                    ),
                    'confidence': float(pred[top_idx]),
                    'all_probabilities': {
                        self.class_names[k]: float(pred[k])
                        for k in range(len(self.class_names))
                    }
                }
                results.append(result)
        
        return results
    
    def is_confident(self, prediction: Dict) -> bool:
        """
        Check if prediction meets confidence threshold.
        
        Args:
            prediction: Prediction result from predict()
            
        Returns:
            True if confidence >= threshold
        """
        return prediction['confidence'] >= self.confidence_threshold


def predict_single_image(
    model_path: str,
    image_path: str,
    class_names: List[str] = None,
    image_size: int = 224
) -> Dict:
    """
    Convenience function for single image prediction.
    
    Args:
        model_path: Path to saved model
        image_path: Path to image file
        class_names: List of class names
        image_size: Target image size
        
    Returns:
        Prediction results dictionary
    """
    predictor = EyeDiseasePredictor(
        model_path=model_path,
        class_names=class_names,
        image_size=(image_size, image_size)
    )
    
    return predictor.predict(image_path)


def format_prediction_result(result: Dict, detailed: bool = False) -> str:
    """
    Format prediction result as a readable string.
    
    Args:
        result: Prediction result dictionary
        detailed: Whether to show detailed probabilities
        
    Returns:
        Formatted string
    """
    output = []
    output.append("="*50)
    output.append("PREDICTION RESULT")
    output.append("="*50)
    output.append(f"\nPredicted Class: {result['predicted_class_display']}")
    output.append(f"Confidence: {result['confidence']*100:.2f}%")
    
    if detailed:
        output.append("\nAll Class Probabilities:")
        output.append("-"*50)
        
        # Sort by probability
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for class_name, prob in sorted_probs:
            display_name = CLASS_NAMES_DISPLAY.get(class_name, class_name)
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            output.append(f"{display_name:25s} |{bar}| {prob*100:5.2f}%")
    
    output.append("="*50)
    
    return "\n".join(output)


def get_prediction_color(class_name: str) -> str:
    """
    Get color for a class (for visualization).
    
    Args:
        class_name: Name of the class
        
    Returns:
        Hex color code
    """
    return CLASS_COLORS.get(class_name, "#3498db")


if __name__ == "__main__":
    # Example usage
    print("Prediction Module - Example Usage")
    print("="*50)
    
    # This would require a trained model
    # For demonstration, we'll show the API
    
    print("\nExample 1: Single Image Prediction")
    print("-"*50)
    print("""
    from src.predict import predict_single_image
    
    result = predict_single_image(
        model_path='models/best_model.h5',
        image_path='data/test/image.jpg'
    )
    
    print(result['predicted_class_display'])
    print(f"Confidence: {result['confidence']*100:.2f}%")
    """)
    
    print("\nExample 2: Using Predictor Class")
    print("-"*50)
    print("""
    from src.predict import EyeDiseasePredictor
    
    predictor = EyeDiseasePredictor('models/best_model.h5')
    
    # Single prediction
    result = predictor.predict('data/test/image.jpg')
    
    # Batch prediction
    results = predictor.predict_batch([
        'image1.jpg',
        'image2.jpg',
        'image3.jpg'
    ])
    """)
