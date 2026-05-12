"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Explainability

This module generates heatmap visualizations to explain model predictions.
"""

import os
import logging
from typing import Tuple, Optional, List
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from tensorflow import keras

from src.preprocessing import preprocess_image
from src.config import CLASS_NAMES_DISPLAY

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    """
    
    def __init__(self, model: keras.Model, last_conv_layer_name: str = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained Keras model
            last_conv_layer_name: Name of the last convolutional layer
                                  If None, will try to auto-detect
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name or self._find_last_conv_layer()
        
        # Create a model that outputs both the last conv layer and the final output
        self.grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(self.last_conv_layer_name).output,
                model.output
            ]
        )
        
        logger.info(f"Grad-CAM initialized with layer: {self.last_conv_layer_name}")
    
    def _find_last_conv_layer(self) -> str:
        """Automatically find the last convolutional layer."""
        for layer in reversed(self.model.layers):
            layer_name = layer.name.lower()
            # Look for Conv2D layers specifically
            if 'conv2d' in layer_name or 'conv' in layer_name:
                try:
                    # Keras 3 compatible shape check
                    output_shape = layer.output.shape
                    if len(output_shape) == 4:  # Conv layers have 4D output (batch, h, w, channels)
                        return layer.name
                except:
                    continue
        
        # Fallback: try to get output shape from model summary
        for layer in reversed(self.model.layers):
            try:
                # Try different methods for Keras 2/3 compatibility
                if hasattr(layer, 'output'):
                    shape = layer.output.shape
                    if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
                        return layer.name
            except:
                continue
        
        # Last resort: find any layer with 4D output
        for layer in reversed(self.model.layers):
            if any(x in layer.name.lower() for x in ['conv', 'add', 'relu', 'activation']):
                return layer.name
        
        raise ValueError("Could not find suitable convolutional layer")
    
    def compute_heatmap(
        self,
        image: np.ndarray,
        class_index: int = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            image: Preprocessed image (batch_size, height, width, channels)
            class_index: Target class index (None for predicted class)
            eps: Small value to avoid division by zero
            
        Returns:
            Heatmap array
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            
            if class_index is None:
                class_index = np.argmax(predictions[0])
            
            loss = predictions[:, class_index]
        
        # Compute gradients of the loss w.r.t. the conv layer outputs
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv layer outputs by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Apply ReLU (only positive values)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy() if hasattr(heatmap, 'numpy') else heatmap
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original image (before preprocessing)
            alpha: Transparency factor
            colormap: Matplotlib colormap name
            
        Returns:
            Overlayed image
        """
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to colormap
        heatmap_colored = plt.get_cmap(colormap)(heatmap_resized)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Overlay
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    def generate_visualization(
        self,
        image_path: str,
        class_index: int = None,
        save_path: str = None,
        image_size: Tuple[int, int] = (224, 224),
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Generate complete Grad-CAM visualization.
        
        Args:
            image_path: Path to input image
            class_index: Target class index (None for predicted class)
            save_path: Path to save the visualization
            image_size: Target size for model input
            alpha: Heatmap transparency
            colormap: Colormap name
            
        Returns:
            Tuple of (original_image, heatmap, overlayed_image, predicted_class)
        """
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        processed_img = preprocess_image(image_path, image_size, normalize=True)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # Get prediction
        predictions = self.model.predict(processed_img, verbose=0)
        pred_class_idx = np.argmax(predictions[0])
        pred_class = list(CLASS_NAMES_DISPLAY.values())[pred_class_idx]
        
        # Use predicted class if not specified
        if class_index is None:
            class_index = pred_class_idx
        
        # Compute heatmap
        heatmap = self.compute_heatmap(processed_img, class_index)
        
        # Create overlay
        overlayed = self.overlay_heatmap(heatmap, original_img, alpha, colormap)
        
        # Create visualization figure
        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap=colormap)
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            axes[2].imshow(overlayed)
            axes[2].set_title(f'Overlay: {pred_class}\nConfidence: {predictions[0][pred_class_idx]*100:.1f}%')
            axes[2].axis('off')
            
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Grad-CAM visualization saved: {save_path}")
            plt.close()
        
        return original_img, heatmap, overlayed, pred_class


def generate_gradcam_for_model(
    model_path: str,
    image_path: str,
    save_path: str = None,
    image_size: int = 224,
    last_conv_layer: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to generate Grad-CAM for a saved model.
    
    Args:
        model_path: Path to saved model
        image_path: Path to input image
        save_path: Path to save visualization
        image_size: Target image size
        last_conv_layer: Name of last conv layer (auto-detect if None)
        
    Returns:
        Tuple of (original_image, heatmap, overlayed_image)
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, last_conv_layer)
    
    # Generate visualization
    original, heatmap, overlayed, pred_class = gradcam.generate_visualization(
        image_path=image_path,
        save_path=save_path,
        image_size=(image_size, image_size)
    )
    
    return original, heatmap, overlayed


def visualize_multiple_layers(
    model: keras.Model,
    image_path: str,
    layer_names: List[str],
    save_path: str = None,
    image_size: int = 224
):
    """
    Generate Grad-CAM for multiple layers.
    
    Args:
        model: Keras model
        image_path: Path to input image
        layer_names: List of layer names to visualize
        save_path: Path to save the figure
        image_size: Target image size
    """
    # Load and preprocess image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    processed_img = preprocess_image(image_path, (image_size, image_size), normalize=True)
    processed_img = np.expand_dims(processed_img, axis=0)
    
    # Create figure
    n_layers = len(layer_names)
    fig, axes = plt.subplots(2, n_layers + 1, figsize=(4 * (n_layers + 1), 8))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Heatmaps
    for i, layer_name in enumerate(layer_names):
        try:
            gradcam = GradCAM(model, layer_name)
            heatmap = gradcam.compute_heatmap(processed_img)
            
            axes[0, i + 1].imshow(heatmap, cmap='jet')
            axes[0, i + 1].set_title(f'Layer: {layer_name}')
            axes[0, i + 1].axis('off')
            
            # Overlay
            overlayed = gradcam.overlay_heatmap(heatmap, original_img)
            axes[1, i + 1].imshow(overlayed)
            axes[1, i + 1].set_title('Overlay')
            axes[1, i + 1].axis('off')
            
        except Exception as e:
            logger.error(f"Error processing layer {layer_name}: {e}")
            axes[0, i + 1].text(0.5, 0.5, f'Error: {layer_name}', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Multi-layer visualization saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Grad-CAM Module - Example Usage")
    print("="*50)
    
    print("""
    # Example 1: Basic Grad-CAM
    from src.gradcam import generate_gradcam_for_model
    
    original, heatmap, overlayed = generate_gradcam_for_model(
        model_path='models/best_model.h5',
        image_path='data/test/image.jpg',
        save_path='results/gradcam_output.png'
    )
    
    # Example 2: Using GradCAM class
    from src.gradcam import GradCAM
    from tensorflow import keras
    
    model = keras.models.load_model('models/best_model.h5')
    gradcam = GradCAM(model)
    
    original, heatmap, overlayed, pred_class = gradcam.generate_visualization(
        image_path='data/test/image.jpg',
        save_path='results/gradcam.png'
    )
    """)
