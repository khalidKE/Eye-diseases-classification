"""
Image Preprocessing Pipeline for Eye Disease Classification

This module handles all image preprocessing tasks including:
- Loading and format conversion
- Medical-specific cropping (remove black borders)
- Resizing and normalization
- Corrupt image detection
"""

import os
import cv2
import glob
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Union
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crop the black borders from fundus images.
    Finds the contours of the actual eye circle and crops around it.
    
    Args:
        img: Input image (2D grayscale or 3D RGB)
        tol: Tolerance for detecting dark pixels
        
    Returns:
        Cropped image
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    apply_crop: bool = True,
    normalize: bool = False
) -> Optional[np.ndarray]:
    """
    Loads, crops, and resizes an image. Returns None if corrupt.
    
    Args:
        image_path: Path to the image file
        target_size: Target size as (width, height)
        apply_crop: Whether to apply medical cropping
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Processed image array or None if corrupt
    """
    try:
        # Load in BGR
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Corrupt or unreadable image -> {image_path}")
            return None
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Medical crop (remove black borders)
        if apply_crop:
            img = crop_image_from_gray(img)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Optional normalization
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def verify_image(image_path: str) -> bool:
    """
    Verify if an image is valid and not corrupt.
    
    Args:
        image_path: Path to the image
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        # Check if image has content
        if img.size == 0:
            return False
        # Check minimum dimensions
        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return False
        return True
    except Exception:
        return False


def process_single_image(
    image_path: str,
    output_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> bool:
    """
    Process a single image and save it.
    
    Args:
        image_path: Input image path
        output_path: Output image path
        target_size: Target size
        
    Returns:
        True if successful, False otherwise
    """
    img = preprocess_image(image_path, target_size)
    if img is not None:
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Ensure correct extension
            if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                output_path += '.jpg'
            
            # Save processed image (convert back to BGR for cv2)
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return True
        except Exception as e:
            logger.error(f"Error saving {output_path}: {e}")
    return False


def process_dataset(
    raw_dir: str,
    processed_dir: str,
    classes: List[str],
    target_size: Tuple[int, int] = (224, 224),
    verbose: bool = True
) -> dict:
    """
    Process entire dataset from raw to processed.
    
    Args:
        raw_dir: Directory containing raw images organized by class
        processed_dir: Directory to save processed images
        classes: List of class names
        target_size: Target image size
        verbose: Whether to print progress
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_images': 0,
        'processed': 0,
        'corrupt': 0,
        'class_counts': {}
    }
    
    if verbose:
        logger.info(f"Starting preprocessing pipeline...")
        logger.info(f"Reading from: {raw_dir}")
        logger.info(f"Writing to: {processed_dir}")
        logger.info(f"Target size: {target_size}")
    
    if not os.path.exists(raw_dir):
        logger.error(f"Error: {raw_dir} does not exist.")
        return stats
    
    for cls in classes:
        in_cls_dir = os.path.join(raw_dir, cls)
        out_cls_dir = os.path.join(processed_dir, cls)
        
        if not os.path.exists(in_cls_dir):
            logger.warning(f"Skipping class {cls}, directory not found.")
            continue
        
        os.makedirs(out_cls_dir, exist_ok=True)
        
        # Get all image files
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(glob.glob(os.path.join(in_cls_dir, ext)))
        
        if verbose:
            logger.info(f"Processing {len(image_paths)} images for class '{cls}'...")
        
        valid_count = 0
        corrupt_count = 0
        
        iterator = tqdm(image_paths, desc=f"Processing {cls}") if verbose else image_paths
        
        for img_path in iterator:
            stats['total_images'] += 1
            
            # Generate output path
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(out_cls_dir, base_name + '.jpg')
            
            if process_single_image(img_path, save_path, target_size):
                valid_count += 1
                stats['processed'] += 1
            else:
                corrupt_count += 1
                stats['corrupt'] += 1
        
        stats['class_counts'][cls] = {
            'valid': valid_count,
            'corrupt': corrupt_count
        }
        
        if verbose:
            logger.info(f"Class '{cls}': {valid_count} valid, {corrupt_count} corrupt.")
    
    return stats


def get_image_statistics(image_dir: str) -> dict:
    """
    Get statistics about images in a directory.
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_images': 0,
        'avg_width': 0,
        'avg_height': 0,
        'min_width': float('inf'),
        'min_height': float('inf'),
        'max_width': 0,
        'max_height': 0,
        'corrupt_images': 0
    }
    
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    widths = []
    heights = []
    
    for img_path in tqdm(image_paths, desc="Analyzing images"):
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                stats['total_images'] += 1
                
                stats['min_width'] = min(stats['min_width'], w)
                stats['min_height'] = min(stats['min_height'], h)
                stats['max_width'] = max(stats['max_width'], w)
                stats['max_height'] = max(stats['max_height'], h)
            else:
                stats['corrupt_images'] += 1
        except Exception:
            stats['corrupt_images'] += 1
    
    if widths and heights:
        stats['avg_width'] = sum(widths) / len(widths)
        stats['avg_height'] = sum(heights) / len(heights)
    
    return stats


if __name__ == "__main__":
    # Example usage
    from src.config import get_config
    
    config = get_config()
    raw_dir = config.get('dataset.raw_data_path', 'data/raw')
    processed_dir = config.get('dataset.processed_data_path', 'data/processed')
    classes = config.get('dataset.classes', [])
    image_size = config.get('dataset.image_size', 224)
    
    stats = process_dataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        classes=classes,
        target_size=(image_size, image_size)
    )
    
    print("\nProcessing Complete!")
    print(f"Total Images: {stats['total_images']}")
    print(f"Successfully Processed: {stats['processed']}")
    print(f"Corrupt/Invalid: {stats['corrupt']}")
    print("\nPer-class breakdown:")
    for cls, counts in stats['class_counts'].items():
        print(f"  {cls}: {counts['valid']} valid, {counts['corrupt']} corrupt")
