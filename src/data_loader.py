"""
Data Loader Module for Eye Disease Classification

Handles:
- Loading image paths and labels
- Train/Validation/Test splitting
- TensorFlow Dataset creation
- Data augmentation integration
"""

import os
import glob
import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'dataset': {
        'processed_data_path': 'data/processed',
        'classes': ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma'],
        'image_size': 224
    },
    'training': {
        'batch_size': 32
    },
    'augmentation': {
        'enabled': True,
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.1,
        'horizontal_flip': True,
        'brightness_range': [0.8, 1.2]
    }
}

def get_image_paths_and_labels(dataset_dir, classes):
    paths = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.exists(cls_dir):
            files = glob.glob(os.path.join(cls_dir, '*.*'))
            paths.extend(files)
            labels.extend([idx] * len(files))
    return paths, labels

def create_tf_dataset(paths, labels, batch_size, is_training=False):
    """
    Creates a high-performance tf.data.Dataset from file paths and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    def parse_image(file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.ensure_shape(image, [224, 224, 3])
        label = tf.one_hot(label, depth=4)
        return image, label

    # Map the parsing function
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def get_data_generators(config: Dict = None):
    """
    Splits the dataset 70/15/15 and returns train, val, and test tf.data.Datasets.
    
    Args:
        config: Configuration dictionary. If None, uses default config.
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds) or (None, None, None) if no data
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    dataset_dir = config.get('dataset', {}).get('processed_data_path', 'data/processed')
    batch_size = config.get('training', {}).get('batch_size', 32)
    classes = config.get('dataset', {}).get('classes', ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma'])
    
    logger.info(f"Loading data paths from {dataset_dir}...")
    paths, labels = get_image_paths_and_labels(dataset_dir, classes)
    
    if not paths:
        logger.warning(f"No images found in {dataset_dir}. Returning empty datasets.")
        return None, None, None

    # First split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    
    # Second split: Split the 30% Temp into 15% Val and 15% Test (50/50 split of Temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    train_ds = create_tf_dataset(X_train, y_train, batch_size, is_training=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size, is_training=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size, is_training=False)
    
    return train_ds, val_ds, test_ds


def create_image_data_generators(
    data_dir: str,
    classes: List[str],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.15,
    test_split: float = 0.15,
    augmentation_config: Dict = None
) -> Tuple:
    """
    Create Keras ImageDataGenerators for train, validation, and test.
    
    This is an alternative to tf.data.Dataset that some users prefer.
    
    Args:
        data_dir: Directory containing processed images organized by class
        classes: List of class names
        image_size: Target image size (height, width)
        batch_size: Batch size
        validation_split: Fraction for validation
        test_split: Fraction for test
        augmentation_config: Augmentation parameters
        
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    if augmentation_config is None:
        augmentation_config = DEFAULT_CONFIG['augmentation']
    
    # Training generator with augmentation
    if augmentation_config.get('enabled', True):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=augmentation_config.get('rotation_range', 15),
            width_shift_range=augmentation_config.get('width_shift_range', 0.1),
            height_shift_range=augmentation_config.get('height_shift_range', 0.1),
            zoom_range=augmentation_config.get('zoom_range', 0.1),
            horizontal_flip=augmentation_config.get('horizontal_flip', True),
            brightness_range=augmentation_config.get('brightness_range', [0.8, 1.2]),
            validation_split=validation_split + test_split
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split + test_split
        )
    
    # Validation/Test generator (no augmentation)
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split + test_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        subset='training',
        seed=42,
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        subset='validation',
        seed=42,
        shuffle=False
    )
    
    # For test set, we need a separate generator with different subset
    # Since we can't easily split validation into val+test with Keras,
    # we'll use the same validation generator and split manually later
    
    return train_generator, val_generator, None


def get_class_distribution(data_dir: str, classes: List[str]) -> Dict[str, int]:
    """
    Get the number of images per class.
    
    Args:
        data_dir: Directory containing class subdirectories
        classes: List of class names
        
    Returns:
        Dictionary mapping class names to image counts
    """
    distribution = {}
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.exists(cls_dir):
            image_files = glob.glob(os.path.join(cls_dir, '*.*'))
            # Filter for image extensions
            image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            distribution[cls] = len(image_files)
        else:
            distribution[cls] = 0
    return distribution


def print_dataset_info(data_dir: str, classes: List[str]):
    """
    Print information about the dataset.
    
    Args:
        data_dir: Directory containing the dataset
        classes: List of class names
    """
    logger.info("\n" + "="*50)
    logger.info("Dataset Information")
    logger.info("="*50)
    
    distribution = get_class_distribution(data_dir, classes)
    total = sum(distribution.values())
    
    logger.info(f"Data Directory: {data_dir}")
    logger.info(f"Total Images: {total}")
    logger.info("\nClass Distribution:")
    
    for cls, count in distribution.items():
        percentage = (count / total * 100) if total > 0 else 0
        logger.info(f"  {cls:25s}: {count:4d} images ({percentage:5.1f}%)")
    
    logger.info("="*50 + "\n")


if __name__ == "__main__":
    # Test the data loader
    config = DEFAULT_CONFIG
    
    # Print dataset info
    data_dir = config['dataset']['processed_data_path']
    classes = config['dataset']['classes']
    
    if os.path.exists(data_dir):
        print_dataset_info(data_dir, classes)
        
        # Test tf.data pipeline
        train_ds, val_ds, test_ds = get_data_generators(config)
        if train_ds:
            logger.info("\nTensorFlow Datasets initialized successfully.")
            for images, labels in train_ds.take(1):
                logger.info(f"Image batch shape: {images.shape}")
                logger.info(f"Label batch shape: {labels.shape}")
    else:
        logger.warning(f"Data directory not found: {data_dir}")
        logger.info("Please run preprocessing first.")
