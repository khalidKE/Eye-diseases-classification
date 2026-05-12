"""
Training Pipeline for Eye Disease Classification

This module handles:
- Model training with callbacks
- Hyperparameter management
- Checkpoint saving
- Training history logging
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)

from src.models import create_model, compile_model
from src.data_loader import get_data_generators, create_image_data_generators

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default training configuration
DEFAULT_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.2,
    'min_lr': 1e-7,
    'model_checkpoint': True,
    'early_stopping': True,
    'reduce_lr_on_plateau': True,
    'tensorboard': True
}


def create_callbacks(
    model_name: str,
    save_dir: str = 'models',
    logs_dir: str = 'logs',
    config: Dict = None
) -> List:
    """
    Create training callbacks.
    
    Args:
        model_name: Name of the model
        save_dir: Directory to save model checkpoints
        logs_dir: Directory to save logs
        config: Callback configuration
        
    Returns:
        List of Keras callbacks
    """
    if config is None:
        config = DEFAULT_TRAINING_CONFIG
    
    callbacks = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model Checkpoint
    if config.get('model_checkpoint', True):
        checkpoint_path = os.path.join(save_dir, f'{model_name}_{timestamp}', 'best_model.h5')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        logger.info(f"Model checkpoint: {checkpoint_path}")
    
    # Early Stopping
    if config.get('early_stopping', True):
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        logger.info(f"Early stopping: patience={config.get('early_stopping_patience', 10)}")
    
    # Reduce LR on Plateau
    if config.get('reduce_lr_on_plateau', True):
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.get('reduce_lr_factor', 0.2),
            patience=config.get('reduce_lr_patience', 5),
            min_lr=config.get('min_lr', 1e-7),
            verbose=1
        )
        callbacks.append(reduce_lr)
        logger.info(f"Reduce LR: factor={config.get('reduce_lr_factor', 0.2)}, "
                   f"patience={config.get('reduce_lr_patience', 5)}")
    
    # TensorBoard
    if config.get('tensorboard', True):
        log_dir = os.path.join(logs_dir, 'tensorboard', f'{model_name}_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)
        
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        logger.info(f"TensorBoard: {log_dir}")
    
    # CSV Logger
    csv_path = os.path.join(logs_dir, f'{model_name}_{timestamp}_training.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_logger = CSVLogger(csv_path, append=True)
    callbacks.append(csv_logger)
    logger.info(f"CSV Logger: {csv_path}")
    
    return callbacks


def train_model(
    model_name: str,
    data_dir: str,
    classes: List[str],
    model_config: Dict = None,
    training_config: Dict = None,
    save_dir: str = 'models',
    logs_dir: str = 'logs',
    use_keras_generator: bool = True
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train a model with the specified configuration.
    
    Args:
        model_name: Name of the model architecture
        data_dir: Directory containing processed data
        classes: List of class names
        model_config: Model-specific configuration
        training_config: Training configuration
        save_dir: Directory to save models
        logs_dir: Directory to save logs
        use_keras_generator: Whether to use Keras ImageDataGenerator
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    if model_config is None:
        model_config = {}
    if training_config is None:
        training_config = DEFAULT_TRAINING_CONFIG.copy()
    
    # Get training parameters
    batch_size = training_config.get('batch_size', 32)
    epochs = training_config.get('epochs', 50)
    learning_rate = training_config.get('learning_rate', 0.001)
    optimizer = training_config.get('optimizer', 'adam')
    image_size = model_config.get('image_size', 224)
    
    logger.info("="*60)
    logger.info(f"Training {model_name}")
    logger.info("="*60)
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Image size: {image_size}")
    
    # Create model
    logger.info("\nCreating model...")
    model = create_model(
        model_name=model_name,
        input_shape=(image_size, image_size, 3),
        num_classes=len(classes),
        **model_config
    )
    
    # Compile model
    model = compile_model(model, learning_rate=learning_rate, optimizer_name=optimizer)
    
    # Print model summary
    model.summary()
    logger.info(f"\nTotal parameters: {model.count_params():,}")
    
    # Create data generators
    logger.info("\nLoading data...")
    
    # Check if data is already split into train/val/test folders
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # Use pre-split data
        logger.info(f"Using pre-split data from: {data_dir}")
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            classes=classes,
            class_mode='categorical',
            shuffle=True
        )
        
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            classes=classes,
            class_mode='categorical',
            shuffle=False
        )
        
        # Calculate steps for pre-split data
        train_steps = train_gen.samples // batch_size
        val_steps = val_gen.samples // batch_size
        
        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
    elif use_keras_generator:
        train_gen, val_gen, _ = create_image_data_generators(
            data_dir=data_dir,
            classes=classes,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            validation_split=0.15,
            test_split=0.15
        )
        
        # Calculate steps
        train_steps = train_gen.samples // batch_size
        val_steps = val_gen.samples // batch_size
        
        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
    else:
        # Use tf.data pipeline
        config = {
            'dataset': {
                'processed_data_path': data_dir,
                'classes': classes,
                'image_size': image_size
            },
            'training': {
                'batch_size': batch_size
            }
        }
        train_gen, val_gen, _ = get_data_generators(config)
        train_steps = None
        val_steps = None
    
    # Create callbacks
    logger.info("\nSetting up callbacks...")
    callbacks = create_callbacks(
        model_name=model_name,
        save_dir=save_dir,
        logs_dir=logs_dir,
        config=training_config
    )
    
    # Train model
    logger.info("\nStarting training...")
    logger.info("="*60)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(save_dir, f'{model_name}_{timestamp}', 'final_model.h5')
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    logger.info(f"\nFinal model saved: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(logs_dir, f'{model_name}_{timestamp}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    logger.info(f"Training history saved: {history_path}")
    
    return model, history


def train_all_models(
    data_dir: str,
    classes: List[str],
    models: List[str] = None,
    model_configs: Dict = None,
    training_config: Dict = None,
    save_dir: str = 'models',
    logs_dir: str = 'logs'
) -> Dict[str, Tuple]:
    """
    Train multiple models and return results.
    
    Args:
        data_dir: Directory containing processed data
        classes: List of class names
        models: List of model names to train
        model_configs: Dictionary of model configurations
        training_config: Training configuration
        save_dir: Directory to save models
        logs_dir: Directory to save logs
        
    Returns:
        Dictionary mapping model names to (model, history) tuples
    """
    if models is None:
        models = ['cnn_baseline', 'resnet50', 'efficientnet']
    if model_configs is None:
        model_configs = {}
    if training_config is None:
        training_config = DEFAULT_TRAINING_CONFIG.copy()
    
    results = {}
    
    for model_name in models:
        logger.info("\n" + "="*60)
        logger.info(f"Training model: {model_name}")
        logger.info("="*60)
        
        try:
            model_config = model_configs.get(model_name, {})
            
            model, history = train_model(
                model_name=model_name,
                data_dir=data_dir,
                classes=classes,
                model_config=model_config,
                training_config=training_config.copy(),
                save_dir=save_dir,
                logs_dir=logs_dir
            )
            
            results[model_name] = (model, history)
            
            # Log final metrics
            final_val_acc = history.history['val_accuracy'][-1]
            final_val_loss = history.history['val_loss'][-1]
            logger.info(f"\nFinal validation accuracy: {final_val_acc:.4f}")
            logger.info(f"Final validation loss: {final_val_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = None
    
    return results


if __name__ == "__main__":
    # Example usage
    from src.config import get_config
    
    config = get_config()
    
    data_dir = config.get('dataset.processed_data_path', 'data/processed')
    classes = config.get('dataset.classes', ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma'])
    
    # Check if data exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please run preprocessing first: python -m src.preprocessing")
        exit(1)
    
    # Train a single model (for testing)
    logger.info("Starting training pipeline...")
    
    model, history = train_model(
        model_name='cnn_baseline',
        data_dir=data_dir,
        classes=classes,
        training_config={
            'batch_size': 32,
            'epochs': 5,  # Reduced for testing
            'learning_rate': 0.001
        }
    )
    
    logger.info("\nTraining complete!")
