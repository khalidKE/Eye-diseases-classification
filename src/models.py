"""
Model Architectures for Eye Disease Classification

This module contains:
- Baseline CNN from scratch
- ResNet50 (Transfer Learning)
- EfficientNetB0 (Transfer Learning)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.regularizers import l2


def create_baseline_cnn(
    input_shape=(224, 224, 3),
    num_classes=4,
    dropout_rate=0.5,
    l2_reg=0.001
) -> Model:
    """
    Create a baseline CNN model from scratch.
    
    Architecture:
    - Conv Block 1: 32 filters, 3x3 kernel
    - Conv Block 2: 64 filters, 3x3 kernel
    - Conv Block 3: 128 filters, 3x3 kernel
    - Global Average Pooling
    - Dense: 512 units with dropout
    - Output: 4 classes with softmax
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', 
                      kernel_regularizer=l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Block 4 (deeper)
    x = layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Global Average Pooling instead of Flatten (reduces parameters)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with dropout
    x = layers.Dense(512, kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='baseline_cnn')
    
    return model


def create_resnet50_model(
    input_shape=(224, 224, 3),
    num_classes=4,
    freeze_base=True,
    dropout_rate=0.5
) -> Model:
    """
    Create ResNet50 model with transfer learning.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        freeze_base: Whether to freeze base model weights
        dropout_rate: Dropout rate for custom top layers
        
    Returns:
        Compiled Keras Model
    """
    # Load base model with pretrained ImageNet weights
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model if specified
    if freeze_base:
        base_model.trainable = False
    else:
        # Fine-tune: unfreeze last few layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    
    # Output layer
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name='resnet50_transfer')
    
    return model


def create_efficientnet_model(
    input_shape=(224, 224, 3),
    num_classes=4,
    freeze_base=True,
    dropout_rate=0.3
) -> Model:
    """
    Create EfficientNetB0 model with transfer learning.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        freeze_base: Whether to freeze base model weights
        dropout_rate: Dropout rate for custom top layers
        
    Returns:
        Compiled Keras Model
    """
    # Load base model with pretrained ImageNet weights
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model if specified
    if freeze_base:
        base_model.trainable = False
    else:
        # Fine-tune: unfreeze last few layers
        for layer in base_model.layers[:-15]:
            layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    
    # Output layer
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name='efficientnet_transfer')
    
    return model


def compile_model(
    model: Model,
    learning_rate=0.001,
    optimizer_name='adam'
) -> Model:
    """
    Compile model with appropriate optimizer and loss.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate
        optimizer_name: Optimizer name ('adam', 'sgd', 'rmsprop')
        
    Returns:
        Compiled model
    """
    if optimizer_name.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.F1Score(name='f1_score', average='weighted')
        ]
    )
    
    return model


def get_model_summary(model: Model) -> str:
    """
    Get a string representation of model summary.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary as string
    """
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    model.summary()
    
    sys.stdout = old_stdout
    summary = buffer.getvalue()
    
    return summary


def count_trainable_params(model: Model) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: Keras model
        
    Returns:
        Number of trainable parameters
    """
    return model.count_params()


def create_model(
    model_name: str,
    input_shape=(224, 224, 3),
    num_classes=4,
    **kwargs
) -> Model:
    """
    Factory function to create models by name.
    
    Args:
        model_name: One of 'cnn_baseline', 'resnet50', 'efficientnet'
        input_shape: Input image shape
        num_classes: Number of classes
        **kwargs: Additional arguments for specific models
        
    Returns:
        Keras Model
    """
    model_name = model_name.lower()
    
    if model_name == 'cnn_baseline':
        return create_baseline_cnn(input_shape, num_classes, **kwargs)
    elif model_name == 'resnet50':
        return create_resnet50_model(input_shape, num_classes, **kwargs)
    elif model_name == 'efficientnet' or model_name == 'efficientnet_b0':
        return create_efficientnet_model(input_shape, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Choose from: cnn_baseline, resnet50, efficientnet")


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...\n")
    
    # Test Baseline CNN
    print("="*50)
    print("Baseline CNN")
    print("="*50)
    cnn = create_baseline_cnn()
    cnn.summary()
    print(f"\nTotal parameters: {count_trainable_params(cnn):,}")
    
    # Test ResNet50
    print("\n" + "="*50)
    print("ResNet50 (Transfer Learning)")
    print("="*50)
    resnet = create_resnet50_model()
    print(f"\nTotal parameters: {count_trainable_params(resnet):,}")
    print(f"Trainable parameters: {sum([w.size for w in resnet.trainable_weights]):,}")
    print(f"Non-trainable parameters: {sum([w.size for w in resnet.non_trainable_weights]):,}")
    
    # Test EfficientNet
    print("\n" + "="*50)
    print("EfficientNetB0 (Transfer Learning)")
    print("="*50)
    efficientnet = create_efficientnet_model()
    print(f"\nTotal parameters: {count_trainable_params(efficientnet):,}")
    print(f"Trainable parameters: {sum([w.size for w in efficientnet.trainable_weights]):,}")
    print(f"Non-trainable parameters: {sum([w.size for w in efficientnet.non_trainable_weights]):,}")
    
    print("\n" + "="*50)
    print("All models created successfully!")
    print("="*50)
