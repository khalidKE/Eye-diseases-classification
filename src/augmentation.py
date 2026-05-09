import tensorflow as tf

def augment_image(image, label):
    """
    Applies image-level augmentations using tf.image.
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random Brightness (0.8 to 1.2 is roughly a max_delta of 0.2)
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random Contrast (0.8 to 1.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Clip to ensure valid range after brightness and contrast
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def get_batch_augmentations():
    """
    Returns a function that applies batch-level augmentations (Rotation, Zoom) using Keras layers.
    """
    # 15 degrees = ~0.0416 of 360 degrees
    rotation_layer = tf.keras.layers.RandomRotation(factor=15.0/360.0)
    # Zoom (0.9-1.1x) means +/- 10% zoom. factor=0.1
    zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))
    
    def batch_augment(x, y):
        x = rotation_layer(x, training=True)
        x = zoom_layer(x, training=True)
        return x, y
        
    return batch_augment
