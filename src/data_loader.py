import os
import glob
import yaml
import tensorflow as tf
from sklearn.model_selection import train_test_split
from augmentation import augment_image, get_batch_augmentations

def load_config(config_path="shared/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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
        # Apply image-level augmentations from our module
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size)
    
    # Apply batch-level augmentations (rotation/zoom) if training
    if is_training:
        batch_augment = get_batch_augmentations()
        dataset = dataset.map(batch_augment, num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def get_data_generators(config_path="configs/config.yaml"):
    """
    Splits the dataset 70/15/15 and returns train, val, and test tf.data.Datasets.
    """
    config = load_config(config_path)
    dataset_dir = config['data']['dataset_dir']
    batch_size = config['data']['batch_size']
    classes = config['data']['classes']
    
    print(f"Loading data paths from {dataset_dir}...")
    paths, labels = get_image_paths_and_labels(dataset_dir, classes)
    
    if not paths:
        print(f"Warning: No images found in {dataset_dir}. Returning empty datasets.")
        return None, None, None

    # First split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    
    # Second split: Split the 30% Temp into 15% Val and 15% Test (50/50 split of Temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"Data Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    train_ds = create_tf_dataset(X_train, y_train, batch_size, is_training=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size, is_training=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size, is_training=False)
    
    # To maintain compatibility with train.py which currently unpacks two generators, 
    # we return train and val. If train.py expects test_ds later, we return all three.
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_data_generators()
    if train_ds:
        print("Data Loaders initialized successfully.")
        for images, labels in train_ds.take(1):
            print(f"Image batch shape: {images.shape}")
            print(f"Label batch shape: {labels.shape}")
