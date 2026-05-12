"""
Utility functions for Eye Disease Classification project.
"""

import os
import json
import logging
import random
import numpy as np
import tensorflow as tf
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For TensorFlow GPU
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass


def create_directory_structure(base_path: str = ".") -> Dict[str, str]:
    """
    Create standard project directory structure.
    
    Args:
        base_path: Base directory path
        
    Returns:
        Dictionary mapping directory names to paths
    """
    directories = {
        'data_raw': os.path.join(base_path, 'data', 'raw'),
        'data_processed': os.path.join(base_path, 'data', 'processed'),
        'models': os.path.join(base_path, 'models'),
        'logs': os.path.join(base_path, 'logs'),
        'results': os.path.join(base_path, 'results'),
        'notebooks': os.path.join(base_path, 'notebooks'),
        'src': os.path.join(base_path, 'src'),
        'tests': os.path.join(base_path, 'tests'),
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """
    Load JSON file to dictionary.
    
    Args:
        filepath: JSON file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information.
    
    Returns:
        Dictionary with GPU information
    """
    gpus = tf.config.list_physical_devices('GPU')
    info = {
        'num_gpus': len(gpus),
        'gpus': []
    }
    
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        info['gpus'].append({
            'name': details.get('device_name', 'Unknown'),
            'compute_capability': details.get('compute_capability', None)
        })
    
    return info


def check_gpu_available() -> bool:
    """
    Check if GPU is available for training.
    
    Returns:
        True if GPU available
    """
    return len(tf.config.list_physical_devices('GPU')) > 0


def print_system_info():
    """Print system and library information."""
    import tensorflow as tf
    import platform
    
    print("="*60)
    print("System Information")
    print("="*60)
    print(f"Python version: {platform.python_version()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {check_gpu_available()}")
    
    if check_gpu_available():
        gpu_info = get_gpu_info()
        for i, gpu in enumerate(gpu_info['gpus']):
            print(f"GPU {i}: {gpu['name']}")
    
    print("="*60)
