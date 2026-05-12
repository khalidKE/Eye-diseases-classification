"""
Configuration management for the Eye Disease Classification project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """Configuration class for managing project settings."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Key in dot notation (e.g., 'dataset.image_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.config.get('dataset', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_classes(self) -> List[str]:
        """Get list of disease classes."""
        return self.config.get('dataset', {}).get('classes', [])
    
    def get_class_labels(self) -> Dict[str, int]:
        """Get class label mappings."""
        return self.config.get('dataset', {}).get('class_labels', {})
    
    def get_image_size(self) -> int:
        """Get target image size."""
        return self.config.get('dataset', {}).get('image_size', 224)
    
    def get_batch_size(self) -> int:
        """Get training batch size."""
        return self.config.get('training', {}).get('batch_size', 32)
    
    def get_epochs(self) -> int:
        """Get number of training epochs."""
        return self.config.get('training', {}).get('epochs', 50)
    
    def create_directories(self):
        """Create necessary project directories."""
        dirs = [
            self.config.get('dataset', {}).get('raw_data_path', 'data/raw'),
            self.config.get('dataset', {}).get('processed_data_path', 'data/processed'),
            self.config.get('paths', {}).get('models', 'models'),
            self.config.get('paths', {}).get('logs', 'logs'),
            self.config.get('paths', {}).get('results', 'results'),
            self.config.get('paths', {}).get('notebooks', 'notebooks'),
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        return True


# Global config instance
_config = None

def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get or create global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


# Constants
CLASS_NAMES = ["normal", "diabetic_retinopathy", "cataract", "glaucoma"]
CLASS_NAMES_DISPLAY = {
    "normal": "Normal",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "cataract": "Cataract",
    "glaucoma": "Glaucoma"
}

# Color mappings for visualization
CLASS_COLORS = {
    "normal": "#2ecc71",  # Green
    "diabetic_retinopathy": "#e74c3c",  # Red
    "cataract": "#f39c12",  # Orange
    "glaucoma": "#9b59b6"  # Purple
}
