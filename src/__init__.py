"""
Eye Disease Classification Package

A deep learning project for classifying retinal images into 4 categories:
- Normal
- Diabetic Retinopathy
- Cataract
- Glaucoma
"""

__version__ = "1.0.0"
__author__ = "Eye Disease Classification Team"

from . import config
from . import data_loader
from . import preprocessing
from . import models
from . import train
from . import evaluation
from . import predict
from . import gradcam
from . import utils

__all__ = [
    "config",
    "data_loader",
    "preprocessing",
    "models",
    "train",
    "evaluation",
    "predict",
    "gradcam",
    "utils",
]
