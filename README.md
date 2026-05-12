# Eye Diseases Classification Using Deep Learning

## Overview

This project implements an automated system for classifying retinal images into four categories: **Normal**, **Diabetic Retinopathy**, **Cataract**, and **Glaucoma**. The system utilizes deep learning techniques to assist in early detection and diagnosis of eye diseases from fundus photographs.

## Dataset Description

**Dataset Source:** [Eye Diseases Classification on Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

The dataset consists of retinal fundus images collected from multiple medical sources including IDRiD, Ocular Recognition, and HRF databases.

| Class | Description | Approximate Count |
|-------|-------------|-------------------|
| **Normal** | Healthy retinal images without any disease | ~1,074 images |
| **Diabetic Retinopathy** | Images showing diabetic retinopathy symptoms | ~1,000 images |
| **Cataract** | Images showing cataract disease | ~500+ images |
| **Glaucoma** | Images showing glaucoma disease | ~760 images |

### Dataset Structure

```
dataset/
├── cataract/              # Cataract retinal images
│   ├── *.jpg
│   └── *.png
├── diabetic_retinopathy/  # Diabetic retinopathy images
│   └── *.jpeg
├── glaucoma/              # Glaucoma retinal images
│   ├── *.jpg
│   └── *.png
└── normal/                # Normal retinal images
    └── *.jpg
```

<<<<<<< HEAD

=======
>>>>>>> dev1/data-pipeline
### Image Specifications

- **Format**: JPG, JPEG, PNG
- **Source**: Multiple medical imaging databases
- **Type**: Color fundus photography
- **Resolution**: Variable (original dataset resolutions preserved)

## System Architecture

### Pipeline Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EYE DISEASE CLASSIFICATION PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   INPUT      │───▶│  DATA        │───▶│  MODEL       │───▶│   OUTPUT     │
│   LAYER      │    │  PROCESSING  │    │  TRAINING    │    │   LAYER      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ • Retinal    │    │ • Image      │    │ • Transfer   │    │ • Disease    │
│   Fundus     │    │   Loading    │    │   Learning     │    │   Class      │
│   Images     │    │ • Resizing   │    │ • CNN        │    │ • Confidence │
│ • Left/Right │    │ • Normaliza- │    │   Architecture │    │   Score      │
│   Eye Images │    │   tion       │    │ • Data       │    │ • Grad-CAM   │
│              │    │ • Augmenta-  │    │   Augmentation │    │   Visualiza- │
│              │    │   tion       │    │ • Training   │    │   tion       │
│              │    │ • Train/Test │    │ • Validation │    │ • Report     │
│              │    │   Split      │    │ • Fine-tuning│    │   Generation │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Detailed Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         DETAILED PROCESSING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │  RAW DATA INPUT │
    │  (4 Classes)    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  DATA LOADER    │────▶│  IMAGE CHECK    │────▶│  CORRUPTION     │
    │  • Read images  │     │  • Verify format│     │  HANDLING       │
    │  • Extract paths│     │  • Check size   │     │  • Remove bad   │
    │  • Label encode │     │  • Validate     │     │  • Log errors   │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
             │
             ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  PREPROCESSING  │────▶│  IMAGE RESIZE   │────▶│  PIXEL          │
    │  PIPELINE       │     │  (224×224 or    │     │  NORMALIZATION  │
    │                 │     │   299×299)      │     │  [0,1] or       │
    │                 │     │                 │     │  [-1,1]         │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
             │
             ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  DATA           │────▶│  TRAIN/VAL/TEST │────▶│  DATA           │
    │  AUGMENTATION   │     │  SPLIT          │     │  GENERATORS     │
    │  • Rotation     │     │  (70/15/15 or   │     │  • Batch        │
    │  • Flip         │     │   80/10/10)     │     │    processing   │
    │  • Zoom         │     │                 │     │  • Shuffling    │
    │  • Brightness   │     │                 │     │  • Prefetching  │
    │  • Contrast     │     │                 │     │                 │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     MODEL ARCHITECTURE SELECTION                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
    │   │   CNN       │   │  Transfer   │   │   Vision    │           │
    │   │   From      │   │  Learning   │   │   Transformers│         │
    │   │   Scratch   │   │  (ResNet,   │   │   (ViT,     │           │
    │   │             │   │   Efficient-│   │   Swin)     │           │
    │   │             │   │   Net,      │   │             │           │
    │   │             │   │   VGG)      │   │             │           │
    │   └─────────────┘   └─────────────┘   └─────────────┘           │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  MODEL TRAINING │────▶│  CALLBACKS      │────▶│  MODEL          │
    │                 │     │                 │     │  EVALUATION     │
    │  • Forward pass │     │  • Early        │     │                 │
    │  • Loss calc    │     │    stopping     │     │  • Accuracy     │
    │  • Backprop     │     │  • Model        │     │  • Precision    │
    │  • Optimizer    │     │    checkpoint   │     │  • Recall       │
    │    step         │     │  • LR reduction │     │  • F1-score     │
    │  • Epoch loop   │     │  • TensorBoard  │     │  • Confusion    │
    │                 │     │    logging      │     │    matrix       │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
             │
             ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  INFERENCE      │────▶│  EXPLAINABILITY │────▶│  DEPLOYMENT     │
    │  PIPELINE       │     │  (Grad-CAM)     │     │  (Optional)     │
    │                 │     │                 │     │                 │
    │  • Load model   │     │  • Heatmap      │     │  • API          │
    │  • Preprocess   │     │    generation   │     │    endpoint     │
    │  • Predict      │     │  • Region       │     │  • Web app      │
    │  • Decode label │     │    highlighting │     │  • Mobile app   │
    │  • Return prob  │     │  • Clinical     │     │                 │
    │                 │     │    relevance    │     │                 │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Technical Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 5 GB | 10+ GB (SSD) |
| **GPU** | Optional | NVIDIA GPU with 8+ GB VRAM |
| **OS** | Windows/Linux/macOS | Linux (Ubuntu 20.04+) |

### Software Dependencies

```
Python >= 3.8
TensorFlow >= 2.10.0  or  PyTorch >= 1.12.0
OpenCV >= 4.5.0
NumPy >= 1.21.0
Pandas >= 1.3.0
Matplotlib >= 3.4.0
Seaborn >= 0.11.0
Scikit-learn >= 1.0.0
Pillow >= 8.0.0
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Eye-diseases-classification

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Eye-diseases-classification/
│
├── dataset/                          # Dataset directory
│   ├── cataract/
│   ├── diabetic_retinopathy/
│   ├── glaucoma/
│   └── normal/
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_loader.py                # Data loading utilities
│   ├── preprocessing.py              # Image preprocessing
│   ├── augmentation.py                 # Data augmentation
│   ├── model.py                      # Model architectures
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   ├── predict.py                    # Inference script
│   └── utils.py                      # Helper functions
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── models/                           # Saved models
│   ├── checkpoints/
│   └── final/
│
├── results/                          # Training results
│   ├── plots/
│   ├── metrics/
│   └── confusion_matrices/
│
├── configs/                          # Configuration files
│   └── config.yaml
│
├── tests/                            # Unit tests
│   └── test_model.py
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore file
```

## Model Architectures

### Supported Architectures

1. **Convolutional Neural Networks (CNN)**
   - Custom CNN from scratch
   - Multi-layer architecture with batch normalization

2. **Transfer Learning Models**
   - ResNet50 / ResNet101
   - EfficientNetB0-B7
   - VGG16 / VGG19
   - InceptionV3
   - DenseNet121

3. **Vision Transformers (Optional)**
   - ViT (Vision Transformer)
   - Swin Transformer

### Recommended Configuration

```python
# Input shape
INPUT_SHAPE = (224, 224, 3)

# Number of classes
NUM_CLASSES = 4

# Batch size
BATCH_SIZE = 32

# Epochs
EPOCHS = 50

# Learning rate
LEARNING_RATE = 0.0001

# Optimizer
OPTIMIZER = 'Adam'

# Loss function
LOSS_FUNCTION = 'categorical_crossentropy'
```

## Data Preprocessing Pipeline

### 1. Image Loading
- Read images using OpenCV or PIL
- Handle different image formats (JPG, JPEG, PNG)
- Convert to RGB color space

### 2. Preprocessing Steps

| Step | Description | Parameters |
|------|-------------|------------|
| **Resize** | Resize to target dimensions | (224, 224) or (299, 299) |
| **Normalization** | Scale pixel values | [0, 1] or [-1, 1] |
| **Augmentation** | Artificial data expansion | See below |

### 3. Data Augmentation Strategy

```python
augmentation_config = {
    "rotation_range": 20,           # Rotate up to 20 degrees
    "width_shift_range": 0.1,       # Horizontal shift
    "height_shift_range": 0.1,      # Vertical shift
    "horizontal_flip": True,        # Flip horizontally
    "vertical_flip": False,         # No vertical flip (medical images)
    "zoom_range": 0.1,              # Zoom in/out
    "brightness_range": [0.8, 1.2], # Brightness adjustment
    "fill_mode": "nearest"          # Fill mode for new pixels
}
```

## Training Pipeline

### 1. Data Splitting

```
Total Dataset: 100%
├── Training Set:   70-80%
├── Validation Set: 10-15%
└── Test Set:       10-15%
```

### 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 16-32 (based on GPU memory) |
| **Epochs** | 50-100 (with early stopping) |
| **Initial LR** | 1e-4 |
| **Optimizer** | Adam |
| **Metrics** | Accuracy, Precision, Recall, F1 |

### 3. Callbacks

- **ModelCheckpoint**: Save best model based on validation accuracy
- **EarlyStopping**: Stop training if no improvement for 10 epochs
- **ReduceLROnPlateau**: Reduce learning rate when validation loss plateaus
- **TensorBoard**: Visualize training metrics

## Evaluation Metrics

### Classification Metrics

| Metric | Description | Target Value |
|--------|-------------|--------------|
| **Accuracy** | Overall correct predictions | > 90% |
| **Precision** | True positives / Predicted positives | > 85% |
| **Recall** | True positives / Actual positives | > 85% |
| **F1-Score** | Harmonic mean of precision and recall | > 85% |

### Visualization Outputs

- **Confusion Matrix**: Class-wise prediction accuracy
- **ROC Curves**: True positive vs False positive rates
- **Precision-Recall Curves**: Per-class performance
- **Grad-CAM Heatmaps**: Model attention visualization

## Usage Examples

### Training

```bash
# Train with default configuration
python src/train.py

# Train with custom configuration
python src/train.py --config configs/config.yaml

# Train specific model
python src/train.py --model efficientnet_b0 --epochs 100
```

### Evaluation

```bash
# Evaluate trained model
python src/evaluate.py --model_path models/final/model.h5

# Generate detailed report
python src/evaluate.py --model_path models/final/model.h5 --report
```

### Inference

```bash
# Predict single image
python src/predict.py --image path/to/image.jpg --model models/final/model.h5

# Predict batch
python src/predict.py --input_dir path/to/images/ --output results/predictions.csv
```

## Expected Results

### Target Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ResNet50 | ~92-95% | ~90-93% | ~90-93% | ~90-93% |
| EfficientNet-B0 | ~94-97% | ~92-95% | ~92-95% | ~92-95% |
| Custom CNN | ~85-90% | ~83-88% | ~83-88% | ~83-88% |

### Confusion Matrix (Expected)

```
                    Predicted
              Normal   DR    Cataract  Glaucoma
Actual Normal    ████    ░░      ░░        ░░
        DR        ░░    ████     ░░        ░░
        Cataract  ░░     ░░     ████       ░░
        Glaucoma  ░░     ░░      ░░       ████
```

## Future Enhancements

1. **Multi-Modal Integration**: Incorporate OCT scans alongside fundus images
2. **Severity Grading**: Classify disease severity levels (mild, moderate, severe)
3. **Real-Time Inference**: Optimize for mobile and edge devices
4. **Federated Learning**: Train on distributed medical datasets
5. **Explainable AI**: Enhance Grad-CAM with region-specific clinical annotations

## References

1. [Kaggle Dataset: Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
2. [IDRiD Dataset](https://idrid.grand-challenge.org/)
3. [Ocular Recognition Dataset](https://www.kaggle.com/datasets/venkatdoddi/ocular-disease-recognition-odir5k)
4. [HRF Dataset](https://www.kaggle.com/datasets/valentinablanco1/hrf-dataset)

## License

This project is intended for educational and research purposes. Medical images used are from publicly available datasets with appropriate licenses.

---
<<<<<<< HEAD
**Note**: This system is designed for research and educational purposes. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.
=======

**Note**: This system is designed for research and educational purposes. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.
>>>>>>> dev1/data-pipeline
