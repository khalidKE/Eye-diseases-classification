# Workflow & Team Collaboration Guide

## Team Structure: 3 Developers Working in Parallel

This document defines how the 3 team members collaborate on the Eye Disease Classification project.

---

## Quick Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PROJECT WORKFLOW (Parallel Execution)                    │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │   DEVELOPER 1   │      │   DEVELOPER 2   │      │   DEVELOPER 3   │
    │  (Data Pipeline)│      │  (Model Training)│      │ (Eval & Deploy) │
    └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
             │                        │                        │
    Day 1-3  │  Data Pipeline         │  Model Architecture    │  Evaluation Framework
             │  ↓                     │  ↓                     │  ↓
    Day 4-6  │  Preprocessing         │  Transfer Learning     │  Metrics & Visualization
             │  ↓                     │  ↓                     │  ↓
    Day 7-9  │  Data Splitting        │  Training Loop         │  Inference Pipeline
             │  ↓                     │  ↓                     │  ↓
    Day 10-12│  Data Augmentation     │  Hyperparameter Tuning │  Streamlit App
             │                        │                        │
             └──────────┬───────────────┴──────────┬───────────────┘
                        │                          │
                        ▼                          ▼
              ┌──────────────────┐     ┌──────────────────┐
              │  INTEGRATION     │────▶│  FINAL TESTING   │
              │  Week 2          │     │  & DEPLOYMENT    │
              └──────────────────┘     └──────────────────┘
```

---

## Developer 1: Data Pipeline & Preprocessing Engineer

### Task 1: Data Acquisition & Setup
**Timeline:** Day 1-2
- [ ] Download dataset from Kaggle
- [ ] Verify dataset integrity (all 4 classes present)
- [ ] Create directory structure
- [ ] Document dataset statistics

**Output:**
- `data/raw/` folder with all images organized
- `notebooks/01_dataset_verification.ipynb`

### Task 2: Data Preprocessing Pipeline
**Timeline:** Day 3-5
- [ ] Image loading function (handle JPG/PNG/JPEG)
- [ ] Image resizing (224×224 or 299×299)
- [ ] Pixel normalization ([0,1] or [-1,1])
- [ ] Corrupt image detection & removal
- [ ] Label encoding (0: Normal, 1: Diabetic, 2: Cataract, 3: Glaucoma)

**Output:**
- `src/data_loader.py` - Complete data loading module
- `src/preprocessing.py` - Preprocessing functions

### Task 3: Data Splitting & Augmentation
**Timeline:** Day 6-8
- [ ] Train/Validation/Test split (70/15/15)
- [ ] Data augmentation pipeline:
  - Rotation (±15°)
  - Horizontal flip
  - Brightness/contrast adjustment
  - Zoom (0.9-1.1x)
- [ ] TF Dataset or PyTorch DataLoader implementation
- [ ] Class balancing (if needed)

**Output:**
- `src/data_loader.py` - DataLoader classes
- `data/processed/` - Cleaned and augmented data ready for training

### Dependencies:
- **Blocks:** Nothing (starts immediately)
- **Blocks others:** Developer 2 needs sample data by Day 3

---

## Developer 2: Model Development & Training Engineer

### Task 1: Model Architecture Design
**Timeline:** Day 1-3
- [ ] Implement baseline CNN from scratch
- [ ] Implement ResNet50 (Transfer Learning)
- [ ] Implement EfficientNetB0 (Transfer Learning)
- [ ] Model summary and parameter count

**Output:**
- `src/models.py` - All model architectures
- `notebooks/02_model_architectures.ipynb`

### Task 2: Training Pipeline
**Timeline:** Day 4-7
- [ ] Training loop implementation
- [ ] Loss function (Categorical Crossentropy)
- [ ] Optimizer (Adam with learning rate scheduling)
- [ ] Callbacks:
  - ModelCheckpoint (save best model)
  - EarlyStopping (patience=5-10)
  - ReduceLROnPlateau
- [ ] TensorBoard logging setup

**Output:**
- `src/train.py` - Complete training script
- `logs/` - Training logs directory

### Task 3: Hyperparameter Tuning & Multi-Model Training
**Timeline:** Day 8-12
- [ ] Train 3 models: CNN, ResNet50, EfficientNet
- [ ] Hyperparameter grid search:
  - Learning rates: [1e-3, 1e-4, 1e-5]
  - Batch sizes: [16, 32, 64]
- [ ] Compare model performance
- [ ] Save best model for each architecture

**Output:**
- `models/` - All trained model files (.h5 or .pth)
- `results/model_comparison.csv`
- `notebooks/03_training_results.ipynb`

### Dependencies:
- **Blocked by:** Developer 1 (needs data structure by Day 3)
- **Uses sample data:** Can start with small subset while waiting
- **Blocks:** Developer 3 needs trained models by Day 10

---

## Developer 3: Evaluation & Deployment Engineer

### Task 1: Evaluation Framework
**Timeline:** Day 1-4
- [ ] Metrics implementation:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC Curve & AUC
- [ ] Per-class metrics calculation
- [ ] Visualization functions for all metrics

**Output:**
- `src/evaluation.py` - All evaluation functions
- `notebooks/04_evaluation_framework.ipynb`

### Task 2: Inference Pipeline & Explainability
**Timeline:** Day 5-8
- [ ] Single image prediction function
- [ ] Batch prediction function
- [ ] Grad-CAM implementation for visualization
- [ ] Prediction confidence scoring
- [ ] Report generation (PDF/HTML)

**Output:**
- `src/predict.py` - Inference script
- `src/gradcam.py` - Grad-CAM visualization
- `results/gradcam_samples/` - Sample visualizations

### Task 3: Web Application & Final Integration
**Timeline:** Day 9-14
- [ ] Streamlit app with:
  - Image upload
  - Prediction display
  - Confidence scores
  - Grad-CAM overlay
- [ ] API endpoint (optional - Flask/FastAPI)
- [ ] Final integration testing
- [ ] Documentation & demo video

**Output:**
- `src/app.py` - Streamlit application
- `app/` - Web app folder (if separate)
- `demo.mp4` - Demo recording

### Dependencies:
- **Blocked by:** Developer 2 (needs trained models by Day 10)
- **Can start early:** Evaluation framework independent
- **Parallel work:** Can prepare UI mockups before models ready

---

## Parallel Work Strategy

### What We Do in Parallel:

| Week 1 | Developer 1 | Developer 2 | Developer 3 |
|--------|-------------|-------------|-------------|
| Day 1-2 | Download & verify data | Research architectures | Design evaluation metrics |
| Day 3-4 | Preprocessing pipeline | Start CNN implementation | Implement metrics code |
| Day 5-6 | Data augmentation | Transfer learning setup | Visualization functions |
| Day 7-8 | Final data prep | Begin training (small data) | Inference skeleton code |
| Day 9-10 | Test data loaders | Full model training | Prepare UI mockups |
| Day 11-12 | Integration support | Hyperparameter tuning | Grad-CAM implementation |

### Week 2: Integration & Testing

| Day | Activity |
|-----|----------|
| 13-14 | Combine all components, fix integration issues |
| 15-16 | Full end-to-end testing |
| 17-18 | Bug fixes & optimization |
| 19-20 | Final documentation & presentation prep |

---

## Handoff Points (Critical Sync)

### Sync Point 1: Day 3
**Developer 1 → Developer 2**
- Sample dataset (100 images from each class)
- DataLoader class skeleton
- Directory structure documentation

### Sync Point 2: Day 7
**Developer 1 → Developer 2 (Full)**
- Complete processed dataset
- Final data loaders
- Augmentation pipeline

### Sync Point 3: Day 10
**Developer 2 → Developer 3**
- Best trained model file
- Model input/output specification
- Prediction function template

### Sync Point 4: Day 14
**All Developers**
- Code integration
- End-to-end pipeline test
- Bug identification

---

## What We DO

### Technical Stack (YES)
- ✅ Python 3.8+
- ✅ TensorFlow 2.x OR PyTorch 1.12+
- ✅ OpenCV for image processing
- ✅ Transfer Learning (ResNet, EfficientNet)
- ✅ Data Augmentation (Albumentations or Keras)
- ✅ Streamlit for web app
- ✅ Git for version control
- ✅ Jupyter Notebooks for exploration

### Models (YES)
- ✅ CNN from scratch (baseline)
- ✅ ResNet50 (Transfer Learning)
- ✅ EfficientNetB0/B3
- ✅ Fine-tuning pretrained models

### Evaluation (YES)
- ✅ Confusion Matrix
- ✅ Precision, Recall, F1-Score
- ✅ ROC-AUC
- ✅ Grad-CAM visualization
- ✅ Per-class accuracy

### Deliverables (YES)
- ✅ Trained model files
- ✅ Training history & logs
- ✅ Evaluation reports
- ✅ Web application
- ✅ Documentation (README, this workflow)

---

## What We DON'T DO

### Models (NO)
- ❌ VGG16 (too heavy, worse than ResNet)
- ❌ Vision Transformers (ViT) - complex for beginners
- ❌ Ensemble methods (time constraint)
- ❌ Custom architectures (stick to proven ones)

### Techniques (NO)
- ❌ Object detection (we do classification only)
- ❌ Segmentation (not required)
- ❌ 3D CNN (we have 2D images)
- ❌ Federated learning (too complex)

### Scope (NO)
- ❌ Mobile app (web app only)
- ❌ Cloud deployment (local/demo only)
- ❌ Real-time camera integration (file upload only)
- ❌ Multi-dataset training (use provided dataset only)
- ❌ Medical certification (research project only)

---

## Communication & Collaboration

### Daily Standup (15 min)
- What did I complete yesterday?
- What will I work on today?
- Any blockers?

### Shared Resources
```
project-root/
├── shared/
│   ├── common_utils.py       # Shared utility functions
│   ├── config.yaml           # Common configuration
│   └── constants.py          # Shared constants
```

### Git Workflow
```
main branch (stable)
  ├── dev1/data-pipeline     (Developer 1)
  ├── dev2/model-training    (Developer 2)
  └── dev3/evaluation        (Developer 3)
```

**Rule:** Pull request to `main` requires at least 1 reviewer approval.

---

## Summary

### Developer 1 (Data Pipeline)
**Goal:** Provide clean, augmented, ready-to-train data
**Key Outputs:** `data_loader.py`, `preprocessing.py`, processed dataset
**Timeline:** 8-10 days

### Developer 2 (Model Training)
**Goal:** Train 3 high-performing models with proper evaluation during training
**Key Outputs:** `models.py`, `train.py`, trained model files, comparison results
**Timeline:** 10-12 days

### Developer 3 (Evaluation & Deployment)
**Goal:** Evaluate models, create explainable predictions, build web interface
**Key Outputs:** `evaluation.py`, `predict.py`, `app.py`, Grad-CAM visualizations
**Timeline:** 12-14 days

### Project Timeline
- **Week 1:** Parallel development
- **Week 2:** Integration, testing, documentation
- **Total:** 2 weeks for MVP, 3 weeks for polished product

### Success Criteria
- [ ] Model achieves >85% accuracy on test set
- [ ] Working web app with upload + prediction + visualization
- [ ] Complete documentation (README + workflow + code comments)
- [ ] All code in GitHub with proper commit history

---

## Emergency Plan (If Someone Falls Behind)

### If Developer 1 is delayed:
- Developer 2 uses public datasets (e.g., CIFAR-10) temporarily
- Developer 2 uses Kaggle's built-in dataset loader

### If Developer 2 is delayed:
- Developer 3 uses pre-trained models from Kaggle/TensorFlow Hub
- Focus on evaluation and deployment with existing models

### If Developer 3 is delayed:
- Developer 1 & 2 help with evaluation code
- Simplify web app (Streamlit is already simple)
