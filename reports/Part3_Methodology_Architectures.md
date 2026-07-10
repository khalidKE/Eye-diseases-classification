---
title: "Proposed Methodology & Architectures"
part: 3
scientific_writing: "Part 3 — Contents of Scientific Articles"
study_file: src/models.py
section: 7
output_pdf: Part3_Methodology_Architectures.pdf
---

# Part 3: Proposed Methodology & Architectures

**Scientific Writing — Part 3: Methodology Section of a Research Article**

| Field | Detail |
|-------|--------|
| **Focus** | Experimental design · CNN · Transfer learning |
| **Study File** | `src/models.py` · `src/train.py` |
| **Architectures** | CNN Baseline · ResNet50 · EfficientNetB0 |
| **Key Techniques** | Conv layers · Batch Normalization · Dropout · Compound Scaling |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Experimental Design Framework](#2-experimental-design-framework)
3. [Proposed Pipeline](#3-proposed-pipeline)
4. [Architecture 1 — Baseline CNN](#4-architecture-1--baseline-cnn-from-scratch)
5. [Architecture 2 — ResNet50 Transfer Learning](#5-architecture-2--resnet50-transfer-learning)
6. [Architecture 3 — EfficientNetB0 Compound Scaling](#6-architecture-3--efficientnetb0-compound-scaling)
7. [Training Methodology](#7-training-methodology)
8. [Implementation Reference (src/models.py)](#8-implementation-reference-srcmodelspy)
9. [References](#references)

---

## 1. Overview

This section presents the **proposed methodology and neural network architectures** for four-class retinal disease classification. We follow standard scientific article structure (Part 3 of *Writing in the Sciences*): clearly defined experimental design, reproducible procedures, and justified architectural choices.

We compare three models implemented in `src/models.py`:

| Model | Type | Key Idea |
|-------|------|----------|
| **CNN Baseline** | From scratch | Conv blocks + BatchNorm + Dropout |
| **ResNet50** | Transfer learning | Residual skip connections |
| **EfficientNetB0** | Transfer learning | Compound scaling (depth × width × resolution) |

**Research question:** Can deep convolutional architectures reliably classify fundus images into Normal, Diabetic Retinopathy, Cataract, and Glaucoma?

**Hypothesis:** Transfer-learning models (ResNet50, EfficientNetB0) outperform the baseline CNN due to ImageNet-pretrained feature extractors.

---

## 2. Experimental Design Framework

We adopted a controlled experimental framework ensuring fair comparison across architectures.

![Experimental Design](figures/fig_part3_experimental_design.png)

**Figure 1.** Experimental design framework — research question, hypothesis, variables, controls, and procedure.

| Element | Specification |
|---------|---------------|
| **Independent variable** | Model architecture (CNN, ResNet50, EfficientNetB0) |
| **Dependent variables** | Accuracy, precision, recall, F1-score, ROC-AUC |
| **Controls** | Same dataset split (70/15/15), random seed 42, input size 224×224, Adam optimizer, categorical cross-entropy loss |
| **Procedure** | Train on training set → validate → select best checkpoint → evaluate on held-out test set |

---

## 3. Proposed Pipeline

The end-to-end laboratory pipeline integrates preprocessing (Part 1), model training (Part 2/3), and evaluation (Part 3).

![Training Pipeline](figures/fig_part3_training_pipeline.png)

**Figure 2.** End-to-end training pipeline from raw fundus images to saved model checkpoints.

**Pipeline stages:**

1. **Input** — Four-class fundus dataset (Kaggle Eye Diseases Classification).
2. **Preprocessing** — Medical border crop, resize to 224×224, pixel normalization.
3. **Split** — Stratified 70% train / 15% validation / 15% test.
4. **Augmentation** — Rotation (±15°), horizontal flip, zoom (0.9–1.1), brightness (0.8–1.2).
5. **Model selection** — One of three architectures via `create_model()` factory.
6. **Training** — Adam optimizer, callbacks, best-model checkpoint (`.h5`).

---

## 4. Architecture 1 — Baseline CNN (From Scratch)

We built a custom CNN in `create_baseline_cnn()` as a non-pretrained baseline.

![CNN Architecture](figures/fig_part3_cnn_architecture.png)

**Figure 3.** Baseline CNN architecture with four convolutional blocks and regularization layers.

### 4.1 Convolutional Layers (Conv2D)

Each block applies a 3×3 convolution with `padding='same'`, preserving spatial dimensions before pooling:

| Block | Filters | Kernel | Pooling | Output scale |
|-------|--------:|--------|---------|--------------|
| 1 | 32 | 3×3 | MaxPool 2×2 | 112×112 |
| 2 | 64 | 3×3 | MaxPool 2×2 | 56×56 |
| 3 | 128 | 3×3 | MaxPool 2×2 | 28×28 |
| 4 | 256 | 3×3 | MaxPool 2×2 | 14×14 |

Conv layers extract hierarchical features: edges → textures → anatomical patterns.

### 4.2 Batch Normalization

After each Conv2D and Dense layer, **Batch Normalization** normalizes activations across the mini-batch:

\[
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
\]

Benefits: faster convergence, reduced internal covariate shift, allows higher learning rates.

### 4.3 Dropout Regularization

**Dropout** (rate = 0.5) randomly deactivates neurons during training, preventing co-adaptation and overfitting. Applied after both dense layers (512 and 256 units).

### 4.4 Additional Regularization

- **L2 weight regularization** (λ = 0.001) on Conv2D and Dense kernels.
- **Global Average Pooling (GAP)** replaces flattening, reducing parameters and overfitting risk.

### 4.5 Classification Head

```
GAP → Dense(512) → BN → ReLU → Dropout(0.5)
    → Dense(256) → BN → ReLU → Dropout(0.5)
    → Dense(4, softmax)
```

**Approximate parameters:** ~2–3 million trainable.

---

## 5. Architecture 2 — ResNet50 Transfer Learning

ResNet50 (He et al., 2016) uses **residual skip connections** that add input directly to block output:

\[
y = F(x) + x
\]

This solves the vanishing gradient problem in very deep networks.

![Transfer Learning](figures/fig_part3_transfer_learning.png)

**Figure 4.** Transfer learning architectures — ResNet50 (left) and EfficientNetB0 (right).

### 5.1 Implementation (`create_resnet50_model`)

| Component | Configuration |
|-----------|---------------|
| Base model | ResNet50, ImageNet weights, `include_top=False` |
| Base layers | Frozen by default (`freeze_base=True`) |
| Fine-tuning | Optional: unfreeze last 20 layers |
| Custom head | GAP → BN → Dropout(0.5) → Dense(512) → BN → Dropout(0.5) → Dense(256) → BN → Dropout(0.25) → Softmax(4) |

**Rationale:** ImageNet pretrained filters capture general visual features transferable to medical textures. The custom head adapts to four retinal classes.

**Approximate parameters:** ~25M total; ~1–2M trainable with frozen base.

---

## 6. Architecture 3 — EfficientNetB0 Compound Scaling

EfficientNetB0 (Tan & Le, 2019) applies **compound scaling** — uniformly scaling network depth, width, and input resolution:

\[
\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi
\]

subject to \(\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2\).

### 6.1 Key Design Features

| Feature | EfficientNetB0 | ResNet50 |
|---------|----------------|----------|
| Building block | MBConv (mobile inverted bottleneck) | Residual blocks |
| Activation (head) | Swish: \(x \cdot \sigma(x)\) | ReLU |
| Dropout (head) | 0.3 | 0.5 |
| Parameters | ~5M | ~25M |
| Efficiency | Higher accuracy per FLOP | Deeper, heavier |

### 6.2 Implementation (`create_efficientnet_model`)

| Component | Configuration |
|-----------|---------------|
| Base model | EfficientNetB0, ImageNet weights, frozen |
| Fine-tuning | Unfreeze last 15 layers (optional) |
| Custom head | GAP → BN → Dropout(0.3) → Dense(256, swish) → BN → Dropout(0.15) → Softmax(4) |

**Rationale:** Compound scaling yields better parameter efficiency — important for future edge-device deployment.

---

## 7. Training Methodology

All models compile via `compile_model()` in `src/models.py`:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 0.001) |
| Loss | Categorical cross-entropy |
| Metrics | Accuracy, precision, recall, weighted F1 |
| Batch size | 32 |
| Epochs | 50 (with early stopping) |

**Callbacks** (`src/train.py`):

| Callback | Monitor | Purpose |
|----------|---------|---------|
| ModelCheckpoint | val_accuracy | Save best weights |
| EarlyStopping | val_loss | Stop if no improvement (patience = 10) |
| ReduceLROnPlateau | val_loss | Reduce lr by 0.2× (patience = 5) |
| TensorBoard | — | Log training curves |
| CSVLogger | — | Export epoch metrics |

**Model factory** — unified entry point:

```python
model = create_model(
    model_name='efficientnet',  # or 'cnn_baseline', 'resnet50'
    input_shape=(224, 224, 3),
    num_classes=4
)
model = compile_model(model, learning_rate=0.001, optimizer_name='adam')
```

---

## 8. Implementation Reference (`src/models.py`)

**Section 7 — Core functions:**

| Function | Lines (approx.) | Role |
|----------|-----------------|------|
| `create_baseline_cnn()` | 17–92 | Custom 4-block CNN |
| `create_resnet50_model()` | 95–147 | ResNet50 + custom head |
| `create_efficientnet_model()` | 150–198 | EfficientNetB0 + custom head |
| `compile_model()` | 201–237 | Optimizer, loss, metrics |
| `create_model()` | 277–305 | Factory by name |

**Design principles applied:**

1. **Modularity** — each architecture in its own function.
2. **Reproducibility** — fixed input shape, documented hyperparameters.
3. **Fair comparison** — identical loss, optimizer, and callback pipeline.
4. **Extensibility** — factory pattern supports adding new architectures.

---

## Summary Table — Architecture Comparison

| Property | CNN Baseline | ResNet50 | EfficientNetB0 |
|----------|-------------|----------|----------------|
| Pretrained | No | ImageNet | ImageNet |
| Conv blocks | 4 custom | 50 residual | MBConv scaled |
| BatchNorm | Yes (all blocks) | Yes (head) | Yes (head) |
| Dropout | 0.5 | 0.5 / 0.25 | 0.3 / 0.15 |
| Activation | ReLU | ReLU | Swish (head) |
| ~Parameters | 2–3M | ~25M | ~5M |
| Use case | Baseline reference | Strong transfer learning | Efficient deployment |

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
2. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*, 6105–6114.
3. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training. *ICML*, 448–456.
4. Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929–1958.
5. Sainani, K. (2020). *Writing in the Sciences* — Part 3: Methods. Stanford Online.

---

*Generate PDF: `python scripts/generate_part3_figures.py && python scripts/export_part3_methodology_pdf.py`*
