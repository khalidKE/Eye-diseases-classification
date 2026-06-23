---
title: "Methodology and Proposed Architecture"
subtitle: "Deep Learning Pipeline for Retinal Disease Classification"
author: "Research Engineering Team"
date: "June 2026"
---

# 3. Methodology and Proposed Architecture 

## 3.1 Overview of the Proposed Pipeline 

This work proposes a complete, end-to-end deep learning pipeline for the automated classification of retinal fundus photographs into four clinically relevant categories: Normal, Diabetic Retinopathy (DR), Cataract, and Glaucoma. The pipeline is composed of five sequential stages: (1) data ingestion and preprocessing, (2) data augmentation, (3) model construction and training via three distinct architectures, (4) model evaluation, and (5) explainability via Gradient-weighted Class Activation Mapping (Grad-CAM). Figure 1 depicts the overall architecture of the proposed system. 

![Figure 1: Overall Architecture of the Proposed System](file:///C:/Users/Arzek/.gemini/antigravity/brain/33b9708f-fa24-4f94-9211-a3dbb0742b0e/detailed_processing_flow_1778789991001.png)
*Figure 1: Overall Architecture of the Proposed System*

---

## 3.2 Image Preprocessing Pipeline 

A standardized preprocessing pipeline was developed in `src/preprocessing.py` to prepare raw fundus images for the deep learning models. The pipeline consists of three sub-stages: cropping, resizing, and pixel normalization.

![Figure 4: Image Preprocessing Pipeline](file:///C:/Users/Arzek/.gemini/antigravity/brain/33b9708f-fa24-4f94-9211-a3dbb0742b0e/augmentation_techniques_1781216577054.png)
*Figure 4: Image Preprocessing and Augmentation Pipeline*

---

## 3.3 Model Architectures 

Three distinct deep learning architectures were designed, implemented, and benchmarked. All architectures produce a 4-dimensional softmax output vector $\hat{y} \in \mathbb{R}^4$ where each component $\hat{y}_k$ represents the posterior probability of class $k$. 

### 3.3.1 Architecture I — Baseline Convolutional Neural Network (CNN) 
 
A custom CNN trained from scratch serves as the baseline for comparison against transfer learning models. The architecture consists of four sequential convolutional blocks, each following the design pattern: 

$$ \text{Conv2D}(f, 3\times3, \text{same}) \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{MaxPool}(2\times2) \quad \text{(Eq. 4)} $$

with filter counts $f \in \{32, 64, 128, 256\}$ for blocks 1–4 respectively. L2 weight regularization ($\lambda = 0.001$) is applied to all convolutional kernels. Global Average Pooling (GAP) replaces conventional flattening after the final block.

![Figure 2: Baseline CNN Architecture](file:///C:/Users/Arzek/.gemini/antigravity/brain/33b9708f-fa24-4f94-9211-a3dbb0742b0e/cnn_4_stage_model_1781217191712.png)
*Figure 2: Baseline CNN Architecture*

**Table 3. Baseline CNN Layer Summary.**

| Layer Block | Operation | Output Shape | Filters |
| :--- | :--- | :--- | :--- |
| **Block 1** | Conv2D + BN + ReLU + MaxPool | $112 \times 112 \times 32$ | 32 |
| **Block 2** | Conv2D + BN + ReLU + MaxPool | $56 \times 56 \times 64$ | 64 |
| **Block 3** | Conv2D + BN + ReLU + MaxPool | $28 \times 28 \times 128$ | 128 |
| **Block 4** | Conv2D + BN + ReLU + MaxPool | $14 \times 14 \times 256$ | 256 |
| **GAP** | GlobalAveragePooling2D | $256$ | — |
| **FC-1** | Dense + BN + ReLU + Dropout(0.5) | $512$ | — |
| **FC-2** | Dense + BN + ReLU + Dropout(0.5) | $256$ | — |
| **Output** | Dense + Softmax | $4$ | — |

---
 
### 3.3.2 Architecture II — ResNet50 with Transfer Learning 

The second architecture leverages ResNet50 [5], a 50-layer residual network pre-trained on ImageNet-1K (1.28 million images, 1,000 classes). ResNet50 introduces skip connections that allow gradients to bypass one or more layers, mitigating the vanishing gradient problem. The residual mapping is defined as: 

$$ y = F(x, \{W_i\}) + x $$

where $F(x, \{W_i\})$ is the residual mapping learned by the stacked layers and $x$ is the identity shortcut connection. A two-phase training protocol is used: 
1. **Feature Extraction** — all base layers frozen, only the custom head is trained.
2. **Fine-Tuning** — the last 20 ResNet50 layers are unfrozen and trained jointly with the classification head at a reduced learning rate for domain adaptation. 

The custom classification head appended to the ResNet50 backbone is: 

$$ \text{GAP} \rightarrow \text{BN} \rightarrow \text{Drop}(0.5) \rightarrow \text{Dense}(512) \rightarrow \text{BN} \rightarrow \text{Drop}(0.5) \rightarrow \text{Dense}(256) \rightarrow \text{BN} \rightarrow \text{Drop}(0.25) \rightarrow \text{Softmax}(4) $$

![Figure 3: ResNet50 Architecture](file:///C:/Users/Arzek/.gemini/antigravity/brain/33b9708f-fa24-4f94-9211-a3dbb0742b0e/resnet50_architecture_1781217201899.png)
*Figure 3: Transfer Learning Architecture (ResNet50)*

---

### 3.3.3 Architecture III — EfficientNetB0 with Compound Scaling 

The third architecture is based on EfficientNetB0 [6], constructed using Neural Architecture Search (NAS) and scaled via a compound coefficient $\phi$ that simultaneously scales network width ($w$), depth ($d$), and input resolution ($r$): 

$$ d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi \quad \text{s.t.} \quad \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \ge 1, \beta \ge 1, \gamma \ge 1 $$

For EfficientNetB0, $\phi=0$ with $\alpha=1.2$, $\beta=1.1$, $\gamma=1.1$. The core building block is the Mobile Inverted Bottleneck (MBConv) combined with a Squeeze-and-Excitation (SE) attention module that adaptively recalibrates channel-wise feature responses: 

$$ \hat{x}_c = \sigma(W_2 \cdot \delta(W_1 \cdot z_c)) \cdot x_c $$

where $z_c$ is the global average-pooled representation of channel $c$, $\delta$ is ReLU, and $\sigma$ is the sigmoid activation. The custom head uses the Swish activation function [$\text{Swish}(x) = x \cdot \sigma(x)$], consistent with the backbone's activation: 

$$ \text{GAP} \rightarrow \text{BN} \rightarrow \text{Drop}(0.3) \rightarrow \text{Dense}(256, \text{Swish}) \rightarrow \text{BN} \rightarrow \text{Drop}(0.15) \rightarrow \text{Softmax}(4) \quad \text{(Eq. 9)} $$

![EfficientNet Compound Scaling](file:///C:/Users/Arzek/.gemini/antigravity/brain/33b9708f-fa24-4f94-9211-a3dbb0742b0e/efficientnet_compound_scaling_1781217220510.png)
*EfficientNet Compound Scaling Visualization*

---

## 3.4 Training Configuration and Optimization 

All three models are compiled with categorical cross-entropy loss, the standard loss function for multi-class classification with one-hot encoded labels: 

$$ \mathcal{L} = -\frac{1}{N} \sum_i \sum_k y_{ik} \cdot \log(\hat{y}_{ik}) $$

where $N$ is the number of samples, $K=4$ is the number of classes, $y_{ik}$ is the ground-truth one-hot label, and $\hat{y}_{ik}$ is the predicted probability. The Adam optimizer is employed with an initial learning rate $\eta = 1 \times 10^{-3}$. 

**Table: Training Callbacks and Configuration.**

| Callback | Monitored Metric | Key Parameter |
| :--- | :--- | :--- |
| **ModelCheckpoint** | `val_accuracy` | `save_best_only=True`, `mode=max` |
| **EarlyStopping** | `val_loss` | `patience=10`, `restore_best_weights=True` |
| **ReduceLROnPlateau** | `val_loss` | `factor=0.2`, `patience=5`, `min_lr=1e-7` |
| **TensorBoard** | — | `histogram_freq=1`, per-epoch updates |
| **CSVLogger** | — | Append mode, timestamped file |

**Table: General Training Hyperparameters.**

| Hyperparameter | Value |
| :--- | :--- |
| **Batch size** | 32 |
| **Maximum epochs** | 50 (with early stopping) |
| **Initial learning rate** | $1 \times 10^{-3}$ |
| **Optimizer** | Adam |
| **Loss function** | Categorical cross-entropy |
| **Input resolution** | $224 \times 224 \times 3$ |
| **Random seed** | 42 |

---
 
## 3.5 Evaluation Metrics 

Model performance is assessed on the held-out test set ($D_{\text{test}}$, 15%) using the following metrics computed via scikit-learn: 

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

$$ \text{Precision}_k = \frac{TP_k}{TP_k + FP_k}, \quad \text{Recall}_k = \frac{TP_k}{TP_k + FN_k}, \quad \text{F1}_k = \frac{2 \cdot P_k \cdot R_k}{P_k + R_k} $$

$$ \text{ROC-AUC}_{\text{macro}} = \frac{1}{K} \sum_k \text{AUC}(\text{FPR}_k, \text{TPR}_k) $$

Both macro-averaged and weighted-averaged variants of Precision, Recall, and F1 are reported. ROC-AUC is computed under the One-vs-Rest (OvR) multi-class scheme. 

**Table 6. Best Model Performance on Test Set.**

| Metric | Best Model Result | Target |
| :--- | :--- | :--- |
| **Accuracy** | 60.75% | > 90% |
| **Precision (Macro)** | 68.8% | > 85% |
| **Recall (Macro)** | 60.4% | > 85% |
| **F1-Score (Macro)** | 60.8% | > 85% |
| **ROC-AUC (Macro)** | 86.0% | > 90% |

---
 
## 3.6 Explainability via Grad-CAM 

To provide clinical interpretability, Gradient-weighted Class Activation Mapping (Grad-CAM) [7] is implemented in `src/gradcam.py`. Grad-CAM produces a class-discriminative localization map highlighting the spatial regions of the input most influential for a given prediction. 
 
Given the last convolutional feature map $A^k \in \mathbb{R}^{u \times v}$ and the class score $y^c$ (before softmax), the importance weight $\alpha_k^c$ is computed via Global Average Pooling of the gradients: 

$$ \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k} $$

The Grad-CAM heatmap is the ReLU-activated weighted combination of the feature maps: 

$$ L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha_k^c \cdot A^k \right) $$

ReLU retains only activations that positively influence the class score. The heatmap is upsampled to $224 \times 224$ via bilinear interpolation and overlaid on the input image with a transparency factor $\alpha_{\text{blend}} = 0.4$ using the Jet colormap. The last convolutional layer is detected automatically by traversing the model's layers in reverse.
