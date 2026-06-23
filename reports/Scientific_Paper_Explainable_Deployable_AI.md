---
title: "Explainable and Deployable Artificial Intelligence for Retinal Disease Classification"
subtitle: "Original Research Setup — Parts 4 and 5"
document_type: Scientific Manuscript (IMRaD)
study_modules:
  - src/gradcam.py
  - src/app.py
output_pdf: Scientific_Paper.pdf
date: June 2026
---

# Explainable and Deployable Artificial Intelligence for Retinal Disease Classification

**Original Research Setup — Parts 4 and 5**

| | |
|---|---|
| **Focus** | Explainable AI (Grad-CAM) · Deployable AI (Streamlit) |
| **Study Files** | `src/gradcam.py` · `src/app.py` · `src/report_pdf.py` |
| **Deliverable** | `Scientific_Paper.pdf` |

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#1-introduction)
3. [Methods](#2-methods)
   - 2.1 [System Overview](#21-system-overview)
   - 2.2 [Part 4 — Explainable AI with Grad-CAM](#22-part-4--explainable-ai-with-grad-cam)
   - 2.3 [Part 5 — Deployable AI with Streamlit](#23-part-5--deployable-ai-with-streamlit)
4. [Results](#3-results)
5. [Discussion](#4-discussion)
6. [Conclusion](#5-conclusion)
7. [References](#references)
8. [Appendix — Reproducibility](#appendix--reproducibility)

---

## Abstract

**Background.** Deep learning models classify retinal fundus images with growing accuracy, yet clinical adoption requires transparent decisions and practical deployment tools.

**Objective.** We designed and evaluated an explainable and deployable artificial intelligence (AI) pipeline for four-class retinal disease classification (Normal, Diabetic Retinopathy, Cataract, Glaucoma) as Parts 4 and 5 of a multi-stage research project.

**Methods.** We implemented Gradient-weighted Class Activation Mapping (Grad-CAM) in `src/gradcam.py` to visualize convolutional network attention. We deployed inference and explanation through a Streamlit web application in `src/app.py`, featuring drag-and-drop image upload, real-time prediction, Grad-CAM overlays, and downloadable PDF session reports via `src/report_pdf.py`.

**Results.** On a held-out test set (*n* = 637), the best model achieved 60.8% accuracy, 60.8% macro F1-score, and 86.0% macro area under the receiver operating characteristic curve (ROC-AUC). Per-class recall ranged from 38.2% (Glaucoma) to 84.0% (Normal). Grad-CAM heatmaps localized attention to the optic disc and vascular regions consistent with clinical anatomy.

**Conclusions.** Combining Grad-CAM with a Streamlit interface provides a human-in-the-loop workflow suitable for research and educational decision support. Accuracy remains below clinical deployment thresholds; the system is intended for research use only.

**Keywords:** explainable artificial intelligence; Grad-CAM; deployable AI; Streamlit; retinal fundus; deep learning; clinical decision support

---

## 1. Introduction

Retinal fundus photography is a low-cost, non-invasive method for screening vision-threatening diseases, including diabetic retinopathy, cataract, and glaucoma. Convolutional neural networks (CNNs) and transfer-learning architectures such as ResNet50 and EfficientNetB0 automate feature extraction from fundus images and achieve competitive classification performance on public datasets.

Despite strong benchmark results, two gaps limit translational value. First, CNNs function as opaque predictors: they output a class label without showing which image regions drove the decision. Clinicians cannot verify whether the model attended to pathology or to confounding artifacts such as acquisition borders or illumination gradients. Second, trained models typically require command-line execution, which excludes non-technical users from interactive evaluation.

Explainable artificial intelligence (XAI) addresses the transparency gap. Gradient-weighted Class Activation Mapping (Grad-CAM) produces post-hoc spatial heatmaps that highlight regions influencing a target class score. Deployable AI addresses the accessibility gap by wrapping inference in a graphical interface with upload, visualization, and export capabilities.

This manuscript describes Parts 4 and 5 of our eye-disease classification research. Part 4 implements Grad-CAM explainability. Part 5 deploys the full inference-explanation pipeline as a Streamlit application with drag-and-drop upload and PDF report download. We report quantitative test-set performance and discuss implications for responsible medical AI research.

---

## 2. Methods

### 2.1 System Overview

We built on a four-class fundus dataset and CNN classifiers trained in Parts 1–3 of the project. Parts 4 and 5 add two modules:

| Part | Module | Role |
|------|--------|------|
| 4 | `src/gradcam.py` | Post-hoc spatial explanation (Grad-CAM) |
| 5 | `src/app.py` | Web deployment, upload, visualization, export |
| 5 | `src/report_pdf.py` | PDF session report generation |

The end-to-end workflow is: **upload fundus image → preprocess → classify → explain (Grad-CAM) → export PDF report**.

![Executive Summary](figures/fig0_executive_summary.png)

**Figure 1.** Executive summary of test-set performance (*n* = 637).

### 2.2 Part 4 — Explainable AI with Grad-CAM

#### 2.2.1 Theoretical Basis

Grad-CAM computes a class-discriminative localization map from the final convolutional layer of a CNN. Let \(A^k\) denote the *k*-th feature map and \(y^c\) the score for class *c*. Importance weights are obtained by global average pooling of gradients:

\[
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}
\]

The heatmap is:

\[
L^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
\]

ReLU retains only features with positive influence on class *c*. We upsample \(L^c\) to the original image size and alpha-blend it (α = 0.4) using an inferno colormap.

#### 2.2.2 Implementation (`src/gradcam.py`)

We implemented a `GradCAM` class with the following design:

| Method | Purpose |
|--------|---------|
| `_find_last_conv_layer()` | Auto-detect the last 4D convolutional layer (Keras 2/3 compatible) |
| `compute_heatmap()` | Compute gradients with `tf.GradientTape`, pool, weight feature maps, apply ReLU |
| `overlay_heatmap()` | Resize heatmap and blend with the original RGB image |
| `generate_visualization()` | End-to-end pipeline: load → preprocess → predict → heatmap → three-panel figure |

We used the same `preprocess_image()` function as training (medical crop, 224×224 resize, normalization) to align inference and explanation distributions.

![Grad-CAM Pipeline](figures/fig1_gradcam_pipeline.png)

**Figure 2.** Grad-CAM explainability pipeline (Part 4).

![Grad-CAM Panels](figures/fig6_gradcam_output_panels.png)

**Figure 3.** Standard three-panel Grad-CAM output: original fundus, heatmap, and clinical overlay.

#### 2.2.3 Expected Clinical Attention Patterns

We expected clinically meaningful activations as follows:

- **Normal:** optic disc and background vasculature without focal pathology.
- **Diabetic retinopathy:** microaneurysms, hemorrhages, and exudates along vascular arcades.
- **Cataract:** lens-related opacity when visible in the capture.
- **Glaucoma:** optic disc, neuroretinal rim, and cup-to-disc region.

Activations on peripheral black borders indicate preprocessing or acquisition artifacts rather than pathology.

### 2.3 Part 5 — Deployable AI with Streamlit

#### 2.3.1 Application Architecture (`src/app.py`)

We developed a Streamlit web application that exposes the trained model to non-programmers. The interface uses a wide layout with a persistent sidebar and a main analysis panel.

![Deployment Architecture](figures/fig2_deployment_architecture.png)

**Figure 4.** Layered deployment architecture (Part 5).

**Sidebar controls (Section 8):**

- Model selection from `models/**/*.h5`
- Confidence threshold slider (default 0.5)
- Grad-CAM visualization toggle
- Project metadata and dataset reference

**Main panel (Section 9):**

- Drag-and-drop file uploader (`st.file_uploader`) for JPG, JPEG, and PNG fundus images
- Side-by-side display of uploaded image and AI prediction
- Color-coded prediction box with class probabilities (`st.progress` bars)
- Low-confidence warning when prediction falls below threshold
- Grad-CAM side-by-side comparison (original vs. overlay)
- Disease-specific educational text
- Export buttons for text and PDF reports

#### 2.3.2 Drag-and-Drop Upload Workflow

Users upload a fundus image through the browser. The application writes the file to a temporary directory, passes the path to `EyeDiseasePredictor` (`src/predict.py`), and displays results without persisting patient data beyond the session.

#### 2.3.3 PDF Report Download

We implemented `generate_prediction_pdf()` in `src/report_pdf.py`. Each session report includes:

- Timestamp and model identifier
- Predicted class and confidence
- Full probability distribution table
- Grad-CAM availability status
- Medical disclaimer (research use only)

Users download the report via `st.download_button` as `Technical_Report.pdf`.

#### 2.3.4 Launch

```bash
python main.py app
# equivalent: streamlit run src/app.py
```

The application runs locally at `http://localhost:8501`.

---

## 3. Results

### 3.1 Overall Classification Performance

We evaluated the best saved model on 637 held-out test images. Table 1 summarizes overall metrics.

**Table 1.** Overall test-set performance.

| Metric | Value |
|--------|------:|
| Accuracy | 60.75% |
| Precision (macro) | 68.82% |
| Recall (macro) | 60.44% |
| F1-score (macro) | 60.77% |
| ROC-AUC (macro) | 86.04% |

![Overall Metrics](figures/fig5_overall_metrics.png)

**Figure 5.** Overall metrics compared with the 85% project target (dashed line).

ROC-AUC (86.0%) exceeded accuracy (60.8%), indicating separable class probabilities despite moderate hard-label performance.

### 3.2 Per-Class Performance

**Table 2.** Per-class precision, recall, F1-score, and support.

| Class | Precision | Recall | F1 | *n* |
|-------|----------:|-------:|---:|----:|
| Normal | 44.3% | 84.0% | 58.0% | 162 |
| Diabetic retinopathy | 92.2% | 57.2% | 70.6% | 166 |
| Cataract | 67.1% | 62.4% | 64.7% | 157 |
| Glaucoma | 71.6% | 38.2% | 49.8% | 152 |

![Per-Class Metrics](figures/fig3_per_class_metrics.png)

**Figure 6.** Per-class precision, recall, and F1-score.

Diabetic retinopathy achieved the highest precision (92.2%). Glaucoma showed the lowest recall (38.2%), indicating a high missed-case rate that motivates Grad-CAM review.

### 3.3 Confusion Matrix

![Confusion Matrix](figures/fig4_confusion_matrix.png)

**Figure 7.** Confusion matrix: raw counts (left) and row-normalized percentages (right).

The largest off-diagonal block was Normal cases predicted as diabetic retinopathy (*n* = 60), supporting combined use of confidence scores and Grad-CAM for clinician verification.

### 3.4 Explainability and Deployment Outcomes

Grad-CAM produced interpretable three-panel figures for compatible model architectures. The Streamlit application successfully integrated upload, prediction, Grad-CAM display, and PDF export in a single session. When Grad-CAM generation failed (e.g., unsupported layer topology), the application retained the prediction and displayed a graceful fallback message.

---

## 4. Discussion

We integrated Grad-CAM explainability with a Streamlit deployment to create a human-in-the-loop screening workflow. Users receive a quantitative prediction and immediately inspect spatial evidence—a design aligned with transparency goals in medical AI research.

**Strengths.** Grad-CAM requires no model retraining and produces intuitive heatmaps. Streamlit enables rapid deployment with drag-and-drop upload and PDF documentation suitable for thesis defense, laboratory demos, and peer review.

**Limitations.** Test accuracy (60.8%) falls below the 85% project target and is insufficient for clinical deployment. Glaucoma recall (38.2%) is particularly concerning for a vision-threatening condition. Grad-CAM explains *where* the model looked but not *why* in causal terms. The application runs locally without authentication, encryption, or HIPAA-compliant infrastructure.

**Comparison with literature.** Selvaraju et al. (2017) established Grad-CAM as a standard XAI tool for CNNs; our implementation follows this framework in TensorFlow with medical preprocessing consistency. Deployable research interfaces increasingly use Streamlit for reproducible ML demos; we extend this pattern with integrated PDF audit trails.

**Future directions.** We plan to evaluate Grad-CAM++ and Score-CAM, apply class-weighted loss to improve glaucoma recall, add batch screening with CSV export, and containerize deployment with Docker.

---

## 5. Conclusion

Parts 4 and 5 complete the translational pathway from trained retinal classifiers to explainable, deployable decision support. Grad-CAM in `src/gradcam.py` reveals spatial attribution aligned with fundus anatomy. The Streamlit application in `src/app.py` delivers drag-and-drop inference, Grad-CAM visualization, and PDF report download for research and education. Together, these components demonstrate a principled approach to explainable and deployable artificial intelligence—provided users treat outputs as research aids, not clinical diagnoses.

> **Medical disclaimer.** This system is for educational and research purposes only. It does not replace professional medical advice, diagnosis, or treatment.

---

## References

1. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2017:618-626.

2. Doddi G. Eye diseases classification [dataset]. Kaggle. 2020. Available from: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

3. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2016:770-778.

4. Tan M, Le Q. EfficientNet: rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning (ICML)*. 2019:6105-6114.

5. Abadi M, et al. TensorFlow: a system for large-scale machine learning. *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*. 2016:265-283.

6. Streamlit Inc. Streamlit documentation. Available from: https://docs.streamlit.io

7. Sainani K. Writing in the sciences. Stanford Online / Coursera. Available from: https://www.coursera.org/learn/sciwrite

---

## Appendix — Reproducibility

**Generate figures and PDF:**

```bash
pip install -r requirements.txt
python scripts/generate_report_figures.py
python scripts/export_scientific_paper_pdf.py
```

**Launch web application:**

```bash
python main.py app
```

**Key files for review:**

| File | Content |
|------|---------|
| `src/gradcam.py` | Grad-CAM implementation (Part 4) |
| `src/app.py` | Streamlit deployment (Part 5, Sections 8–9) |
| `src/report_pdf.py` | PDF report generator |
| `reports/Scientific_Paper.pdf` | This manuscript (PDF) |
| `results/best_model_evaluation.json` | Test-set metrics |

---

*Manuscript prepared following IMRaD structure and scientific writing principles (Writing in the Sciences, Stanford Online).*
