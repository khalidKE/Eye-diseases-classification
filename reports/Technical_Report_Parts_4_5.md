# Explainable and Deployable Artificial Intelligence for Retinal Disease Classification

**Original Research — Parts 4 & 5**  
**Modules:** `src/gradcam.py` · `src/app.py` · `src/report_pdf.py`  
**Date:** June 2026

---

## Abstract

Deep learning classifiers for retinal fundus images achieve useful discrimination but remain difficult to trust without transparent reasoning and accessible deployment. We present **Part 4 (Explainable AI)** using Grad-CAM and **Part 5 (Deployable AI)** using a Streamlit web interface. Grad-CAM highlights spatial evidence from the final convolutional layer via TensorFlow `GradientTape`. The application supports drag-and-drop upload, real-time inference, interactive heatmaps, and PDF export. On a held-out test set (*n* = 637), the best model achieved **60.8% accuracy** and **86.0% macro ROC-AUC**. Grad-CAM enables human-in-the-loop verification, especially for low-recall classes such as glaucoma (38.2% recall).

**Keywords:** Explainable AI, Grad-CAM, Streamlit, Retinal Fundus, Clinical Decision Support, Deployable AI

![Executive Summary](figures/fig0_executive_summary.png)

**Figure 0.** Executive summary — model KPIs and per-class F1 overview.

---

## 1. Introduction

Automated analysis of retinal fundus photographs supports early detection of diabetic retinopathy, cataract, and glaucoma. Convolutional neural networks and transfer-learning models (ResNet50, EfficientNetB0) extract strong visual features, yet two barriers limit adoption:

1. **Opacity** — clinicians cannot see why a prediction was made.
2. **Inaccessibility** — trained models require command-line tools.

Parts 4 and 5 close this gap. Part 4 implements Grad-CAM explainability. Part 5 deploys inference and explanation through Streamlit with PDF documentation.

---

## 2. Methods

### 2.1 Dataset and Model Context

We used the [Kaggle Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) dataset (Normal, Diabetic Retinopathy, Cataract, Glaucoma). Images were cropped, resized to 224×224, and normalized. The best model was evaluated on **637 test images**.

### 2.2 Grad-CAM Explainability (Part 4)

Grad-CAM computes importance weights α<sub>k</sub> by global average pooling of gradients ∂y<sup>c</sup>/∂A<sup>k</sup>. The map L = ReLU(Σ α<sub>k</sub> A<sup>k</sup>) is overlaid on the fundus image (α = 0.4).

![Grad-CAM Pipeline](figures/fig1_gradcam_pipeline.png)

**Figure 1.** Grad-CAM pipeline implemented in `src/gradcam.py`.

| Component | Function |
|-----------|----------|
| `_find_last_conv_layer()` | Auto-detect last 4D conv layer |
| `compute_heatmap()` | GradientTape → weighted maps → ReLU |
| `overlay_heatmap()` | Colormap blend on original image |
| `generate_visualization()` | End-to-end 3-panel export |

### 2.3 Streamlit Deployment (Part 5)

The Streamlit app (`src/app.py`) provides model selection, drag-and-drop upload, confidence thresholding, Grad-CAM toggle, and PDF/text report download.

![Deployment Architecture](figures/fig2_deployment_architecture.png)

**Figure 2.** Layered deployment architecture.

**Inference workflow:** Upload → preprocess → predict → (optional) Grad-CAM → export PDF.

```bash
python main.py app
# or: streamlit run src/app.py
```

---

## 3. Results

### 3.1 Quantitative Performance

| Metric | Value |
|--------|-------|
| Accuracy | 60.75% |
| Precision (macro) | 68.82% |
| Recall (macro) | 60.44% |
| F1-Score (macro) | 60.77% |
| ROC-AUC (macro) | 86.04% |

![Overall Metrics](figures/fig5_overall_metrics.png)

**Figure 3.** Overall metrics vs. 85% project target.

![Per-Class Metrics](figures/fig3_per_class_metrics.png)

**Figure 4.** Per-class precision, recall, and F1.

![Confusion Matrix](figures/fig4_confusion_matrix.png)

**Figure 5.** Confusion matrix — counts and row-normalized percentages.

### 3.2 Explainability Outputs

![Grad-CAM Panels](figures/fig6_gradcam_output_panels.png)

**Figure 6.** Three-panel Grad-CAM visualization.

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Normal | 44.3% | 84.0% | 58.0% | 162 |
| Diabetic Retinopathy | 92.2% | 57.2% | 70.6% | 166 |
| Cataract | 67.1% | 62.4% | 64.7% | 157 |
| Glaucoma | 71.6% | 38.2% | 49.8% | 152 |

---

## 4. Discussion

Integrating Grad-CAM with Streamlit creates a **human-in-the-loop** workflow. Users receive a prediction and immediately inspect spatial evidence — essential when accuracy (60.8%) remains below the 85% clinical target.

**Limitations:** research-only accuracy; Grad-CAM may fail on some Keras 3 topologies; fundus-only input; local deployment without HIPAA infrastructure.

**Future work:** Grad-CAM++, class-weighted loss for glaucoma, batch screening, Docker deployment.

---

## 5. Conclusion

Parts 4 and 5 complete the pathway from trained classifier to explainable, deployable decision support. Grad-CAM reveals spatial attribution; Streamlit delivers accessible inference and PDF documentation for retinal disease screening research.

> **Medical Disclaimer:** For educational and research purposes only. Not a substitute for professional medical diagnosis.

---

## References

1. Selvaraju, R. R., et al. (2017). Grad-CAM. *ICCV*.
2. Doddi, G. Eye Diseases Classification. *Kaggle*.
3. He, K., et al. (2016). ResNet. *CVPR*.
4. Tan, M., & Le, Q. (2019). EfficientNet. *ICML*.
5. Sainani, K. *Writing in the Sciences*. Stanford Online.

---

*Generate PDF:* `python scripts/export_technical_report_pdf.py`
