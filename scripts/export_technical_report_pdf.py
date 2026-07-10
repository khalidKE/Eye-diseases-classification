"""
Export a premium IMRaD Technical Report (Parts 4 & 5) to PDF.
"""

import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "results" / "best_model_evaluation.json"
OUT_PATH = ROOT / "reports" / "Technical_Report.pdf"

# Brand colors
INK = colors.HexColor("#0B1220")
SUB = colors.HexColor("#475569")
MUTED = colors.HexColor("#94A3B8")
ACCENT = colors.HexColor("#0284C7")
ACCENT2 = colors.HexColor("#0891B2")
PANEL = colors.HexColor("#F8FAFC")
LINE = colors.HexColor("#E2E8F0")
WHITE = colors.white


class NumberedCanvas:
    """Injected via doc.build callback pattern — handled in onPage."""


def styles():
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", fontName="Helvetica-Bold", fontSize=26, leading=32,
            textColor=WHITE, alignment=TA_LEFT, spaceAfter=8,
        ),
        "cover_sub": ParagraphStyle(
            "CoverSub", fontName="Helvetica", fontSize=12, leading=16,
            textColor=colors.HexColor("#CBD5E1"), alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "H1", fontName="Helvetica-Bold", fontSize=18, leading=22,
            textColor=INK, spaceBefore=18, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2", fontName="Helvetica-Bold", fontSize=13, leading=17,
            textColor=ACCENT, spaceBefore=14, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3", fontName="Helvetica-Bold", fontSize=11, leading=14,
            textColor=INK, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", fontName="Helvetica", fontSize=10.5, leading=15,
            textColor=SUB, alignment=TA_JUSTIFY, spaceAfter=8,
        ),
        "abstract": ParagraphStyle(
            "Abstract", fontName="Helvetica", fontSize=10.5, leading=15,
            textColor=SUB, alignment=TA_JUSTIFY, leftIndent=6, rightIndent=6,
        ),
        "caption": ParagraphStyle(
            "Caption", fontName="Helvetica-Oblique", fontSize=9, leading=12,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=14,
        ),
        "keyword": ParagraphStyle(
            "Keyword", fontName="Helvetica", fontSize=9, leading=12, textColor=SUB,
        ),
        "footer": ParagraphStyle(
            "Footer", fontName="Helvetica", fontSize=8, textColor=MUTED,
        ),
    }


def hr(color=LINE, thickness=1):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceBefore=6, spaceAfter=10)


def styled_table(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), SUB),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        style_cmds.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PANEL]))
    t.setStyle(TableStyle(style_cmds))
    return t


def add_figure(story, st, filename, caption, width=16 * cm):
    path = FIG_DIR / filename
    if not path.exists():
        return
    img = Image(str(path))
    aspect = img.imageHeight / float(img.imageWidth) if img.imageWidth else 0.55
    img.drawWidth = width
    img.drawHeight = width * aspect
    story.append(Spacer(1, 0.15 * cm))
    story.append(img)
    story.append(Paragraph(caption, st["caption"]))


def cover_page(story, st):
    # Full-width cover block simulated with table
    cover_data = [[Paragraph(
        "Explainable and Deployable Artificial Intelligence<br/>"
        "for Retinal Disease Classification",
        st["cover_title"],
    )]]
    cover = Table(cover_data, colWidths=[17 * cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 24),
        ("RIGHTPADDING", (0, 0), (-1, -1), 24),
        ("TOPPADDING", (0, 0), (-1, -1), 28),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 28),
    ]))
    story.append(cover)
    story.append(Spacer(1, 0.5 * cm))

    meta = styled_table([
        ["Document Type", "Original Research — Parts 4 & 5"],
        ["Focus", "Explainable AI (Grad-CAM) · Deployable AI (Streamlit)"],
        ["Modules", "src/gradcam.py · src/app.py · src/report_pdf.py"],
        ["Framework", "TensorFlow/Keras · Streamlit · ReportLab"],
        ["Date", datetime.now().strftime("%B %d, %Y")],
    ], [5 * cm, 12 * cm], header=False)
    story.append(meta)
    story.append(Spacer(1, 0.4 * cm))

    if (FIG_DIR / "fig0_executive_summary.png").exists():
        add_figure(story, st, "fig0_executive_summary.png",
                   "Figure 0. Executive summary of model performance and per-class F1 scores.")


def abstract_section(story, st):
    story.append(Paragraph("Abstract", st["h1"]))
    abstract_text = (
        "Deep learning classifiers for retinal fundus images achieve useful discrimination "
        "but remain difficult to trust in clinical settings without transparent reasoning and "
        "accessible deployment. We present Parts 4 and 5 of an eye-disease classification "
        "system covering <b>Explainable AI</b> via Gradient-weighted Class Activation Mapping "
        "(Grad-CAM) and <b>Deployable AI</b> via a Streamlit web interface. Grad-CAM highlights "
        "spatial evidence from the final convolutional layer using TensorFlow GradientTape. "
        "The deployed application supports drag-and-drop upload, real-time inference, interactive "
        "heatmaps, and downloadable PDF reports. On a held-out test set (<i>n</i> = 637), the best "
        "model achieved 60.8% accuracy and 86.0% macro ROC-AUC. Grad-CAM enables human-in-the-loop "
        "verification, particularly for low-recall classes such as glaucoma (38.2% recall). "
        "This work demonstrates a reproducible pathway from model output to explainable, "
        "user-facing decision support for research and education."
    )
    abs_box = Table([[Paragraph(abstract_text, st["abstract"])]], colWidths=[17 * cm])
    abs_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 1.5, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(abs_box)
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph(
        "<b>Keywords:</b> Explainable AI · Grad-CAM · Streamlit · Retinal Fundus · "
        "Clinical Decision Support · Deep Learning · Deployable AI",
        st["keyword"],
    ))
    story.append(hr())


def introduction(story, st):
    story.append(Paragraph("1. Introduction", st["h1"]))
    story.append(Paragraph(
        "Automated analysis of retinal fundus photographs supports early detection of "
        "diabetic retinopathy, cataract, and glaucoma. Convolutional neural networks and "
        "transfer-learning architectures such as ResNet50 and EfficientNetB0 provide strong "
        "feature extraction, yet two barriers limit translational use: models do not explain "
        "their decisions, and trained weights are not accessible to non-programmers.",
        st["body"],
    ))
    story.append(Paragraph(
        "Parts 4 and 5 address these gaps. Part 4 implements post-hoc explainability with "
        "Grad-CAM. Part 5 deploys inference and explanation through a Streamlit application "
        "with PDF export. We evaluate explainability in the context of quantitative test-set "
        "performance and discuss limitations relevant to responsible medical AI research.",
        st["body"],
    ))


def methods(story, st):
    story.append(PageBreak())
    story.append(Paragraph("2. Methods", st["h1"]))

    story.append(Paragraph("2.1 Dataset and Model Context", st["h2"]))
    story.append(Paragraph(
        "We used the Kaggle Eye Diseases Classification dataset with four classes: Normal, "
        "Diabetic Retinopathy, Cataract, and Glaucoma. Images were preprocessed (medical crop, "
        "224×224 resize, normalization) as described in Parts 1–3. The best saved model was "
        "evaluated on 637 test images with stratified class representation.",
        st["body"],
    ))

    story.append(Paragraph("2.2 Grad-CAM Explainability (Part 4)", st["h2"]))
    story.append(Paragraph(
        "We implemented Grad-CAM in <font face='Courier'>src/gradcam.py</font>. Given final "
        "convolutional feature maps <i>A<sup>k</sup></i> and class score <i>y<sup>c</sup></i>, "
        "importance weights are computed by global average pooling of gradients "
        "<i>∂y<sup>c</sup>/∂A<sup>k</sup></i>. The localization map "
        "L = ReLU(Σ α<sub>k</sub> A<sup>k</sup>) is upsampled and alpha-blended (α = 0.4) "
        "onto the original image. The last convolutional layer is auto-detected for Keras 2/3 "
        "compatibility.",
        st["body"],
    ))
    add_figure(story, st, "fig1_gradcam_pipeline.png",
               "Figure 1. Grad-CAM explainability pipeline from fundus input to spatial heatmap.")

    story.append(Paragraph("2.3 Streamlit Deployment (Part 5)", st["h2"]))
    story.append(Paragraph(
        "We built a Streamlit interface in <font face='Courier'>src/app.py</font>. The application "
        "loads cached Keras models, accepts drag-and-drop uploads, runs "
        "<font face='Courier'>EyeDiseasePredictor</font>, optionally generates Grad-CAM overlays, "
        "and exports session reports as PDF via <font face='Courier'>src/report_pdf.py</font>. "
        "A configurable confidence threshold triggers low-confidence warnings.",
        st["body"],
    ))
    add_figure(story, st, "fig2_deployment_architecture.png",
               "Figure 2. Layered deployment architecture for the Streamlit clinical interface.")

    impl_data = [
        ["Component", "Module", "Function"],
        ["Grad-CAM engine", "gradcam.py", "Heatmap computation and overlay"],
        ["Predictor", "predict.py", "Preprocessing and softmax inference"],
        ["Web interface", "app.py", "Upload, visualization, export"],
        ["PDF generator", "report_pdf.py", "Session report documentation"],
    ]
    story.append(styled_table(impl_data, [4 * cm, 4 * cm, 9 * cm]))


def results(story, st, metrics):
    story.append(PageBreak())
    story.append(Paragraph("3. Results", st["h1"]))

    overall = metrics["overall"]
    story.append(Paragraph("3.1 Quantitative Performance", st["h2"]))
    story.append(Paragraph(
        f"Table 1 summarizes test-set performance. Accuracy was {overall['accuracy']*100:.1f}%, "
        f"macro ROC-AUC was {overall['roc_auc_macro']*100:.1f}%, and macro F1 was "
        f"{overall['f1_macro']*100:.1f}%. ROC-AUC exceeded accuracy, indicating separable "
        "class probabilities despite moderate hard-label accuracy.",
        st["body"],
    ))

    perf_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{overall['accuracy']*100:.2f}%"],
        ["Precision (macro)", f"{overall['precision_macro']*100:.2f}%"],
        ["Recall (macro)", f"{overall['recall_macro']*100:.2f}%"],
        ["F1-Score (macro)", f"{overall['f1_macro']*100:.2f}%"],
        ["ROC-AUC (macro)", f"{overall['roc_auc_macro']*100:.2f}%"],
        ["Test samples", "637"],
    ]
    story.append(styled_table(perf_data, [8 * cm, 9 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    add_figure(story, st, "fig5_overall_metrics.png",
               "Figure 3. Overall metrics compared with the 85% project target.")
    add_figure(story, st, "fig3_per_class_metrics.png",
               "Figure 4. Per-class precision, recall, and F1 on the test set.")
    add_figure(story, st, "fig4_confusion_matrix.png",
               "Figure 5. Confusion matrix — counts and row-normalized percentages.")

    story.append(Paragraph("3.2 Explainability Outputs", st["h2"]))
    story.append(Paragraph(
        "Grad-CAM produced three-panel visualizations: original fundus, raw heatmap, and clinical "
        "overlay. Attention concentrated on the optic disc and peripapillary region — anatomically "
        "relevant for glaucoma and vascular pathology. When heatmaps highlight acquisition borders, "
        "this indicates preprocessing or dataset artifacts rather than pathology.",
        st["body"],
    ))
    add_figure(story, st, "fig6_gradcam_output_panels.png",
               "Figure 6. Representative Grad-CAM output panels (demonstration layout).")

    pc = metrics["per_class"]
    class_data = [
        ["Class", "Precision", "Recall", "F1", "Support"],
        ["Normal", f"{pc['normal']['precision']*100:.1f}%", f"{pc['normal']['recall']*100:.1f}%",
         f"{pc['normal']['f1_score']*100:.1f}%", str(pc['normal']['support'])],
        ["Diabetic Retinopathy", f"{pc['diabetic_retinopathy']['precision']*100:.1f}%",
         f"{pc['diabetic_retinopathy']['recall']*100:.1f}%",
         f"{pc['diabetic_retinopathy']['f1_score']*100:.1f}%", str(pc['diabetic_retinopathy']['support'])],
        ["Cataract", f"{pc['cataract']['precision']*100:.1f}%", f"{pc['cataract']['recall']*100:.1f}%",
         f"{pc['cataract']['f1_score']*100:.1f}%", str(pc['cataract']['support'])],
        ["Glaucoma", f"{pc['glaucoma']['precision']*100:.1f}%", f"{pc['glaucoma']['recall']*100:.1f}%",
         f"{pc['glaucoma']['f1_score']*100:.1f}%", str(pc['glaucoma']['support'])],
    ]
    story.append(styled_table(class_data, [4.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3.5 * cm]))


def discussion(story, st):
    story.append(PageBreak())
    story.append(Paragraph("4. Discussion", st["h1"]))
    story.append(Paragraph(
        "Integrating Grad-CAM with Streamlit creates a human-in-the-loop workflow: users receive "
        "a quantitative prediction and immediately inspect spatial evidence. This closed loop "
        "supports trust calibration — essential when accuracy remains below the 85% clinical "
        "target. Diabetic retinopathy achieved 92.2% precision but only 57.2% recall; glaucoma "
        "recall was 38.2%. Grad-CAM review is therefore most valuable for suspected false "
        "negatives in glaucoma screening.",
        st["body"],
    ))
    story.append(Paragraph("4.1 Limitations", st["h2"]))
    limits = [
        "Test accuracy (60.8%) limits clinical applicability; deployment is research-only.",
        "Grad-CAM layer auto-detection may fail on some Keras 3 topologies.",
        "Single-modality fundus input — no OCT or patient metadata.",
        "Local deployment without authentication or HIPAA-compliant infrastructure.",
    ]
    for item in limits:
        story.append(Paragraph(f"• {item}", st["body"]))

    story.append(Paragraph("4.2 Future Work", st["h2"]))
    story.append(Paragraph(
        "We plan to evaluate Grad-CAM++ and Score-CAM, improve glaucoma recall via class-weighted "
        "loss, add batch screening with CSV export, and containerize deployment with Docker or "
        "Streamlit Cloud for reproducible demonstrations.",
        st["body"],
    ))


def conclusion(story, st):
    story.append(Paragraph("5. Conclusion", st["h1"]))
    story.append(Paragraph(
        "Parts 4 and 5 complete the translational pathway from trained classifier to explainable, "
        "deployable decision support. Grad-CAM reveals spatial attribution aligned with retinal "
        "anatomy; the Streamlit application delivers accessible inference, visualization, and PDF "
        "documentation. Together, these components demonstrate responsible, transparent AI design "
        "for retinal disease screening research.",
        st["body"],
    ))

    disclaimer = Table([[Paragraph(
        "<b>Medical Disclaimer.</b> This system is for educational and research purposes only. "
        "It does not replace professional medical diagnosis or treatment.",
        ParagraphStyle("Disc", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#92400E"), leading=13),
    )]], colWidths=[17 * cm])
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#F59E0B")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(disclaimer)


def references(story, st):
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("References", st["h1"]))
    refs = [
        "1. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.",
        "2. Doddi, G. Eye Diseases Classification Dataset. Kaggle.",
        "3. Abadi, M., et al. (2016). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. OSDI.",
        "4. Streamlit Inc. Streamlit Documentation. streamlit.io.",
        "5. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.",
        "6. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.",
        "7. Sainani, K. Writing in the Sciences. Stanford Online / Coursera.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle("Ref", fontName="Helvetica", fontSize=9, textColor=SUB, leading=13, spaceAfter=4)))


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Header line
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, h - 1.4 * cm, w - 2 * cm, h - 1.4 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, h - 1.15 * cm, "Explainable & Deployable AI — Retinal Disease Classification")
    canvas.drawRightString(w - 2 * cm, h - 1.15 * cm, "Parts 4 & 5")
    # Footer
    canvas.line(2 * cm, 1.4 * cm, w - 2 * cm, 1.4 * cm)
    canvas.drawString(2 * cm, 0.9 * cm, "Technical Report · Research Use Only")
    canvas.drawRightString(w - 2 * cm, 0.9 * cm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics not found: {METRICS_PATH}")

    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    st = styles()
    story = []

    cover_page(story, st)
    story.append(PageBreak())
    abstract_section(story, st)
    introduction(story, st)
    methods(story, st)
    results(story, st, metrics)
    discussion(story, st)
    conclusion(story, st)
    references(story, st)

    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Technical Report — Explainable & Deployable AI",
        author="Eye Disease Classification Research Team",
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF exported: {OUT_PATH}")


if __name__ == "__main__":
    main()
