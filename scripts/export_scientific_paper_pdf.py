"""
Export Scientific Paper (Parts 4 & 5) to PDF — IMRaD journal manuscript style.
Output: reports/Scientific_Paper.pdf
"""

import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
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
OUT_PATH = ROOT / "reports" / "Scientific_Paper.pdf"

INK = colors.HexColor("#0B1220")
SUB = colors.HexColor("#475569")
MUTED = colors.HexColor("#94A3B8")
ACCENT = colors.HexColor("#0369A1")
ACCENT_LIGHT = colors.HexColor("#0EA5E9")
PANEL = colors.HexColor("#F8FAFC")
LINE = colors.HexColor("#E2E8F0")
WHITE = colors.white
GOLD = colors.HexColor("#B45309")


def S():
    return {
        "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=22, leading=28,
                                 textColor=WHITE, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("Sub", fontName="Helvetica", fontSize=11, leading=15,
                                    textColor=colors.HexColor("#BAE6FD"), alignment=TA_LEFT),
        "journal": ParagraphStyle("J", fontName="Helvetica-Oblique", fontSize=9,
                                   textColor=MUTED, alignment=TA_CENTER, spaceAfter=6),
        "h1": ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=16, leading=20,
                              textColor=INK, spaceBefore=20, spaceAfter=8),
        "h2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=12, leading=16,
                              textColor=ACCENT, spaceBefore=14, spaceAfter=5),
        "h3": ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                              textColor=INK, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14.5,
                                textColor=SUB, alignment=TA_JUSTIFY, spaceAfter=7),
        "abstract": ParagraphStyle("Abs", fontName="Helvetica", fontSize=10, leading=14.5,
                                    textColor=SUB, alignment=TA_JUSTIFY),
        "toc": ParagraphStyle("TOC", fontName="Helvetica", fontSize=10, leading=16,
                               textColor=SUB, leftIndent=12),
        "caption": ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
                                   textColor=MUTED, alignment=TA_CENTER, spaceAfter=12),
        "keyword": ParagraphStyle("KW", fontName="Helvetica", fontSize=9, textColor=SUB, spaceAfter=4),
        "ref": ParagraphStyle("Ref", fontName="Helvetica", fontSize=9, leading=13,
                               textColor=SUB, spaceAfter=5, leftIndent=14, firstLineIndent=-14),
    }


def table(data, widths, header=True):
    t = Table(data, colWidths=widths)
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), SUB),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PANEL]),
        ]
    t.setStyle(TableStyle(cmds))
    return t


def fig(story, styles, fname, cap, w=15.5 * cm):
    p = FIG_DIR / fname
    if not p.exists():
        return
    img = Image(str(p))
    ar = img.imageHeight / float(img.imageWidth) if img.imageWidth else 0.5
    img.drawWidth = w
    img.drawHeight = w * ar
    story.append(Spacer(1, 0.12 * cm))
    story.append(img)
    story.append(Paragraph(cap, styles["caption"]))


def abstract_box(story, styles, text):
    box = Table([[Paragraph(text, styles["abstract"])]], colWidths=[16.5 * cm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 2, ACCENT_LIGHT),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(box)


def title_page(story, st):
    band = Table([[
        Paragraph(
            "Explainable and Deployable Artificial Intelligence<br/>"
            "for Retinal Disease Classification",
            st["title"],
        )
    ]], colWidths=[17 * cm])
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 22),
        ("RIGHTPADDING", (0, 0), (-1, -1), 22),
        ("TOPPADDING", (0, 0), (-1, -1), 26),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(band)

    sub = Table([[
        Paragraph(
            "Original Research Setup · Parts 4 and 5<br/>"
            "<font size='9'>Grad-CAM Explainability · Streamlit Deployment · PDF Export</font>",
            st["subtitle"],
        )
    ]], colWidths=[17 * cm])
    sub.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0F2744")),
        ("LEFTPADDING", (0, 0), (-1, -1), 22),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
    ]))
    story.append(sub)
    story.append(Spacer(1, 0.35 * cm))

    meta = table([
        ["Document", "Scientific Manuscript (IMRaD)"],
        ["Study Modules", "src/gradcam.py · src/app.py · src/report_pdf.py"],
        ["Part 4", "Explainable AI — Grad-CAM"],
        ["Part 5", "Deployable AI — Drag-and-Drop · PDF Download"],
        ["Date", datetime.now().strftime("%B %d, %Y")],
    ], [5 * cm, 12 * cm], header=False)
    story.append(meta)
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph(
        "Prepared following <i>Writing in the Sciences</i> (Stanford Online) — "
        "active voice, IMRaD structure, minimal clutter.",
        st["journal"],
    ))


def toc_page(story, st):
    story.append(PageBreak())
    story.append(Paragraph("Table of Contents", st["h1"]))
    items = [
        "Abstract",
        "1. Introduction",
        "2. Methods",
        "    2.1 System Overview",
        "    2.2 Part 4 — Explainable AI (Grad-CAM) · src/gradcam.py",
        "    2.3 Part 5 — Deployable AI (Streamlit) · src/app.py",
        "3. Results",
        "4. Discussion",
        "5. Conclusion",
        "References",
        "Appendix — Reproducibility",
    ]
    for item in items:
        story.append(Paragraph(item, st["toc"]))
    story.append(HRFlowable(width="100%", thickness=1, color=LINE, spaceBefore=12, spaceAfter=12))


def on_page(c, doc):
    w, h = A4
    c.saveState()
    c.setStrokeColor(LINE)
    c.setLineWidth(0.4)
    c.line(2 * cm, h - 1.35 * cm, w - 2 * cm, h - 1.35 * cm)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(2 * cm, h - 1.1 * cm, "Scientific Paper · Explainable & Deployable AI · Parts 4 & 5")
    c.drawRightString(w - 2 * cm, h - 1.1 * cm, "IMRaD Manuscript")
    c.line(2 * cm, 1.35 * cm, w - 2 * cm, 1.35 * cm)
    c.drawString(2 * cm, 0.85 * cm, "Research Use Only — Not for Clinical Diagnosis")
    c.drawRightString(w - 2 * cm, 0.85 * cm, f"Page {c.getPageNumber()}")
    c.restoreState()


def build_story(metrics):
    st = S()
    story = []
    o = metrics["overall"]
    pc = metrics["per_class"]

    title_page(story, st)
    toc_page(story, st)

    # ABSTRACT
    story.append(Paragraph("Abstract", st["h1"]))
    abs_text = (
        "<b>Background.</b> Deep learning classifiers for retinal fundus images achieve useful "
        "accuracy but lack transparent reasoning and accessible deployment.<br/><br/>"
        "<b>Objective.</b> We designed an explainable and deployable AI pipeline for four-class "
        "retinal disease classification as Parts 4 and 5 of a multi-stage research project.<br/><br/>"
        "<b>Methods.</b> We implemented Grad-CAM in <font face='Courier' size='8'>src/gradcam.py</font> "
        "and deployed inference through a Streamlit application in "
        "<font face='Courier' size='8'>src/app.py</font> with drag-and-drop upload, Grad-CAM "
        "overlays, and PDF export via <font face='Courier' size='8'>src/report_pdf.py</font>.<br/><br/>"
        f"<b>Results.</b> On <i>n</i> = 637 test images, accuracy was {o['accuracy']*100:.1f}%, "
        f"macro F1 was {o['f1_macro']*100:.1f}%, and macro ROC-AUC was {o['roc_auc_macro']*100:.1f}%. "
        f"Glaucoma recall was {pc['glaucoma']['recall']*100:.1f}%. Grad-CAM localized attention "
        "to optic disc and vascular regions.<br/><br/>"
        "<b>Conclusions.</b> Grad-CAM plus Streamlit enables human-in-the-loop research decision "
        "support. Accuracy remains below clinical thresholds; use is research-only."
    )
    abstract_box(story, st, abs_text)
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "<b>Keywords:</b> explainable AI; Grad-CAM; deployable AI; Streamlit; retinal fundus; "
        "deep learning; clinical decision support",
        st["keyword"],
    ))

    # INTRODUCTION
    story.append(Paragraph("1. Introduction", st["h1"]))
    for para in [
        "Retinal fundus photography supports screening for diabetic retinopathy, cataract, "
        "and glaucoma. Convolutional neural networks automate feature extraction, yet two gaps "
        "limit adoption: models do not explain their decisions, and trained weights require "
        "command-line tools inaccessible to most clinicians.",
        "Explainable AI addresses transparency. Grad-CAM produces post-hoc heatmaps showing "
        "which regions influenced a class score. Deployable AI addresses accessibility through "
        "graphical interfaces with upload, visualization, and export.",
        "This manuscript presents Parts 4 and 5 of our eye-disease classification project. "
        "Part 4 implements Grad-CAM in src/gradcam.py. Part 5 deploys inference and explanation "
        "in src/app.py with drag-and-drop upload and PDF report download.",
    ]:
        story.append(Paragraph(para, st["body"]))

    # METHODS
    story.append(PageBreak())
    story.append(Paragraph("2. Methods", st["h1"]))
    story.append(Paragraph("2.1 System Overview", st["h2"]))
    story.append(Paragraph(
        "Parts 4 and 5 extend CNN classifiers trained in Parts 1–3. The workflow is: "
        "upload fundus image → preprocess → classify → explain with Grad-CAM → export PDF.",
        st["body"],
    ))
    story.append(table([
        ["Part", "Module", "Function"],
        ["4", "src/gradcam.py", "Grad-CAM spatial explanation"],
        ["5", "src/app.py", "Streamlit web interface"],
        ["5", "src/report_pdf.py", "PDF session reports"],
    ], [1.5 * cm, 4.5 * cm, 11 * cm]))
    fig(story, st, "fig0_executive_summary.png",
        "Figure 1. Executive summary of test-set performance (n = 637).")

    story.append(Paragraph("2.2 Part 4 — Explainable AI with Grad-CAM", st["h2"]))
    story.append(Paragraph("2.2.1 Implementation (src/gradcam.py)", st["h3"]))
    story.append(Paragraph(
        "The GradCAM class auto-detects the last convolutional layer, computes gradients with "
        "tf.GradientTape, pools gradient weights, applies ReLU, and alpha-blends the heatmap "
        "(α = 0.4) onto the original fundus image. Preprocessing matches training: medical crop, "
        "224×224 resize, normalization.",
        st["body"],
    ))
    fig(story, st, "fig1_gradcam_pipeline.png",
        "Figure 2. Grad-CAM explainability pipeline (Part 4).")
    fig(story, st, "fig6_gradcam_output_panels.png",
        "Figure 3. Three-panel Grad-CAM output: original, heatmap, overlay.")

    story.append(Paragraph("2.3 Part 5 — Deployable AI with Streamlit", st["h2"]))
    story.append(Paragraph("2.3.1 Application Design (src/app.py)", st["h3"]))
    story.append(Paragraph(
        "The Streamlit application provides sidebar controls (model selection, confidence "
        "threshold, Grad-CAM toggle) and a main panel with drag-and-drop upload, prediction "
        "display, Grad-CAM comparison, and export buttons. Users download session reports as "
        "Technical_Report.pdf.",
        st["body"],
    ))
    fig(story, st, "fig2_deployment_architecture.png",
        "Figure 4. Deployable AI architecture — Streamlit interface (Part 5).")

    # RESULTS
    story.append(PageBreak())
    story.append(Paragraph("3. Results", st["h1"]))
    story.append(Paragraph("3.1 Overall Performance", st["h2"]))
    story.append(Paragraph(
        f"Table 1 summarizes test-set metrics. Accuracy was {o['accuracy']*100:.2f}%; "
        f"macro ROC-AUC was {o['roc_auc_macro']*100:.2f}%.",
        st["body"],
    ))
    story.append(table([
        ["Metric", "Value"],
        ["Accuracy", f"{o['accuracy']*100:.2f}%"],
        ["Precision (macro)", f"{o['precision_macro']*100:.2f}%"],
        ["Recall (macro)", f"{o['recall_macro']*100:.2f}%"],
        ["F1 (macro)", f"{o['f1_macro']*100:.2f}%"],
        ["ROC-AUC (macro)", f"{o['roc_auc_macro']*100:.2f}%"],
        ["Test samples", "637"],
    ], [8 * cm, 8.5 * cm]))
    story.append(Spacer(1, 0.2 * cm))
    fig(story, st, "fig5_overall_metrics.png",
        "Figure 5. Overall metrics vs. 85% project target.")

    story.append(Paragraph("3.2 Per-Class and Confusion Analysis", st["h2"]))
    fig(story, st, "fig3_per_class_metrics.png",
        "Figure 6. Per-class precision, recall, and F1-score.")
    fig(story, st, "fig4_confusion_matrix.png",
        "Figure 7. Confusion matrix — counts and row-normalized percentages.")
    story.append(table([
        ["Class", "Precision", "Recall", "F1", "n"],
        ["Normal", f"{pc['normal']['precision']*100:.1f}%", f"{pc['normal']['recall']*100:.1f}%",
         f"{pc['normal']['f1_score']*100:.1f}%", "162"],
        ["Diabetic retinopathy", f"{pc['diabetic_retinopathy']['precision']*100:.1f}%",
         f"{pc['diabetic_retinopathy']['recall']*100:.1f}%",
         f"{pc['diabetic_retinopathy']['f1_score']*100:.1f}%", "166"],
        ["Cataract", f"{pc['cataract']['precision']*100:.1f}%", f"{pc['cataract']['recall']*100:.1f}%",
         f"{pc['cataract']['f1_score']*100:.1f}%", "157"],
        ["Glaucoma", f"{pc['glaucoma']['precision']*100:.1f}%", f"{pc['glaucoma']['recall']*100:.1f}%",
         f"{pc['glaucoma']['f1_score']*100:.1f}%", "152"],
    ], [4.5 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 1.6 * cm]))

    # DISCUSSION
    story.append(PageBreak())
    story.append(Paragraph("4. Discussion", st["h1"]))
    for para in [
        "Integrating Grad-CAM with Streamlit creates a human-in-the-loop workflow: users "
        "receive a prediction and inspect spatial evidence immediately. This design supports "
        "trust calibration essential when accuracy (60.8%) remains below clinical targets.",
        "Diabetic retinopathy achieved 92.2% precision but only 57.2% recall. Glaucoma recall "
        "was 38.2%. Grad-CAM review is most valuable for suspected false negatives in glaucoma "
        "screening. The drag-and-drop interface and PDF export make the pipeline accessible "
        "for thesis defense and peer review without command-line expertise.",
        "Limitations include research-only accuracy, possible Grad-CAM failure on some Keras 3 "
        "topologies, fundus-only input, and local deployment without HIPAA infrastructure.",
    ]:
        story.append(Paragraph(para, st["body"]))

    # CONCLUSION
    story.append(Paragraph("5. Conclusion", st["h1"]))
    story.append(Paragraph(
        "Parts 4 and 5 deliver explainable and deployable AI for retinal disease classification. "
        "Grad-CAM in src/gradcam.py reveals spatial attribution; Streamlit in src/app.py provides "
        "drag-and-drop inference, Grad-CAM visualization, and PDF documentation. These components "
        "demonstrate responsible transparent AI design for research and education—not clinical diagnosis.",
        st["body"],
    ))
    disc = Table([[Paragraph(
        "<b>Medical Disclaimer.</b> For educational and research purposes only. "
        "Consult a qualified healthcare provider for medical decisions.",
        ParagraphStyle("D", fontName="Helvetica", fontSize=9, textColor=GOLD, leading=13),
    )]], colWidths=[16.5 * cm])
    disc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#F59E0B")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(disc)

    # REFERENCES
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("References", st["h1"]))
    refs = [
        "1. Selvaraju RR, et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. ICCV. 2017.",
        "2. Doddi G. Eye diseases classification [dataset]. Kaggle. 2020.",
        "3. He K, et al. Deep residual learning for image recognition. CVPR. 2016.",
        "4. Tan M, Le Q. EfficientNet: rethinking model scaling for CNNs. ICML. 2019.",
        "5. Abadi M, et al. TensorFlow: large-scale machine learning. OSDI. 2016.",
        "6. Streamlit Inc. Streamlit documentation. docs.streamlit.io.",
        "7. Sainani K. Writing in the Sciences. Stanford Online / Coursera.",
    ]
    for r in refs:
        story.append(Paragraph(r, st["ref"]))

    return story


def main():
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    doc = SimpleDocTemplate(
        str(OUT_PATH), pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=1.75 * cm, bottomMargin=1.75 * cm,
        title="Scientific Paper — Explainable & Deployable AI",
    )
    doc.build(build_story(metrics), onFirstPage=on_page, onLaterPages=on_page)
    print(f"Scientific paper exported: {OUT_PATH}")


if __name__ == "__main__":
    main()
