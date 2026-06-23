"""
Export Part 3: Proposed Methodology & Architectures to PDF.
Output: reports/Part3_Methodology_Architectures.pdf
"""

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "Part3_Methodology_Architectures.pdf"

INK = colors.HexColor("#0B1220")
SUB = colors.HexColor("#475569")
MUTED = colors.HexColor("#94A3B8")
ACCENT = colors.HexColor("#0284C7")
PANEL = colors.HexColor("#F8FAFC")
LINE = colors.HexColor("#E2E8F0")
WHITE = colors.white
CNN_C = colors.HexColor("#6366F1")
RES_C = colors.HexColor("#DC2626")
EFF_C = colors.HexColor("#059669")


def st():
    return {
        "title": ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=20, textColor=WHITE, leading=25),
        "sub": ParagraphStyle("S", fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#BAE6FD")),
        "h1": ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=15, textColor=INK, spaceBefore=16, spaceAfter=8),
        "h2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=12, textColor=ACCENT, spaceBefore=12, spaceAfter=5),
        "h3": ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=10.5, textColor=INK, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("B", fontName="Helvetica", fontSize=10, leading=14, textColor=SUB, alignment=TA_JUSTIFY, spaceAfter=6),
        "cap": ParagraphStyle("C", fontName="Helvetica-Oblique", fontSize=8.5, textColor=MUTED, alignment=TA_CENTER, spaceAfter=10),
        "code": ParagraphStyle("CD", fontName="Courier", fontSize=8, textColor=SUB, backColor=PANEL, leftIndent=8, spaceAfter=8),
    }


def tbl(data, widths, header=True):
    t = Table(data, colWidths=widths)
    c = [("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 9),
         ("TEXTCOLOR", (0, 0), (-1, -1), SUB), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
         ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    if header:
        c += [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
              ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PANEL])]
    t.setStyle(TableStyle(c))
    return t


def fig(story, styles, name, cap, w=15.5 * cm):
    p = FIG / name
    if not p.exists():
        return
    img = Image(str(p))
    ar = img.imageHeight / float(img.imageWidth) if img.imageWidth else 0.45
    img.drawWidth = w
    img.drawHeight = w * ar
    story.append(Spacer(1, 0.1 * cm))
    story.append(img)
    story.append(Paragraph(cap, styles["cap"]))


def on_page(c, doc):
    w, h = A4
    c.saveState()
    c.setStrokeColor(LINE)
    c.line(2 * cm, h - 1.3 * cm, w - 2 * cm, h - 1.3 * cm)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(2 * cm, h - 1.05 * cm, "Part 3 · Proposed Methodology & Architectures · src/models.py")
    c.drawRightString(w - 2 * cm, h - 1.05 * cm, "Scientific Writing Part 3")
    c.line(2 * cm, 1.3 * cm, w - 2 * cm, 1.3 * cm)
    c.drawRightString(w - 2 * cm, 0.85 * cm, f"Page {c.getPageNumber()}")
    c.restoreState()


def build():
    s = st()
    story = []

    # Cover
    cover = Table([[Paragraph(
        "Part 3: Proposed Methodology<br/>&amp; Architectures",
        s["title"],
    )]], colWidths=[17 * cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 22), ("TOPPADDING", (0, 0), (-1, -1), 22),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 22),
    ]))
    story.append(cover)
    sub = Table([[Paragraph(
        "Scientific Writing Part 3 · Experimental Design · CNN · Transfer Learning<br/>"
        "<font size='9'>Study: src/models.py · Section 7 · Technical Report</font>",
        s["sub"],
    )]], colWidths=[17 * cm])
    sub.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0F2744")),
        ("LEFTPADDING", (0, 0), (-1, -1), 22), ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(sub)
    story.append(Spacer(1, 0.3 * cm))
    story.append(tbl([
        ["Architectures", "CNN Baseline · ResNet50 · EfficientNetB0"],
        ["Techniques", "Conv2D · Batch Normalization · Dropout · Compound Scaling"],
        ["Date", datetime.now().strftime("%B %d, %Y")],
    ], [4.5 * cm, 12.5 * cm], header=False))

    # 1 Overview
    story.append(PageBreak())
    story.append(Paragraph("1. Overview", s["h1"]))
    story.append(Paragraph(
        "We propose three neural network architectures for four-class retinal disease classification, "
        "implemented in src/models.py. We compare a baseline CNN (from scratch) against two transfer-learning "
        "models: ResNet50 (residual learning) and EfficientNetB0 (compound scaling).",
        s["body"],
    ))
    story.append(tbl([
        ["Model", "Type", "Key Technique"],
        ["CNN Baseline", "From scratch", "Conv + BatchNorm + Dropout"],
        ["ResNet50", "Transfer learning", "Residual skip connections"],
        ["EfficientNetB0", "Transfer learning", "Compound scaling (depth × width × resolution)"],
    ], [4 * cm, 4 * cm, 9 * cm]))

    # 2 Experimental Design
    story.append(Paragraph("2. Experimental Design Framework", s["h1"]))
    story.append(Paragraph(
        "Independent variable: architecture (3 models). Dependent variables: accuracy, F1, ROC-AUC. "
        "Controls: same 70/15/15 split, seed 42, 224×224 input, Adam optimizer, categorical cross-entropy.",
        s["body"],
    ))
    fig(story, s, "fig_part3_experimental_design.png",
        "Figure 1. Experimental design framework — research question, hypothesis, variables, controls.")

    # 3 Pipeline
    story.append(Paragraph("3. Proposed Pipeline", s["h1"]))
    fig(story, s, "fig_part3_training_pipeline.png",
        "Figure 2. End-to-end training pipeline from raw fundus images to model checkpoint.")

    # 4 CNN
    story.append(PageBreak())
    story.append(Paragraph("4. Architecture 1 — Baseline CNN", s["h1"]))
    story.append(Paragraph(
        "Four Conv2D blocks (32→64→128→256 filters) each followed by Batch Normalization, ReLU, and MaxPooling. "
        "Global Average Pooling reduces parameters. Two dense layers (512, 256) use BatchNorm, ReLU, and Dropout (0.5). "
        "L2 regularization (λ=0.001) penalizes large weights.",
        s["body"],
    ))
    fig(story, s, "fig_part3_cnn_architecture.png",
        "Figure 3. Baseline CNN architecture — Conv blocks, BatchNorm, Dropout, L2 regularization.")

    story.append(tbl([
        ["Block", "Filters", "Kernel", "After Pool"],
        ["1", "32", "3×3", "112×112"],
        ["2", "64", "3×3", "56×56"],
        ["3", "128", "3×3", "28×28"],
        ["4", "256", "3×3", "14×14"],
    ], [3 * cm, 3 * cm, 3 * cm, 8 * cm]))

    # 5 & 6 Transfer Learning
    story.append(Paragraph("5. Architecture 2 — ResNet50", s["h1"]))
    story.append(Paragraph(
        "ResNet50 uses residual connections y = F(x) + x to enable training of very deep networks. "
        "We load ImageNet-pretrained weights, freeze the base, and attach a custom head: "
        "GAP → BN → Dropout(0.5) → Dense(512) → Dense(256) → Softmax(4). "
        "Optional fine-tuning unfreezes the last 20 layers.",
        s["body"],
    ))

    story.append(Paragraph("6. Architecture 3 — EfficientNetB0", s["h1"]))
    story.append(Paragraph(
        "EfficientNetB0 applies compound scaling — jointly scaling depth, width, and resolution "
        "for optimal accuracy-efficiency trade-off. MBConv blocks with Swish activation in the custom head. "
        "Dropout rate 0.3 (lighter than ResNet50). ~5M parameters vs ~25M for ResNet50.",
        s["body"],
    ))
    fig(story, s, "fig_part3_transfer_learning.png",
        "Figure 4. Transfer learning architectures — ResNet50 vs EfficientNetB0.")

    story.append(tbl([
        ["Property", "CNN", "ResNet50", "EfficientNetB0"],
        ["Pretrained", "No", "ImageNet", "ImageNet"],
        ["BatchNorm", "All blocks", "Head", "Head"],
        ["Dropout", "0.5", "0.5/0.25", "0.3/0.15"],
        ["Activation", "ReLU", "ReLU", "Swish"],
        ["~Params", "2–3M", "~25M", "~5M"],
    ], [3.5 * cm, 3.5 * cm, 3.5 * cm, 6.5 * cm]))

    # 7 Training
    story.append(PageBreak())
    story.append(Paragraph("7. Training Methodology", s["h1"]))
    story.append(Paragraph(
        "All models compile with Adam (lr=0.001), categorical cross-entropy, and metrics: "
        "accuracy, precision, recall, weighted F1. Callbacks: ModelCheckpoint, EarlyStopping, "
        "ReduceLROnPlateau, TensorBoard, CSVLogger.",
        s["body"],
    ))
    story.append(tbl([
        ["Parameter", "Value"],
        ["Batch size", "32"],
        ["Epochs", "50 (early stopping)"],
        ["Optimizer", "Adam"],
        ["Learning rate", "0.001"],
        ["Loss", "Categorical cross-entropy"],
    ], [8 * cm, 8.5 * cm]))

    story.append(Paragraph("7.1 Implementation Reference (src/models.py)", s["h2"]))
    story.append(tbl([
        ["Function", "Role"],
        ["create_baseline_cnn()", "Custom 4-block CNN"],
        ["create_resnet50_model()", "ResNet50 + custom head"],
        ["create_efficientnet_model()", "EfficientNetB0 + custom head"],
        ["compile_model()", "Optimizer, loss, metrics"],
        ["create_model()", "Factory: cnn_baseline | resnet50 | efficientnet"],
    ], [6 * cm, 10.5 * cm]))

    story.append(Paragraph(
        "model = create_model('efficientnet', input_shape=(224,224,3), num_classes=4)<br/>"
        "model = compile_model(model, learning_rate=0.001, optimizer_name='adam')",
        s["code"],
    ))

    # References
    story.append(Paragraph("References", s["h1"]))
    refs = [
        "[1] K. He et al., \"Deep residual learning for image recognition,\" CVPR, 2016.",
        "[2] M. Tan and Q. Le, \"EfficientNet: Rethinking model scaling,\" ICML, 2019.",
        "[3] S. Ioffe and C. Szegedy, \"Batch normalization,\" ICML, 2015.",
        "[4] N. Srivastava et al., \"Dropout,\" JMLR, 2014.",
        "[5] K. Sainani, Writing in the Sciences Part 3, Stanford Online, 2020.",
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("R", fontName="Helvetica", fontSize=9, textColor=SUB, spaceAfter=4, leftIndent=14, firstLineIndent=-14)))

    return story


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                              topMargin=1.7*cm, bottomMargin=1.7*cm)
    doc.build(build(), onFirstPage=on_page, onLaterPages=on_page)
    print(f"Part 3 exported: {OUT}")


if __name__ == "__main__":
    main()
