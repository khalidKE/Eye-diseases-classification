"""
Generate premium PDF prediction reports for the Streamlit application.
"""

from datetime import datetime
from io import BytesIO
from typing import Dict

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK = colors.HexColor("#0B1220")
SUB = colors.HexColor("#475569")
ACCENT = colors.HexColor("#0284C7")
PANEL = colors.HexColor("#F8FAFC")
LINE = colors.HexColor("#E2E8F0")
WHITE = colors.white

CLASS_DISPLAY = {
    "normal": "Normal",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "cataract": "Cataract",
    "glaucoma": "Glaucoma",
}


def _table(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths)
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), SUB),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
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


def generate_prediction_pdf(
    result: Dict,
    image_name: str = "uploaded_image.jpg",
    model_name: str = "best_model.h5",
    gradcam_available: bool = True,
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    title = ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=20, textColor=INK, alignment=TA_CENTER, spaceAfter=6)
    sub = ParagraphStyle("S", fontName="Helvetica", fontSize=10, textColor=SUB, alignment=TA_CENTER, spaceAfter=16)
    h2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13, textColor=ACCENT, spaceBefore=14, spaceAfter=8)
    body = ParagraphStyle("B", fontName="Helvetica", fontSize=10.5, leading=15, textColor=SUB, alignment=TA_JUSTIFY)

    story = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Cover strip
    cover = Table([[Paragraph("AI Inference Report<br/><font size='10' color='#64748B'>Explainable &amp; Deployable Eye Disease Classification</font>", title)]], colWidths=[17 * cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), INK),
        ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
    ]))
    story.append(cover)
    story.append(Paragraph(f"Session generated · {now}", sub))
    story.append(HRFlowable(width="100%", thickness=1, color=LINE, spaceAfter=12))

    story.append(Paragraph("Inference Summary", h2))
    story.append(_table([
        ["Field", "Value"],
        ["Input Image", image_name],
        ["Model", model_name],
        ["Predicted Class", result.get("predicted_class_display", "N/A")],
        ["Confidence", f"{result.get('confidence', 0) * 100:.2f}%"],
        ["Grad-CAM", "Available" if gradcam_available else "Unavailable"],
    ], [5.5 * cm, 11.5 * cm]))

    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Class Probabilities", h2))
    probs = result.get("all_probabilities", {})
    prob_rows = [["Class", "Probability"]]
    for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        prob_rows.append([CLASS_DISPLAY.get(cls, cls), f"{prob * 100:.2f}%"])
    story.append(_table(prob_rows, [10 * cm, 7 * cm]))

    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Explainability Note", h2))
    story.append(Paragraph(
        "Grad-CAM heatmaps highlight retinal regions that most influenced this prediction. "
        "Review the overlay alongside the predicted class before drawing clinical conclusions.",
        body,
    ))

    story.append(Spacer(1, 0.4 * cm))
    disc = Table([[Paragraph(
        "<b>Research Use Only.</b> Not for clinical diagnosis. Consult a qualified ophthalmologist.",
        ParagraphStyle("D", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#92400E"), leading=13),
    )]], colWidths=[17 * cm])
    disc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#F59E0B")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(disc)

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
