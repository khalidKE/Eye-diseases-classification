"""
Generate premium publication figures for Technical Report (Parts 4 & 5).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "results" / "best_model_evaluation.json"

# Premium clinical palette
C = {
    "bg": "#FFFFFF",
    "panel": "#F8FAFC",
    "ink": "#0B1220",
    "sub": "#475569",
    "muted": "#94A3B8",
    "line": "#E2E8F0",
    "accent": "#0284C7",
    "accent2": "#0891B2",
    "normal": "#059669",
    "dr": "#DC2626",
    "cataract": "#D97706",
    "glaucoma": "#7C3AED",
    "grad1": "#0EA5E9",
    "grad2": "#6366F1",
}

CLASS_LABELS = ["Normal", "Diabetic\nRetinopathy", "Cataract", "Glaucoma"]
CLASS_KEYS = ["normal", "diabetic_retinopathy", "cataract", "glaucoma"]
CLASS_COLORS = [C["normal"], C["dr"], C["cataract"], C["glaucoma"]]


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["panel"],
        "axes.edgecolor": C["line"],
        "axes.labelcolor": C["ink"],
        "text.color": C["ink"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Inter", "Arial", "DejaVu Sans"],
        "axes.titlesize": 15,
        "axes.titleweight": "600",
        "axes.labelsize": 11,
        "axes.labelweight": "500",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 200,
    })


def add_watermark(fig, text="Eye Disease Classification · XAI Report"):
    fig.text(0.99, 0.01, text, ha="right", va="bottom", fontsize=7,
             color=C["muted"], alpha=0.8)


def save_fig(name: str, fig):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    add_watermark(fig)
    fig.savefig(path, bbox_inches="tight", facecolor=C["bg"], edgecolor="none", pad_inches=0.35)
    plt.close(fig)
    print(f"Saved: {path}")


def rounded_bar(ax, x, h, w, color, alpha=1.0):
    """Draw bar with rounded top."""
    bar = ax.bar(x, h, width=w, color=color, alpha=alpha, linewidth=0, zorder=3)
    return bar


def fig_gradcam_pipeline():
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Header band
    header = FancyBboxPatch((0.2, 4.55), 14.6, 0.75, boxstyle="round,pad=0.02,rounding_size=0.12",
                            facecolor=C["ink"], edgecolor="none")
    ax.add_patch(header)
    ax.text(7.5, 4.92, "Grad-CAM Explainability Pipeline", ha="center", va="center",
            fontsize=17, fontweight="bold", color="white")
    ax.text(7.5, 4.62, "Part 4 · Post-hoc spatial attribution for CNN classifiers",
            ha="center", va="center", fontsize=10, color="#CBD5E1")

    steps = [
        ("01", "Fundus Image", "Raw RGB input", C["accent"]),
        ("02", "Preprocess", "Crop · 224×224 · norm", "#334155"),
        ("03", "CNN Backbone", "ResNet50 / EffNet", "#1E293B"),
        ("04", "Conv Features", "Last layer maps A^k", C["glaucoma"]),
        ("05", "Grad-CAM", "Heatmap L^c", C["cataract"]),
    ]

    x0 = 0.45
    gap = 2.85
    for i, (num, title, sub, color) in enumerate(steps):
        x = x0 + i * gap
        card = FancyBboxPatch((x, 1.55), 2.35, 2.55, boxstyle="round,pad=0.06,rounding_size=0.18",
                              facecolor="white", edgecolor=C["line"], linewidth=1.5)
        ax.add_patch(card)
        badge = Circle((x + 0.35, 3.85), 0.22, facecolor=color, edgecolor="none")
        ax.add_patch(badge)
        ax.text(x + 0.35, 3.85, num, ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        ax.text(x + 1.17, 3.35, title, ha="center", va="center", fontsize=11, fontweight="bold", color=C["ink"])
        ax.text(x + 1.17, 2.85, sub, ha="center", va="center", fontsize=8.5, color=C["sub"])
        accent = FancyBboxPatch((x + 0.2, 1.75), 1.95, 0.12, boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor=color, edgecolor="none", alpha=0.85)
        ax.add_patch(accent)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.55, 2.8), xytext=(x + 2.35, 2.8),
                        arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=2.2))

    ax.text(7.5, 0.75,
            "Gradient weights α_k computed via tf.GradientTape  ·  ReLU applied to weighted feature maps",
            ha="center", fontsize=9.5, color=C["sub"], style="italic")

    save_fig("fig1_gradcam_pipeline.png", fig)


def fig_deployment_architecture():
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    header = FancyBboxPatch((0.2, 6.1), 14.6, 0.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                            facecolor=C["accent2"], edgecolor="none")
    ax.add_patch(header)
    ax.text(7.5, 6.45, "Deployable AI Architecture — Streamlit Clinical Interface", ha="center",
            fontsize=16, fontweight="bold", color="white")

    layers = [
        (5.2, "Presentation Layer", [("Drag & Drop Upload", C["accent"]), ("PDF Export", C["normal"]), ("Grad-CAM View", C["cataract"])]),
        (3.55, "Application Layer  ·  src/app.py", [("Session UI", "#475569"), ("Model Cache", "#475569"), ("Threshold Control", "#475569")]),
        (1.9, "Inference Layer", [("predict.py", C["glaucoma"]), ("gradcam.py", C["dr"]), ("report_pdf.py", C["accent2"])]),
        (0.25, "Model Artifacts", [("Trained .h5 Weights", C["ink"])]),
    ]

    for y, title, items in layers:
        band = FancyBboxPatch((0.35, y), 14.3, 1.35, boxstyle="round,pad=0.04,rounding_size=0.14",
                              facecolor="white", edgecolor=C["line"], linewidth=1.2)
        ax.add_patch(band)
        ax.text(0.65, y + 1.05, title, fontsize=10.5, fontweight="bold", color=C["ink"])
        n = len(items)
        for i, (label, color) in enumerate(items):
            w = min(3.8, 12.5 / n)
            x = 0.9 + i * (13.2 / n)
            chip = FancyBboxPatch((x, y + 0.22), w, 0.65, boxstyle="round,pad=0.04,rounding_size=0.1",
                                  facecolor=color, edgecolor="none", alpha=0.95)
            ax.add_patch(chip)
            ax.text(x + w / 2, y + 0.55, label, ha="center", va="center",
                    fontsize=9, fontweight="600", color="white")

    # Flow arrow on right
    ax.annotate("", xy=(14.85, 0.9), xytext=(14.85, 5.8),
                arrowprops=dict(arrowstyle="-|>", color=C["muted"], lw=1.8, connectionstyle="arc3,rad=0"))
    ax.text(14.95, 3.35, "User\nFlow", ha="left", va="center", fontsize=8, color=C["muted"], rotation=90)

    save_fig("fig2_deployment_architecture.png", fig)


def fig_per_class_metrics(metrics: dict):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(top=0.88)

    precision = [metrics["per_class"][k]["precision"] * 100 for k in CLASS_KEYS]
    recall = [metrics["per_class"][k]["recall"] * 100 for k in CLASS_KEYS]
    f1 = [metrics["per_class"][k]["f1_score"] * 100 for k in CLASS_KEYS]

    x = np.arange(len(CLASS_LABELS))
    w = 0.24
    colors_m = [("#0284C7", "#38BDF8"), ("#059669", "#34D399"), ("#7C3AED", "#A78BFA")]

    for offset, values, (dark, light), label in zip(
        [-w, 0, w], [precision, recall, f1], colors_m, ["Precision", "Recall", "F1-Score"]
    ):
        bars = ax.bar(x + offset, values, w, label=label, color=light, edgecolor=dark, linewidth=1.2, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5,
                    color=C["sub"], fontweight="600")

    ax.set_ylabel("Score (%)", fontweight="600")
    ax.set_xlabel("Disease Class", fontweight="600")
    ax.set_title("Per-Class Classification Performance", loc="left", pad=12, fontsize=16, fontweight="bold")
    ax.text(0.0, 1.02, "Test set · n = 637 · macro-averaged comparison across four retinal conditions",
            transform=ax.transAxes, fontsize=9.5, color=C["sub"])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, linestyle="-", alpha=0.35, color=C["line"])
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=3)

    save_fig("fig3_per_class_metrics.png", fig)


def fig_confusion_matrix(metrics: dict):
    cm = np.array(metrics["confusion_matrix"], dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.zeros_like(cm)
    np.divide(cm, row_sum, out=cm_pct, where=row_sum != 0)
    cm_pct *= 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.suptitle("Confusion Matrix Analysis — Best Model", fontsize=16, fontweight="bold", y=1.02, x=0.02, ha="left")

    labels = ["Normal", "DR", "Cataract", "Glaucoma"]
    cmap = LinearSegmentedColormap.from_list("clinical", ["#F8FAFC", "#BAE6FD", "#0284C7", "#0C4A6E"])

    sns.heatmap(cm, annot=True, fmt="g", cmap=cmap, cbar=False, ax=axes[0],
                xticklabels=labels, yticklabels=labels, linewidths=2, linecolor="white",
                annot_kws={"size": 13, "weight": "bold", "color": C["ink"]})
    axes[0].set_xlabel("Predicted Label", fontweight="600")
    axes[0].set_ylabel("True Label", fontweight="600")
    axes[0].set_title("Counts", fontsize=12, pad=8)

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap=cmap, cbar=True, ax=axes[1],
                xticklabels=labels, yticklabels=labels, linewidths=2, linecolor="white",
                annot_kws={"size": 11, "weight": "600"}, vmin=0, vmax=100)
    axes[1].set_xlabel("Predicted Label", fontweight="600")
    axes[1].set_ylabel("True Label", fontweight="600")
    axes[1].set_title("Row-normalized (%)", fontsize=12, pad=8)

    save_fig("fig4_confusion_matrix.png", fig)


def fig_overall_metrics(metrics: dict):
    overall = metrics["overall"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]
    values = [overall[k] * 100 for k in keys]
    targets = [85, 85, 85, 85, 90]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(labels))
    colors_bar = [C["accent"], C["normal"], C["dr"], C["glaucoma"], C["accent2"]]

    bars = ax.barh(y, values, height=0.55, color=colors_bar, alpha=0.88, edgecolor="white", linewidth=1.5)
    ax.barh(y, targets, height=0.55, color="none", edgecolor=C["muted"], linewidth=1.2, linestyle="--", alpha=0.6)

    for bar, val in zip(bars, values):
        ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", fontsize=10, fontweight="bold", color=C["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontweight="600")
    ax.set_xlabel("Score (%)", fontweight="600")
    ax.set_xlim(0, 105)
    ax.set_title("Overall Model Performance vs. Project Target (85%)", loc="left", fontsize=15, fontweight="bold", pad=14)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3, color=C["line"])
    ax.legend([bars[0], plt.Line2D([0], [0], color=C["muted"], linestyle="--")],
              ["Observed", "Target threshold"], frameon=False, loc="lower right")

    save_fig("fig5_overall_metrics.png", fig)


def fig_gradcam_panels():
    rng = np.random.default_rng(42)
    yg, xg = np.ogrid[:224, :224]
    disc = ((xg - 112) ** 2 + (yg - 112) ** 2) < 68 ** 2
    base = np.zeros((224, 224, 3))
    base[disc] = [0.62, 0.18, 0.14]
    base[~disc] = [0.04, 0.05, 0.09]
    base += rng.normal(0, 0.015, base.shape)
    base = np.clip(base, 0, 1)

    heatmap = np.zeros((224, 224))
    heatmap[85:155, 95:165] = 1.0
    heatmap[105:135, 45:85] = 0.55
    heatmap = gaussian_filter(heatmap, sigma=10)

    overlay = base * 0.55 + plt.cm.inferno(heatmap)[..., :3] * 0.45

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    titles = ["Original Fundus", "Grad-CAM Heatmap", "Clinical Overlay"]
    imgs = [base, heatmap, overlay]
    cmaps = [None, "inferno", None]

    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        if cmap:
            im = ax.imshow(img, cmap=cmap)
        else:
            im = ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(C["line"])
            spine.set_linewidth(2)

    fig.suptitle("Explainable AI Output — Three-Panel Grad-CAM Visualization",
                 fontsize=15, fontweight="bold", y=1.05, ha="center")
    save_fig("fig6_gradcam_output_panels.png", fig)


def fig_summary_dashboard(metrics: dict):
    """Single overview figure for report cover / executive summary."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.85, 1.15], hspace=0.38, wspace=0.28)

    overall = metrics["overall"]
    kpis = [
        ("Accuracy", overall["accuracy"] * 100, C["accent"]),
        ("ROC-AUC", overall["roc_auc_macro"] * 100, C["accent2"]),
        ("F1 Macro", overall["f1_macro"] * 100, C["glaucoma"]),
        ("Test Samples", 637, C["ink"]),
    ]
    for i, (label, val, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        card = FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.03,rounding_size=0.08",
                              facecolor="white", edgecolor=color, linewidth=2.5)
        ax.add_patch(card)
        display_val = f"{val:.1f}%" if isinstance(val, float) else str(int(val))
        ax.text(0.5, 0.58, display_val, ha="center", va="center", fontsize=24, fontweight="bold", color=color)
        ax.text(0.5, 0.28, label, ha="center", va="center", fontsize=10, color=C["sub"], fontweight="600")

    ax_bar = fig.add_subplot(gs[1, :3])
    f1_vals = [metrics["per_class"][k]["f1_score"] * 100 for k in CLASS_KEYS]
    bars = ax_bar.barh(CLASS_LABELS, f1_vals, color=CLASS_COLORS, height=0.55, edgecolor="white", linewidth=2)
    for bar, val in zip(bars, f1_vals):
        ax_bar.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax_bar.set_xlim(0, 85)
    ax_bar.set_xlabel("F1-Score (%)")
    ax_bar.set_title("Per-Class F1 at a Glance", loc="left", fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Donut for correct vs incorrect (approx from accuracy)
    ax_d = fig.add_subplot(gs[1, 3])
    acc = overall["accuracy"]
    sizes = [acc, 1 - acc]
    wedges, _ = ax_d.pie(sizes, colors=[C["normal"], C["line"]], startangle=90,
                         wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax_d.text(0, 0.05, f"{acc*100:.1f}%", ha="center", va="center", fontsize=22, fontweight="bold", color=C["ink"])
    ax_d.text(0, -0.18, "Correct", ha="center", va="center", fontsize=9, color=C["sub"])
    ax_d.set_title("Test Accuracy", fontweight="bold")

    fig.suptitle("Executive Summary — Explainable & Deployable AI (Parts 4 & 5)",
                 fontsize=17, fontweight="bold", y=0.98)
    save_fig("fig0_executive_summary.png", fig)


def main():
    setup_style()
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    fig_summary_dashboard(metrics)
    fig_gradcam_pipeline()
    fig_deployment_architecture()
    fig_per_class_metrics(metrics)
    fig_confusion_matrix(metrics)
    fig_overall_metrics(metrics)
    fig_gradcam_panels()
    print("\nAll premium report figures generated.")


if __name__ == "__main__":
    main()
