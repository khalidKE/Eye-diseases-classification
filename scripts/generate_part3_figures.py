"""
Generate Part 3 figures: Methodology, Architectures, and Lab Pipeline.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"

C = {
    "bg": "#FFFFFF", "ink": "#0B1220", "sub": "#475569", "muted": "#94A3B8",
    "line": "#E2E8F0", "accent": "#0284C7", "cnn": "#6366F1",
    "resnet": "#DC2626", "effnet": "#059669", "panel": "#F8FAFC",
}


def setup():
    plt.rcParams.update({
        "figure.facecolor": C["bg"], "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial"], "figure.dpi": 200,
    })


def save(name, fig):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight", facecolor=C["bg"], pad_inches=0.35)
    plt.close(fig)
    print(f"Saved: {FIG_DIR / name}")


def fig_training_pipeline():
    """End-to-end lab training pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7, 8.5, "Proposed Methodology — End-to-End Training Pipeline",
            ha="center", fontsize=16, fontweight="bold", color=C["ink"])
    ax.text(7, 8.05, "Experimental design framework · src/models.py · src/train.py",
            ha="center", fontsize=10, color=C["sub"])

    rows = [
        (6.8, "INPUT", ["Raw Fundus Images\n(4 Classes)"], C["accent"]),
        (5.5, "PART 1 — PREPROCESSING", ["Medical Crop", "Resize 224×224", "Normalize"], "#334155"),
        (4.2, "DATA SPLIT", ["Train 70%", "Val 15%", "Test 15%"], "#475569"),
        (3.0, "AUGMENTATION", ["Rotation ±15°", "Flip · Zoom", "Brightness"], "#64748B"),
        (1.7, "MODEL SELECTION (Part 3)", ["CNN Baseline", "ResNet50", "EfficientNetB0"], C["cnn"]),
        (0.4, "TRAINING & OUTPUT", ["Adam · Cross-Entropy", "Callbacks · Checkpoint", "Best .h5 Model"], C["resnet"]),
    ]

    for y, title, boxes, color in rows:
        band = FancyBboxPatch((0.3, y - 0.05), 13.4, 0.95 if len(boxes) <= 1 else 1.05,
                              boxstyle="round,pad=0.03,rounding_size=0.1",
                              facecolor="white", edgecolor=C["line"], linewidth=1.2)
        ax.add_patch(band)
        ax.text(0.55, y + 0.65, title, fontsize=9, fontweight="bold", color=C["ink"])
        n = len(boxes)
        for i, label in enumerate(boxes):
            x = 0.7 + i * (12.8 / max(n, 1))
            w = min(3.5, 12.5 / n)
            chip = FancyBboxPatch((x, y + 0.05), w, 0.48, boxstyle="round,pad=0.03,rounding_size=0.08",
                                  facecolor=color, edgecolor="none", alpha=0.92)
            ax.add_patch(chip)
            ax.text(x + w / 2, y + 0.29, label, ha="center", va="center",
                    fontsize=8, fontweight="600", color="white")

    for y_from, y_to in [(6.5, 6.0), (5.2, 4.7), (3.9, 3.5), (2.7, 2.3), (1.4, 1.0)]:
        ax.annotate("", xy=(7, y_to), xytext=(7, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=2))

    save("fig_part3_training_pipeline.png", fig)


def fig_cnn_architecture():
    """Baseline CNN architecture from src/models.py."""
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6.5, 7.5, "Baseline CNN Architecture (from scratch)",
            ha="center", fontsize=15, fontweight="bold", color=C["ink"])
    ax.text(6.5, 7.05, "Conv2D → BatchNorm → ReLU → MaxPool  ·  L2 Regularization  ·  Dropout 0.5",
            ha="center", fontsize=9.5, color=C["sub"])

    blocks = [
        ("Input\n224×224×3", C["accent"], 0.4),
        ("Conv Block 1\n32 filters · 3×3\nBN · ReLU · Pool", C["cnn"], 2.0),
        ("Conv Block 2\n64 filters · 3×3\nBN · ReLU · Pool", C["cnn"], 3.6),
        ("Conv Block 3\n128 filters · 3×3\nBN · ReLU · Pool", C["cnn"], 5.2),
        ("Conv Block 4\n256 filters · 3×3\nBN · ReLU · Pool", C["cnn"], 6.8),
        ("GAP\nGlobal Avg Pool", "#475569", 8.4),
        ("Dense 512\nBN · ReLU · Dropout", "#64748B", 9.8),
        ("Dense 256\nBN · ReLU · Dropout", "#64748B", 11.2),
        ("Softmax\n4 Classes", C["resnet"], 12.4),
    ]

    for label, color, x in blocks:
        w = 1.15 if "Block" not in label else 1.35
        rect = FancyBboxPatch((x, 2.8), w, 2.2, boxstyle="round,pad=0.05,rounding_size=0.12",
                              facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.93)
        ax.add_patch(rect)
        ax.text(x + w / 2, 3.9, label, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white")

    for i in range(len(blocks) - 1):
        x1 = blocks[i][2] + (1.35 if "Block" in blocks[i][0] else 1.15)
        x2 = blocks[i + 1][2]
        ax.annotate("", xy=(x2, 3.9), xytext=(x1, 3.9),
                    arrowprops=dict(arrowstyle="-|>", color=C["muted"], lw=1.5))

    # Legend boxes
    legends = [
        (0.5, 1.2, "Batch Normalization", "Stabilizes layer activations; faster convergence"),
        (4.5, 1.2, "Dropout (0.5)", "Randomly drops neurons; reduces overfitting"),
        (8.5, 1.2, "L2 Regularization", "Penalizes large weights; λ = 0.001"),
    ]
    for x, y, title, desc in legends:
        box = FancyBboxPatch((x, y), 3.8, 1.0, boxstyle="round,pad=0.04,rounding_size=0.08",
                             facecolor=C["panel"], edgecolor=C["line"])
        ax.add_patch(box)
        ax.text(x + 0.15, y + 0.65, title, fontsize=9, fontweight="bold", color=C["ink"])
        ax.text(x + 0.15, y + 0.25, desc, fontsize=7.5, color=C["sub"])

    save("fig_part3_cnn_architecture.png", fig)


def fig_transfer_learning():
    """ResNet50 vs EfficientNetB0 transfer learning comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle("Transfer Learning Architectures — ResNet50 vs EfficientNetB0",
                 fontsize=15, fontweight="bold", y=1.02)

    configs = [
        ("ResNet50\n(Deep Residual Learning)", C["resnet"], [
            ("ImageNet\nPretrained Base", "Frozen · 50 layers"),
            ("Residual Blocks", "Skip connections\nsolve vanishing gradients"),
            ("Custom Head", "GAP → BN → Dropout 0.5"),
            ("Dense Layers", "512 → 256 → Softmax(4)"),
            ("Fine-tune Option", "Unfreeze last 20 layers"),
        ]),
        ("EfficientNetB0\n(Compound Scaling)", C["effnet"], [
            ("ImageNet\nPretrained Base", "Frozen · MBConv blocks"),
            ("Compound Scaling", "Depth × Width × Resolution"),
            ("Swish Activation", "x · sigmoid(x) in head"),
            ("Custom Head", "GAP → BN → Dropout 0.3"),
            ("Dense Layer", "256 → Softmax(4)"),
        ]),
    ]

    for ax, (title, color, layers) in zip(axes, configs):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.text(5, 9.3, title, ha="center", fontsize=12, fontweight="bold", color=color)

        for i, (layer_title, layer_desc) in enumerate(layers):
            y = 7.5 - i * 1.55
            rect = FancyBboxPatch((1.5, y - 0.45), 7, 1.1, boxstyle="round,pad=0.04,rounding_size=0.1",
                                  facecolor=color if i == 0 else C["panel"],
                                  edgecolor=color, linewidth=1.5,
                                  alpha=0.9 if i == 0 else 1.0)
            ax.add_patch(rect)
            tc = "white" if i == 0 else C["ink"]
            ax.text(5, y + 0.15, layer_title, ha="center", va="center", fontsize=9, fontweight="bold", color=tc)
            ax.text(5, y - 0.2, layer_desc, ha="center", va="center", fontsize=7.5, color=C["sub"] if i else "#FECACA")
            if i < len(layers) - 1:
                ax.annotate("", xy=(5, y - 0.55), xytext=(5, y - 0.45),
                            arrowprops=dict(arrowstyle="-|>", color=C["muted"], lw=1.2))

    save("fig_part3_transfer_learning.png", fig)


def fig_experimental_design():
    """Experimental design framework diagram."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6.5, 6.5, "Experimental Design Framework",
            ha="center", fontsize=16, fontweight="bold", color=C["ink"])

    cols = [
        ("Research Question", "Can CNN and transfer-learning\nmodels classify retinal diseases?", C["accent"]),
        ("Hypothesis", "Transfer learning outperforms\nbaseline CNN on fundus data", C["cnn"]),
        ("Variables", "IV: Architecture (3 models)\nDV: Accuracy, F1, ROC-AUC", "#475569"),
        ("Controls", "Same split · seed 42 · 224×224\nSame optimizer & loss", "#64748B"),
        ("Procedure", "Train → Validate → Test\nCompare best checkpoint", C["effnet"]),
    ]

    for i, (title, body, color) in enumerate(cols):
        x = 0.4 + i * 2.55
        card = FancyBboxPatch((x, 1.0), 2.35, 4.8, boxstyle="round,pad=0.06,rounding_size=0.14",
                              facecolor="white", edgecolor=color, linewidth=2)
        ax.add_patch(card)
        header = FancyBboxPatch((x, 5.0), 2.35, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="none")
        ax.add_patch(header)
        ax.text(x + 1.17, 5.4, title, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        ax.text(x + 1.17, 3.2, body, ha="center", va="center", fontsize=8.5, color=C["sub"])

    ax.text(6.5, 0.45, "Framework aligned with Scientific Writing Part 3 · reproducible experimental protocol",
            ha="center", fontsize=9, color=C["muted"], style="italic")

    save("fig_part3_experimental_design.png", fig)


def main():
    setup()
    fig_training_pipeline()
    fig_cnn_architecture()
    fig_transfer_learning()
    fig_experimental_design()
    print("\nPart 3 methodology figures generated.")


if __name__ == "__main__":
    main()
