"""
ציור דיאגרמת הארכיטקטורה של מסווג האודיו (wav2vec2 + attention pooling + פיצ'רים).

שימוש:
    python scripts/plot_architecture.py

פלט: docs/model_architecture.png
התוויות באנגלית בכוונה — matplotlib לא מציג עברית RTL כראוי, וזו גם
הקונבנציה המקובלת לדיאגרמות ארכיטקטורה.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"

# ── פלטת צבעים ────────────────────────────────────────────
C_FROZEN = "#aed6f1"   # שכבות קפואות (encoder) — תכלת
C_TRAIN = "#a9dfbf"    # שכבות מתאמנות — ירוק
C_FEAT = "#f5cba7"     # פיצ'רים אקוסטיים — כתום
C_OP = "#d5d8dc"       # פעולות (concat) — אפור
C_IO = "#d2b4de"       # קלט/פלט — סגול
EDGE = "#566573"


def box(ax, cx, cy, w, h, text, color, fontsize=10, bold_first=True):
    """מצייר תיבה ממורכזת עם טקסט במרכז."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.6, edgecolor=EDGE, facecolor=color, zorder=2,
        )
    )
    if bold_first and "\n" in text:
        head, rest = text.split("\n", 1)
        ax.text(cx, cy + h * 0.18, head, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=3)
        ax.text(cx, cy - h * 0.22, rest, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#34495e", zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, zorder=3)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=16,
            linewidth=1.6, color=EDGE, zorder=1,
            connectionstyle="arc3,rad=0",
        )
    )
    if label:
        ax.text((x1 + x2) / 2 + 1.5, (y1 + y2) / 2, label,
                ha="left", va="center", fontsize=7.5, color="#7f8c8d",
                style="italic", zorder=3)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 13.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 132)
    ax.axis("off")

    ax.text(50, 128, "Audio Classifier — Model Architecture",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(50, 123.5, "wav2vec2-xls-r-300m  +  Attention Pooling  +  category-tuned acoustic features",
            ha="center", fontsize=9.5, color="#566573")

    MAIN = 33   # ציר ה-pipeline הראשי
    FEAT = 78   # ציר ענף הפיצ'רים
    MID = 50    # ציר משותף (concat ומטה)

    # ── התיבות ─────────────────────────────────────────────
    box(ax, MAIN, 114, 30, 9,
        "Raw Audio\n16 kHz · mono · up to 5 s   (B, 80000)", C_IO)
    box(ax, MAIN, 98, 36, 12,
        "wav2vec2-xls-r-300m Encoder\nFROZEN · ~300M params   →  (B, T, 1024)", C_FROZEN)
    box(ax, MAIN, 82, 30, 9,
        "Attention Pooling\nlearned frame weights   →  (B, 1024)", C_TRAIN)

    box(ax, FEAT, 90, 34, 13,
        "Acoustic Features (librosa)\n15 category-tuned features\ntempo · DTMF · bandwidth · pitch …   (B, 15)", C_FEAT)

    box(ax, MID, 66, 26, 7,
        "Concatenate   →  (B, 1024 + 15 = 1039)", C_OP, fontsize=9, bold_first=False)

    # ראש הסיווג — תיבה מקובצת עם השכבות הפנימיות
    ax.add_patch(
        FancyBboxPatch(
            (MID - 21, 38), 42, 18,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.8, edgecolor=EDGE, facecolor=C_TRAIN, zorder=2,
        )
    )
    ax.text(MID, 54.2, "Classifier Head (trainable)", ha="center",
            fontsize=10, fontweight="bold", zorder=3)
    for i, line in enumerate([
        "Linear(1039 → 128)",
        "GELU",
        "Dropout(0.2)",
        "Linear(128 → 4)",
    ]):
        ax.text(MID, 50.5 - i * 2.8, line, ha="center", va="center",
                fontsize=9, color="#1b4f3c", zorder=3)

    box(ax, MID, 28, 24, 8, "Softmax\n4 logits → probabilities", C_OP)
    box(ax, MID, 12, 56, 9,
        "Prediction:  human  ·  ivr  ·  music  ·  recording", C_IO,
        fontsize=10.5, bold_first=False)

    # ── חיצים ──────────────────────────────────────────────
    arrow(ax, MAIN, 109.5, MAIN, 104)            # audio → encoder
    arrow(ax, MAIN, 92, MAIN, 86.5)              # encoder → pooling
    # audio → features (אלבו ימינה ואז למטה)
    ax.plot([MAIN + 15, FEAT], [114, 114], color=EDGE, linewidth=1.6, zorder=1)
    ax.add_patch(FancyArrowPatch((FEAT, 114), (FEAT, 96.5),
                 arrowstyle="-|>", mutation_scale=16, linewidth=1.6,
                 color=EDGE, zorder=1))
    # pooling → concat (אלבו)
    arrow(ax, MAIN, 77.5, MAIN, 69.5)
    arrow(ax, MAIN, 69.5, MID - 13, 67)
    # features → concat
    arrow(ax, FEAT, 83.5, FEAT, 69.5)
    arrow(ax, FEAT, 69.5, MID + 13, 67)
    # concat → classifier → softmax → output
    arrow(ax, MID, 62.5, MID, 56.2)
    arrow(ax, MID, 38, MID, 32)
    arrow(ax, MID, 24, MID, 16.5)

    # ── מקרא ───────────────────────────────────────────────
    legend = [
        (C_FROZEN, "Frozen (pretrained)"),
        (C_TRAIN, "Trainable"),
        (C_FEAT, "Feature engineering"),
        (C_OP, "Operation"),
        (C_IO, "Input / Output"),
    ]
    lx, ly = 2, 72
    for i, (col, lab) in enumerate(legend):
        y = ly - i * 4
        ax.add_patch(FancyBboxPatch((lx, y - 1.3), 3, 2.6,
                     boxstyle="round,pad=0.1,rounding_size=0.5",
                     linewidth=1, edgecolor=EDGE, facecolor=col))
        ax.text(lx + 4, y, lab, ha="left", va="center", fontsize=8)

    out_dir = Path(__file__).resolve().parent.parent / "docs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "model_architecture.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"נשמר: {out_path}")


if __name__ == "__main__":
    main()
