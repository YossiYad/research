"""
ציור דיאגרמת ארכיטקטורת המערכת (system architecture) של מסווג האודיו —
גרסה דיגיטלית נקייה של סקיצת הלוח: UI/UX, Main APP (PC/Cloud) עם המודל,
DB, ומטריצת בלבול.

שימוש:
    python scripts/plot_system_architecture.py

פלט: docs/system_architecture.png
התוויות באנגלית בכוונה — matplotlib לא מציג עברית RTL כראוי.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"

C_APP = "#d6eaf8"     # Main APP
C_INNER = "#ffffff"   # תיבה פנימית
C_ML = "#a9dfbf"      # מודל ה-ML
C_UI = "#fcf3cf"      # UI/UX
C_DB = "#d2b4de"      # מסד נתונים
C_IO = "#f5cba7"      # קלט/פלט
EDGE = "#34495e"


def rbox(ax, cx, cy, w, h, color, lw=1.6, ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        linewidth=lw, edgecolor=EDGE, facecolor=color, linestyle=ls, zorder=2))


def label(ax, cx, cy, text, fs=10, bold=False, color="#1b2631"):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color, zorder=4)


def arrow(ax, x1, y1, x2, y2, text=None, rad=0.0, fs=9, dy=2.2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.8, color=EDGE, zorder=3,
        connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center",
                va="center", fontsize=fs, color="#566573", style="italic", zorder=4)


def cylinder(ax, cx, cy, w, h, color):
    eh = h * 0.16
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + eh / 2), w, h - eh,
                 facecolor=color, edgecolor="none", zorder=2))
    ax.add_patch(Ellipse((cx, cy - h / 2 + eh / 2), w, eh,
                 facecolor=color, edgecolor=EDGE, lw=1.6, zorder=2))
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2 + eh / 2, cy + h / 2 - eh / 2],
            color=EDGE, lw=1.6, zorder=3)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2 + eh / 2, cy + h / 2 - eh / 2],
            color=EDGE, lw=1.6, zorder=3)
    ax.add_patch(Ellipse((cx, cy + h / 2 - eh / 2), w, eh,
                 facecolor=color, edgecolor=EDGE, lw=1.6, zorder=3))


def conf_matrix(ax, x0, y0, cell, labels):
    n = len(labels)
    for i in range(n):  # תאי האלכסון מודגשים
        rx, ry = x0 + i * cell, y0 + (n - 1 - i) * cell
        ax.add_patch(Rectangle((rx, ry), cell, cell, facecolor=C_ML,
                     edgecolor="none", zorder=1))
        ax.text(rx + cell / 2, ry + cell / 2, "✓", ha="center", va="center",
                fontsize=10, color="#1b4f3c", zorder=2)
    for i in range(n + 1):
        ax.plot([x0, x0 + n * cell], [y0 + i * cell] * 2, color=EDGE, lw=1.1, zorder=2)
        ax.plot([x0 + i * cell] * 2, [y0, y0 + n * cell], color=EDGE, lw=1.1, zorder=2)
    for i, lab in enumerate(labels):
        ax.text(x0 - 1.0, y0 + (n - 1 - i) * cell + cell / 2, lab,
                ha="right", va="center", fontsize=8.5)
        ax.text(x0 + i * cell + cell / 2, y0 - 1.2, lab,
                ha="center", va="top", fontsize=8.5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97, "System Architecture — Audio Classifier (Holdie)",
            ha="center", fontsize=15, fontweight="bold")

    # ── Main APP ───────────────────────────────────────────
    rbox(ax, 26, 70, 46, 42, C_APP)
    label(ax, 26, 88, "Main APP", fs=13, bold=True)
    label(ax, 26, 84.5, "(runs on PC / Cloud)", fs=8.5, color="#566573")

    rbox(ax, 28, 69, 36, 24, C_INNER)
    # מודל ה-ML
    rbox(ax, 16, 69, 11, 9, C_ML)
    label(ax, 16, 70.5, "ML", fs=10, bold=True)
    label(ax, 16, 67, "model", fs=7.5, color="#1b4f3c")
    # עמודת הקלט
    rbox(ax, 35, 77, 18, 5, C_IO)
    label(ax, 35, 77, "tel / audio in", fs=8.5)
    rbox(ax, 35, 70, 18, 5, "#fdebd0")
    label(ax, 35, 70, "DATA", fs=9, bold=True)
    rbox(ax, 35, 63, 18, 5, "#fdebd0")
    label(ax, 35, 63, "★ Features", fs=8.5)
    arrow(ax, 35, 74.5, 35, 72.5)      # tel → DATA
    arrow(ax, 35, 67.5, 35, 65.5)      # DATA → Features
    arrow(ax, 26, 65, 21, 68)          # Features → ML
    # פלט %
    ax.add_patch(Ellipse((14, 55), 9, 6, facecolor=C_IO, edgecolor=EDGE, lw=1.6, zorder=3))
    label(ax, 14, 55.6, "%", fs=12, bold=True)
    label(ax, 14, 53.2, "score", fs=7, color="#566573")
    arrow(ax, 16, 64.5, 14, 58)        # ML → %

    # ── UI / UX ────────────────────────────────────────────
    rbox(ax, 78, 70, 38, 44, C_UI)
    label(ax, 76, 88, "UI / UX", fs=14, bold=True)
    ax.text(90, 89, "★", fontsize=13, color="#b7950b", ha="center", va="center", zorder=4)
    ax.plot([61, 95], [85, 85], color=EDGE, lw=1.4, zorder=3)
    label(ax, 78, 70, "(client screen)", fs=9, color="#7f8c8d")

    # ── חיצים בין UI/UX ל-Main APP ─────────────────────────
    arrow(ax, 59, 76, 49, 76, text="Q — query / audio", dy=2.0)
    arrow(ax, 49, 62, 59, 62, text="result + confidence %", dy=-2.6)

    # ── DB ─────────────────────────────────────────────────
    cylinder(ax, 18, 26, 16, 22, C_DB)
    label(ax, 18, 24, "DB", fs=13, bold=True)
    # חיבור דו-כיווני ל-Main APP
    arrow(ax, 15, 49, 15, 38)          # APP → DB (write)
    arrow(ax, 21, 38, 21, 49)          # DB → APP (read)
    ax.text(8.5, 43.5, "read /\nwrite", fontsize=8, color="#566573",
            ha="center", va="center", style="italic")
    ax.text(27, 45, "JOIN", fontsize=8.5, color="#566573", ha="left", va="center")

    # תוכן ה-DB
    ax.annotate("", xy=(32, 33), xytext=(32, 19),
                arrowprops=dict(arrowstyle="-", lw=1.3, color="#7f8c8d"))
    for y, t in [(31, "history"), (26, "search"), (21, "records")]:
        ax.text(34, y, f"• {t}", fontsize=9, va="center")
    ax.text(34, 15.5, "stored sessions / IVR logs", fontsize=8,
            color="#7f8c8d", va="center", style="italic")

    # ── Confusion Matrix ───────────────────────────────────
    label(ax, 80, 40, "Confusion Matrix (eval)", fs=10, bold=True)
    conf_matrix(ax, 68, 14, 5.5, ["H", "I", "M", "R"])
    ax.text(96, 27, "H human\nI  ivr\nM music\nR recording",
            fontsize=8, va="center", ha="left", color="#566573")
    ax.text(90, 6.5, "predicted →", fontsize=8, color="#7f8c8d", ha="center")
    ax.text(64.5, 27, "actual\n↓", fontsize=8, color="#7f8c8d", ha="center", va="center")

    out_dir = Path(__file__).resolve().parent.parent / "docs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "system_architecture.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"נשמר: {out_path}")


if __name__ == "__main__":
    main()
