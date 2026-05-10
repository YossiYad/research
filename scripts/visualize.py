"""
יצירת גרפים וויזואליזציות מתוצאות האימון — להגשה אקדמית.

שימוש (אחרי אימון):
    python scripts/visualize.py

פלט: תמונות PNG בתיקיית checkpoints/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASSES, AudioDataset, collate_fn, load_metadata, split_by_source
from model import AudioClassifier

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
except ImportError:
    print("שגיאה: נדרש matplotlib — pip install matplotlib")
    sys.exit(1)


# ── הגדרת פונט שתומך בעברית (אם קיים) ────────────────────
rcParams["font.family"] = "DejaVu Sans"
rcParams["figure.dpi"] = 150


def plot_training_curves(history: list[dict], output_dir: Path) -> None:
    """גרף loss ו-accuracy לאורך האימון."""
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    accs = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, losses, "o-", color="#e74c3c", linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, accs, "o-", color="#2ecc71", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  training_curves.png")


def plot_confusion_matrix(y_true: list[int], y_pred: list[int], output_dir: Path) -> None:
    """מטריצת בלבול ויזואלית."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  confusion_matrix.png")


def plot_class_distribution(train_recs, val_recs, test_recs, output_dir: Path) -> None:
    """התפלגות הקטגוריות בכל פיצול."""
    from collections import Counter

    splits = {"Train": train_recs, "Val": val_recs, "Test": test_recs}
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(CLASSES))
    width = 0.25
    colors = ["#3498db", "#f39c12", "#e74c3c"]

    for i, (name, recs) in enumerate(splits.items()):
        counts = Counter(r["class"] for r in recs)
        values = [counts.get(c, 0) for c in CLASSES]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(v), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Samples")
    ax.set_title("Class Distribution by Split")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "class_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  class_distribution.png")


def plot_per_class_metrics(y_true: list[int], y_pred: list[int], output_dir: Path) -> None:
    """גרף precision / recall / f1 לכל קטגוריה."""
    report = classification_report(y_true, y_pred, target_names=CLASSES,
                                   output_dict=True, zero_division=0)

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(CLASSES))
    width = 0.25
    colors = ["#1abc9c", "#9b59b6", "#e67e22"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, metric in enumerate(metrics):
        values = [report[c][metric] for c in CLASSES]
        ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics (Test Set)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "per_class_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  per_class_metrics.png")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "checkpoints"
    history_path = output_dir / "history.json"
    model_path = output_dir / "best.pt"
    metadata_path = project_root / "dataset" / "metadata.csv"
    data_dir = project_root / "dataset" / "processed"

    # ── גרף אימון ──────────────────────────────────────────
    print("מייצר גרפים:\n")

    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as fh:
            history = json.load(fh)
        plot_training_curves(history, output_dir)
    else:
        print("  ⚠ לא נמצא history.json — מדלג על גרפי אימון")

    # ── התפלגות + מטריצת בלבול ─────────────────────────────
    if not metadata_path.exists():
        print("  ⚠ לא נמצא metadata.csv — מדלג")
        return

    metadata = load_metadata(metadata_path)
    train_recs, val_recs, test_recs = split_by_source(metadata)
    plot_class_distribution(train_recs, val_recs, test_recs, output_dir)

    if not model_path.exists():
        print("  ⚠ לא נמצא best.pt — מדלג על מטריצת בלבול")
        return

    # ── הרצת המודל על קבוצת הבדיקה ─────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioClassifier(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = AudioDataset(test_recs, data_dir)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=8, collate_fn=collate_fn)

    y_true, y_pred = [], []
    with torch.no_grad():
        for audio, mask, labels in test_loader:
            audio = audio.to(device)
            mask = mask.to(device)
            preds = model(audio, attention_mask=mask).argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_per_class_metrics(y_true, y_pred, output_dir)

    print(f"\nכל הגרפים נשמרו ב: {output_dir}/")


if __name__ == "__main__":
    main()
