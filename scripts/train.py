"""
סקריפט אימון מסווג אודיו.

מיועד להרצה ב-Google Colab עם GPU של NVIDIA.

שימוש (מהשורש של הפרויקט):
    python scripts/train.py

או עם פרמטרים מותאמים:
    python scripts/train.py --base-model facebook/wav2vec2-xls-r-300m --epochs 30 --batch-size 4

הפלט:
    - checkpoints/best.pt — המודל הטוב ביותר על קבוצת האימות.
    - checkpoints/last.pt — המודל מהאפוק האחרון.
    - הדפסת ציוני בדיקה (test) ומטריצת בלבול בסוף.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    CLASSES,
    AudioDataset,
    class_distribution,
    class_language_distribution,
    collate_fn,
    language_distribution,
    load_metadata,
    split_by_source,
)
from augment import AudioAugmentor
from model import AudioClassifier


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[float, list[int], list[int]]:
    """מריץ הערכה על מאגר נתונים. מחזיר (דיוק, תוויות אמיתיות, חיזויים)."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for audio, mask, labels in loader:
            audio = audio.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            logits = model(audio, attention_mask=mask)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    correct = sum(1 for t, p in zip(all_labels, all_preds) if t == p)
    accuracy = correct / max(1, len(all_labels))
    return accuracy, all_labels, all_preds


def main() -> None:
    parser = argparse.ArgumentParser(description="אימון מסווג אודיו")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--data-dir", type=Path,
                        default=project_root / "dataset" / "processed",
                        help="תיקיית האודיו המעובד")
    parser.add_argument("--metadata", type=Path,
                        default=project_root / "dataset" / "metadata.csv",
                        help="קובץ metadata.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=project_root / "checkpoints",
                        help="תיקיית פלט לשמירת המודלים")
    parser.add_argument("--base-model", default="facebook/wav2vec2-base",
                        help="שם המודל המאומן מראש מ-HuggingFace")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="קצב למידה")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                        help="להקפיא את האנקודר (ברירת מחדל: כן)")
    parser.add_argument("--no-freeze-encoder", dest="freeze_encoder",
                        action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5,
                        help="מספר אפוקים ללא שיפור לפני עצירה מוקדמת")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="מספר תהליכי טעינת נתונים")
    args = parser.parse_args()

    # ── הגדרות ראשוניות ────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"מכשיר: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not args.metadata.exists():
        print(f"שגיאה: לא נמצא {args.metadata}")
        print("ודא שהרצת את chop_audio.py על כל הקטגוריות.")
        sys.exit(1)

    # ── טעינת מאגר ופיצול ──────────────────────────────────────
    metadata = load_metadata(args.metadata)
    print(f"\nסך הכל חתיכות במאגר: {len(metadata)}")
    print("התפלגות:", class_distribution(metadata))

    train_recs, val_recs, test_recs = split_by_source(
        metadata,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"\nאימון: {len(train_recs)} | אימות: {len(val_recs)} | בדיקה: {len(test_recs)}")
    print("התפלגות אימון:", class_distribution(train_recs))
    print("התפלגות אימות:", class_distribution(val_recs))
    print("התפלגות בדיקה:", class_distribution(test_recs))

    all_langs = language_distribution(metadata)
    if len(all_langs) > 1 or next(iter(all_langs), "unknown") != "unknown":
        print(f"\nשפות במאגר: {all_langs}")
        print("התפלגות שפה באימון:", language_distribution(train_recs))
        print("התפלגות שפה באימות:", language_distribution(val_recs))
        print("התפלגות שפה בבדיקה:", language_distribution(test_recs))
        print("פירוט (קטגוריה, שפה) בבדיקה:", class_language_distribution(test_recs))

    if not train_recs or not val_recs or not test_recs:
        print("\nשגיאה: אחד הפיצולים ריק. צריך לפחות 3 קובצי מקור לכל קטגוריה.")
        sys.exit(1)

    augmentor = AudioAugmentor(p=0.5)
    train_ds = AudioDataset(train_recs, args.data_dir, augmentor=augmentor)
    val_ds = AudioDataset(val_recs, args.data_dir)
    test_ds = AudioDataset(test_recs, args.data_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"),
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"),
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"),
                             collate_fn=collate_fn)

    # ── בניית המודל ──────────────────────────────────────────
    print(f"\nטוען מודל בסיס: {args.base_model}")
    model = AudioClassifier(
        base_model=args.base_model,
        num_classes=len(CLASSES),
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"פרמטרים מתאמנים: {trainable:,} מתוך {total:,}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # class weights — מפצה על חוסר איזון בין הקטגוריות
    train_dist = class_distribution(train_recs)
    weights = torch.tensor(
        [1.0 / max(1, train_dist.get(c, 1)) for c in CLASSES],
        dtype=torch.float,
    )
    weights = weights / weights.sum() * len(CLASSES)  # נרמול כך שהממוצע = 1
    print(f"\nclass weights: {dict(zip(CLASSES, weights.tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history: list[dict] = []

    # ── לולאת אימון ─────────────────────────────────────────
    print("\nמתחיל אימון...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for audio, mask, labels in train_loader:
            audio = audio.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(audio, attention_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss = total_loss / max(1, n_batches)
        val_acc, _, _ = evaluate(model, val_loader, device)

        print(f"אפוק {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output_dir / "best.pt")
            print(f"    ✓ נשמר best.pt (דיוק חדש: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\n⏹ עצירה מוקדמת — אין שיפור כבר {args.patience} אפוקים.")
                break

    torch.save(model.state_dict(), args.output_dir / "last.pt")
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

    # ── בדיקה סופית ────────────────────────────────────────
    print(f"\nטוען best.pt לבדיקה (דיוק אימות: {best_val_acc:.4f})")
    model.load_state_dict(torch.load(args.output_dir / "best.pt"))
    test_acc, y_true, y_pred = evaluate(model, test_loader, device)

    print(f"\n{'=' * 50}")
    print(f"דיוק על קבוצת הבדיקה: {test_acc:.4f}")
    print(f"{'=' * 50}\n")

    print("דוח סיווג:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0))

    print("מטריצת בלבול (שורות = אמת, עמודות = חיזוי):")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    header = "          " + "  ".join(f"{c:>10}" for c in CLASSES)
    print(header)
    for cls, row in zip(CLASSES, cm):
        print(f"{cls:>10}  " + "  ".join(f"{v:>10}" for v in row))

    # ── דיוק פר-שפה ─────────────────────────────────────────
    # ה-test_loader רץ בלי shuffle, ולכן y_pred[i] תואם את test_recs[i].
    test_langs = sorted(language_distribution(test_recs))
    if len(test_langs) > 1 or test_langs != ["unknown"]:
        per_lang_acc: dict[str, tuple[int, int]] = {}
        for rec, t, p in zip(test_recs, y_true, y_pred):
            lang = rec.get("language", "unknown") or "unknown"
            correct, total = per_lang_acc.get(lang, (0, 0))
            per_lang_acc[lang] = (correct + int(t == p), total + 1)
        print("\nדיוק על קבוצת הבדיקה פר שפה:")
        for lang in sorted(per_lang_acc):
            correct, total = per_lang_acc[lang]
            print(f"  {lang:>8}: {correct}/{total} = {correct / max(1, total):.4f}")


if __name__ == "__main__":
    main()
