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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    CLASSES,
    AudioDataset,
    class_distribution,
    collate_fn,
    load_metadata,
    split_by_source,
)
from augment import AudioAugmentor
from features import NUM_BREATH_FEATURES
from model import AudioClassifier


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float, list[int], list[int]]:
    """מריץ הערכה על מאגר נתונים. מחזיר (loss, accuracy, macro_f1, תוויות, חיזויים)."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for audio, mask, breath_feats, labels in loader:
            audio = audio.to(device)
            mask = mask.to(device)
            breath_feats = breath_feats.to(device)
            labels = labels.to(device)
            logits = model(audio, attention_mask=mask, aux_features=breath_feats)
            total_loss += criterion(logits, labels).item()
            n_batches += 1
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    correct = sum(1 for t, p in zip(all_labels, all_preds) if t == p)
    accuracy = correct / max(1, len(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, accuracy, macro_f1, all_labels, all_preds


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
    parser.add_argument("--base-model", default="facebook/wav2vec2-xls-r-300m",
                        help="שם המודל המאומן מראש מ-HuggingFace")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="קצב למידה לראש הסיווג")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                        help="להקפיא את האנקודר (ברירת מחדל: כן)")
    parser.add_argument("--no-freeze-encoder", dest="freeze_encoder",
                        action="store_false")
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="מספר שכבות encoder עליונות לשחרור (0 = הכל קפוא)")
    parser.add_argument("--encoder-lr", type=float, default=1e-5,
                        help="קצב למידה לשכבות encoder משוחררות (נמוך מ-lr)")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="מספר אפוקי warmup לינארי לפני cosine annealing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5,
                        help="מספר אפוקים ללא שיפור ב-macro F1 לפני עצירה מוקדמת")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="מספר תהליכי טעינת נתונים")
    parser.add_argument("--no-aux-features", action="store_true",
                        help="להשבית פיצ'רי נשימה (לדיבוג — מאמן רק על wav2vec2)")
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
    aux_dim = 0 if args.no_aux_features else NUM_BREATH_FEATURES
    if args.no_aux_features:
        print("פיצ'רי נשימה מושבתים (--no-aux-features)")
    model = AudioClassifier(
        base_model=args.base_model,
        num_classes=len(CLASSES),
        freeze_encoder=args.freeze_encoder,
        unfreeze_layers=args.unfreeze_layers,
        aux_features_dim=aux_dim,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"פרמטרים מתאמנים: {trainable:,} מתוך {total:,}")

    # קבוצות פרמטרים — LR נמוך יותר לשכבות encoder משוחררות
    param_groups = []
    if args.unfreeze_layers > 0:
        encoder_params = [
            p for layer in model.encoder.encoder.layers[-args.unfreeze_layers:]
            for p in layer.parameters() if p.requires_grad
        ]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.encoder_lr})
            print(f"שכבות encoder משוחררות: {args.unfreeze_layers} (lr={args.encoder_lr})")

    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("encoder.encoder.layers")
    ]
    param_groups.append({"params": head_params, "lr": args.lr})

    optimizer = AdamW(param_groups, weight_decay=0.01)

    # warmup לינארי + cosine annealing
    warmup = args.warmup_epochs
    cosine_epochs = max(1, args.epochs - warmup)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup])

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
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    history: list[dict] = []

    # ── לולאת אימון ─────────────────────────────────────────
    print("\nמתחיל אימון...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for audio, mask, breath_feats, labels in train_loader:
            audio = audio.to(device)
            mask = mask.to(device)
            breath_feats = breath_feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(audio, attention_mask=mask, aux_features=breath_feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss = total_loss / max(1, n_batches)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"אפוק {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | "
              f"val_f1={val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output_dir / "best.pt")
            print(f"    ✓ נשמר best.pt (macro F1 חדש: {val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\n⏹ עצירה מוקדמת — אין שיפור ב-macro F1 כבר {args.patience} אפוקים.")
                break

    torch.save(model.state_dict(), args.output_dir / "last.pt")
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

    # ── בדיקה סופית ────────────────────────────────────────
    print(f"\nטוען best.pt לבדיקה (macro F1 אימות: {best_val_f1:.4f})")
    model.load_state_dict(torch.load(args.output_dir / "best.pt"))
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    print(f"\n{'=' * 50}")
    print(f"דיוק על קבוצת הבדיקה: {test_acc:.4f}")
    print(f"Macro F1 על קבוצת הבדיקה: {test_f1:.4f}")
    print(f"{'=' * 50}\n")

    print("דוח סיווג:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0))

    print("מטריצת בלבול (שורות = אמת, עמודות = חיזוי):")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    header = "          " + "  ".join(f"{c:>10}" for c in CLASSES)
    print(header)
    for cls, row in zip(CLASSES, cm):
        print(f"{cls:>10}  " + "  ".join(f"{v:>10}" for v in row))



if __name__ == "__main__":
    main()
