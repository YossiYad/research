"""
השוואת ריצות אימון.

סורק תיקייה שמכילה תיקיות run-* (כל אחת עם run_config.json שנשמר ע"י
train.py), ובונה טבלת השוואה של ההגדרות והתוצאות של כל הריצות.

שימוש:
    # ב-Colab (תיקיות ה-run ב-Drive):
    python scripts/compare_runs.py --runs-dir /content/drive/MyDrive/holdie-audio/checkpoints

    # מקומית (ברירת מחדל: תיקיית checkpoints של הפרויקט):
    python scripts/compare_runs.py

פלט:
    - טבלה מודפסת למסך, ממוינת לפי macro F1 על הבדיקה (מהטוב לפחות).
    - קובץ comparison.csv בתוך runs-dir (לטעינה לדו"ח/אקסל).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

CLASSES = ["human", "ivr", "music", "recording"]


def find_run_configs(runs_dir: Path) -> list[Path]:
    """מוצא את כל קבצי run_config.json תחת runs-dir (כולל בתוך תיקיות run-*)."""
    configs = sorted(runs_dir.glob("**/run_config.json"))
    return configs


def load_run(config_path: Path) -> dict | None:
    """קורא run_config.json ומשטח אותו לשורה אחת לטבלה."""
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠ דילוג על {config_path}: {e}", file=sys.stderr)
        return None

    cfg = data.get("config", {})
    res = data.get("results", {})
    per_class = res.get("per_class", {})

    # שם הריצה = שם התיקייה המכילה (למשל run-20260525-1030), או '.' אם בשורש
    run_name = config_path.parent.name or config_path.parent.resolve().name

    row = {
        "run": run_name,
        "timestamp": data.get("timestamp", ""),
        "model": cfg.get("base_model", "").split("/")[-1],
        "lr": cfg.get("lr", ""),
        "unfreeze": cfg.get("unfreeze_layers", ""),
        "aux": "yes" if cfg.get("aux_features") else "no",
        "epochs": res.get("epochs_trained", ""),
        "test_acc": res.get("test_accuracy", ""),
        "macro_f1": res.get("test_macro_f1", ""),
    }
    for c in CLASSES:
        row[f"f1_{c}"] = per_class.get(c, {}).get("f1", "")
    return row


def _fmt(v) -> str:
    """עיצוב ערך לתא בטבלה (.4g שומר על דיוק גם ל-lr קטן וגם למטריקות)."""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def print_table(rows: list[dict]) -> None:
    """מדפיס טבלה מיושרת למסך."""
    cols = ["run", "model", "lr", "unfreeze", "aux", "epochs",
            "test_acc", "macro_f1"] + [f"f1_{c}" for c in CLASSES]
    headers = {
        "run": "run", "model": "model", "lr": "lr", "unfreeze": "unfrz",
        "aux": "aux", "epochs": "ep", "test_acc": "test_acc", "macro_f1": "macro_f1",
        "f1_human": "human", "f1_ivr": "ivr", "f1_music": "music", "f1_recording": "rec",
    }
    widths = {c: len(headers[c]) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(_fmt(row.get(c, ""))))

    line = "  ".join(headers[c].ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(_fmt(row.get(c, "")).ljust(widths[c]) for c in cols))


def write_csv(rows: list[dict], out_path: Path) -> None:
    """כותב את הטבלה ל-CSV."""
    cols = ["run", "timestamp", "model", "lr", "unfreeze", "aux", "epochs",
            "test_acc", "macro_f1"] + [f"f1_{c}" for c in CLASSES]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="השוואת ריצות אימון")
    parser.add_argument("--runs-dir", type=Path,
                        default=project_root / "checkpoints",
                        help="תיקייה שמכילה תיקיות run-* (או run_config.json בודד)")
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"שגיאה: לא נמצאה התיקייה {args.runs_dir}")
        sys.exit(1)

    configs = find_run_configs(args.runs_dir)
    if not configs:
        print(f"לא נמצאו קבצי run_config.json תחת {args.runs_dir}")
        print("(ודא שהרצת אימון אחרי העדכון ששומר run_config.json)")
        sys.exit(1)

    rows = [r for r in (load_run(c) for c in configs) if r is not None]

    # מיון לפי macro F1 (יורד); ריצות בלי תוצאה בסוף
    rows.sort(key=lambda r: r["macro_f1"] if isinstance(r["macro_f1"], (int, float)) else -1,
              reverse=True)

    print(f"\nנמצאו {len(rows)} ריצות תחת {args.runs_dir}\n")
    print_table(rows)

    out_path = args.runs_dir / "comparison.csv"
    write_csv(rows, out_path)
    print(f"\n✓ נשמרה טבלת השוואה: {out_path}")


if __name__ == "__main__":
    main()
