"""
חיתוך קבצי אודיו לחתיכות של 5 שניות לאימון מסווג.

קלט: קובץ או תיקיה עם קבצי אודיו (mp3, wav, m4a, flac, ogg, וכו').
פלט: קבצי WAV של 5 שניות, 16 קילוהרץ, ערוץ אחד (מונו), 16 ביט PCM.

שימוש:
    python chop_audio.py --input <קובץ_או_תיקיה> --class <human|ivr|music|recording>

דוגמה:
    python chop_audio.py --input dataset/raw/human --class human
    python chop_audio.py --input dataset/raw/music --class music
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# ── הגדרות ────────────────────────────────────────────────────
TARGET_SR = 16000              # קצב דגימה: 16 קילוהרץ
CLIP_DURATION_SEC = 5.0        # אורך כל חתיכה ארוכה: 5 שניות
HOP_DURATION_SEC = 2.5         # קפיצה: 2.5 שניות (חפיפה של 50%)
MIN_DURATION_SEC = 0.3         # אורך מינימלי שיישמר (קצר מזה הוא רעש/קליק)
SILENCE_THRESHOLD_DB = -45     # סף שתיקה — חתיכות שקטות יותר ידולגו
VALID_CLASSES = ["human", "ivr", "music", "recording"]
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".webm", ".opus"}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def is_silent(clip: np.ndarray) -> bool:
    """בודק אם חתיכה שקטה לחלוטין (כדי לא לבזבז דוגמאות ריקות)."""
    rms = float(np.sqrt(np.mean(clip ** 2)))
    if rms == 0:
        return True
    db = 20 * np.log10(rms)
    return db < SILENCE_THRESHOLD_DB


def chop_file(input_path: Path, output_dir: Path, class_name: str) -> list[dict]:
    """
    חותך קובץ אודיו אחד לחתיכות של עד 5 שניות.
    מחזיר רשימת רשומות לכתיבה ל-metadata.csv.

    - קבצים ארוכים מ-5 שניות: חתוכים עם חפיפה, פלוס חתיכת זנב אם נשאר חומר לא מסווג.
    - קבצים קצרים מ-5 שניות (אבל >= MIN_DURATION_SEC): נשמרים כחתיכה בודדת באורך המקורי.
      (ה-collate_fn יבצע padding ב-batch; ה-attention_mask יבטיח שהמודל מתעלם מה-padding.)
    """
    logger.info(f"קורא: {input_path.name}")
    try:
        audio, _ = librosa.load(str(input_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        logger.warning(f"  ✗ נכשל לקרוא {input_path.name}: {e}")
        return []

    clip_samples = int(TARGET_SR * CLIP_DURATION_SEC)
    hop_samples = int(TARGET_SR * HOP_DURATION_SEC)
    min_samples = int(TARGET_SR * MIN_DURATION_SEC)

    if len(audio) < min_samples:
        duration = len(audio) / TARGET_SR
        logger.warning(f"  ✗ קצר מ-{MIN_DURATION_SEC} שניות, מדלג ({duration:.2f} שניות)")
        return []

    base_name = input_path.stem
    records: list[dict] = []
    saved = 0
    skipped_silent = 0

    # קצר מ-5 שניות: שומר את כל האודיו כחתיכה אחת
    if len(audio) < clip_samples:
        if is_silent(audio):
            logger.info(f"  ⏭ אודיו קצר ושקט, מדלג ({len(audio) / TARGET_SR:.2f} שניות)")
            return []
        out_name = f"{base_name}_0000.wav"
        out_path = output_dir / out_name
        sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
        records.append({
            "clip_id": out_name,
            "source_file": input_path.name,
            "start_sec": 0.0,
            "duration_sec": round(len(audio) / TARGET_SR, 3),
            "class": class_name,
        })
        logger.info(f"  ✓ נשמרה חתיכה קצרה ({len(audio) / TARGET_SR:.2f} שניות)")
        return records

    # אודיו ארוך: חיתוך עם חפיפה
    starts = list(range(0, len(audio) - clip_samples + 1, hop_samples))
    # הוספת חתיכת זנב אם יש זנב משמעותי (>= חצי hop) שנשאר לא מסווג
    last_end = starts[-1] + clip_samples
    if len(audio) - last_end >= hop_samples // 2:
        starts.append(len(audio) - clip_samples)

    chunk_idx = 0
    for start in starts:
        clip = audio[start:start + clip_samples]

        if is_silent(clip):
            chunk_idx += 1
            skipped_silent += 1
            continue

        out_name = f"{base_name}_{chunk_idx:04d}.wav"
        out_path = output_dir / out_name
        sf.write(str(out_path), clip, TARGET_SR, subtype="PCM_16")

        records.append({
            "clip_id": out_name,
            "source_file": input_path.name,
            "start_sec": round(start / TARGET_SR, 3),
            "duration_sec": CLIP_DURATION_SEC,
            "class": class_name,
        })
        chunk_idx += 1
        saved += 1

    logger.info(f"  ✓ נשמרו {saved} חתיכות (דולגו {skipped_silent} שקטות)")
    return records


def find_audio_files(input_path: Path) -> list[Path]:
    """מוצא את כל קבצי האודיו בתיקיה (כולל תת-תיקיות), או מחזיר את הקובץ הבודד."""
    if input_path.is_file():
        if input_path.suffix.lower() not in AUDIO_EXTS:
            logger.warning(f"אזהרה: סיומת לא מוכרת {input_path.suffix}")
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


FIELDNAMES = ["clip_id", "source_file", "start_sec", "duration_sec", "class"]


def append_metadata(metadata_path: Path, records: list[dict]) -> None:
    """מוסיף רשומות חדשות ל-metadata.csv (יוצר את הקובץ אם לא קיים)."""
    if not records:
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        write_header = False
    else:
        write_header = True
    with metadata_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="חיתוך אודיו לחתיכות של 5 שניות")
    parser.add_argument("--input", type=Path, required=True,
                        help="קובץ אודיו או תיקיה עם קבצי אודיו")
    parser.add_argument("--class", dest="class_name", choices=VALID_CLASSES, required=True,
                        help="הקטגוריה של האודיו")
    default_processed = Path(__file__).resolve().parent.parent / "dataset" / "processed"
    parser.add_argument("--output-root", type=Path, default=default_processed,
                        help="תיקית פלט (ברירת מחדל: dataset/processed)")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"שגיאה: הנתיב לא קיים: {args.input}")
        sys.exit(1)

    output_dir = args.output_root / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(args.input)
    if not files:
        logger.error(f"שגיאה: לא נמצאו קבצי אודיו ב-{args.input}")
        sys.exit(1)

    logger.info(f"נמצאו {len(files)} קבצי אודיו, מחתך לקטגוריה '{args.class_name}'")
    logger.info(f"פלט: {output_dir}")
    logger.info("")

    all_records: list[dict] = []
    for f in files:
        all_records.extend(chop_file(f, output_dir, args.class_name))

    metadata_path = args.output_root.parent / "metadata.csv"
    append_metadata(metadata_path, all_records)

    logger.info("")
    logger.info(f"סיכום: {len(all_records)} חתיכות נשמרו לקטגוריה '{args.class_name}'")
    logger.info(f"מטה-דאטה: {metadata_path}")


if __name__ == "__main__":
    main()
