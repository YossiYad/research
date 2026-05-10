"""
מחלקות ופונקציות לטעינת מאגר הנתונים לאימון.

חשוב: הפיצול ל-train/val/test נעשה לפי **קובץ מקור** (source_file),
לא לפי חתיכה (clip). אחרת, חתיכות שנחתכו מאותה הקלטה היו עלולות
להתפזר בין train ל-test, והמודל היה מקבל ציון בדיקה מנופח שלא משקף
את הביצועים האמיתיים על הקלטות חדשות.
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES: list[str] = ["human", "ivr", "music", "recording"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}


def load_metadata(metadata_path: Path) -> list[dict]:
    """קורא את metadata.csv לרשימת רשומות."""
    with metadata_path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def split_by_source(
    metadata: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    מחלק את המאגר ל-train/val/test לפי קובץ מקור.

    כל החתיכות שנחתכו מאותו קובץ מקור ילכו לאותו פיצול. זה מונע
    דליפת מידע (data leakage) — מצב שבו המודל "מכיר" את ההקלטה
    מאימון וזוכה לציון מנופח על אותה הקלטה ב-test.

    Returns:
        (train_records, val_records, test_records)
    """
    rng = random.Random(seed)

    # קיבוץ לפי (קטגוריה, קובץ מקור)
    by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in metadata:
        by_source[(rec["class"], rec["source_file"])].append(rec)

    # קיבוץ של קובצי מקור לפי קטגוריה
    sources_by_class: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for (cls, src), recs in by_source.items():
        sources_by_class[cls].append((src, recs))

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for cls in CLASSES:
        sources = sources_by_class.get(cls, [])
        rng.shuffle(sources)
        n = len(sources)
        if n == 0:
            continue
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        for i, (_, recs) in enumerate(sources):
            if i < n_test:
                test.extend(recs)
            elif i < n_test + n_val:
                val.extend(recs)
            else:
                train.extend(recs)

    return train, val, test


class AudioDataset(Dataset):
    """
    מאגר אודיו ל-PyTorch.

    כל פריט: (waveform, label_idx) — צמד של טנזור אודיו ותווית מספרית.
    """

    def __init__(
        self,
        records: list[dict],
        processed_dir: Path,
        sample_rate: int = 16000,
        normalize: bool = True,
    ) -> None:
        self.records = records
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        wav_path = self.processed_dir / rec["class"] / rec["clip_id"]
        audio, _ = librosa.load(str(wav_path), sr=self.sample_rate, mono=True)

        # נורמליזציה (חובה ל-wav2vec2 — מצפה לקלט עם ממוצע 0 וסטיית תקן 1)
        if self.normalize:
            audio = (audio - audio.mean()) / (audio.std() + 1e-7)

        return torch.from_numpy(audio).float(), CLASS_TO_IDX[rec["class"]]


def class_distribution(records: list[dict]) -> dict[str, int]:
    """מחזיר ספירה של דוגמאות לכל קטגוריה."""
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r["class"]] += 1
    return dict(counts)
