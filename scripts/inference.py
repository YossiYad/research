"""
סקריפט inference — מסווג קובץ אודיו חדש.

שימוש:
    python scripts/inference.py path/to/audio.wav
    python scripts/inference.py path/to/audio.mp3 --checkpoint checkpoints/best.pt
    python scripts/inference.py path/to/folder/ --base-model facebook/wav2vec2-xls-r-300m
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASSES, IDX_TO_CLASS
from features import extract_features, NUM_FEATURES
from model import AudioClassifier


SAMPLE_RATE = 16000
CLIP_DURATION = 5.0


def load_and_chunk(audio_path: Path) -> list[np.ndarray]:
    """טוען קובץ אודיו וחותך לחתיכות של 5 שניות."""
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    clip_samples = int(CLIP_DURATION * SAMPLE_RATE)

    if len(audio) <= clip_samples:
        return [audio]

    # חתיכות עם hop של 2.5 שניות (כמו ב-chop_audio.py)
    hop_samples = clip_samples // 2
    chunks = []
    for start in range(0, len(audio) - clip_samples + 1, hop_samples):
        chunks.append(audio[start : start + clip_samples])
    return chunks


def classify_chunks(
    model: AudioClassifier,
    chunks: list[np.ndarray],
    device: str,
) -> dict:
    """מסווג חתיכות אודיו ומחזיר תוצאות מצטברות."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for chunk in chunks:
            # חילוץ פיצ'רים אקוסטיים (לפני נורמליזציה)
            feats = extract_features(chunk, sr=SAMPLE_RATE)
            feats_tensor = torch.from_numpy(feats).float().unsqueeze(0).to(device)

            # נורמליזציה (כמו ב-dataset.py)
            audio = (chunk - chunk.mean()) / (chunk.std() + 1e-7)
            waveform = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            mask = torch.ones(1, waveform.shape[1], dtype=torch.long, device=device)

            logits = model(waveform, attention_mask=mask, aux_features=feats_tensor)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])

    # ממוצע הסתברויות על כל החתיכות
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = int(np.argmax(avg_probs))

    return {
        "prediction": IDX_TO_CLASS[pred_idx],
        "confidence": float(avg_probs[pred_idx]),
        "probabilities": {cls: float(avg_probs[i]) for i, cls in enumerate(CLASSES)},
        "num_chunks": len(chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="סיווג קובץ אודיו")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("input", type=Path,
                        help="קובץ אודיו או תיקייה לסיווג")
    parser.add_argument("--checkpoint", type=Path,
                        default=project_root / "checkpoints" / "best.pt",
                        help="נתיב ל-checkpoint")
    parser.add_argument("--base-model", default="facebook/wav2vec2-xls-r-300m",
                        help="שם המודל המאומן מראש (חייב להתאים ל-checkpoint)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"שגיאה: לא נמצא checkpoint ב-{args.checkpoint}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AudioClassifier(
        base_model=args.base_model,
        num_classes=len(CLASSES),
        aux_features_dim=NUM_FEATURES,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"נטען checkpoint: {args.checkpoint}")

    # איסוף קבצים לסיווג
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".ogg", ".aac", ".webm", ".opus"}
    if args.input.is_dir():
        files = sorted(f for f in args.input.iterdir() if f.suffix.lower() in audio_exts)
    elif args.input.is_file():
        files = [args.input]
    else:
        print(f"שגיאה: לא נמצא {args.input}")
        sys.exit(1)

    if not files:
        print("לא נמצאו קבצי אודיו.")
        sys.exit(1)

    # סיווג
    for audio_path in files:
        chunks = load_and_chunk(audio_path)
        result = classify_chunks(model, chunks, device)

        print(f"\n{'─' * 50}")
        print(f"קובץ: {audio_path.name}")
        print(f"תחזית: {result['prediction']}  (ביטחון: {result['confidence']:.1%})")
        print(f"חתיכות: {result['num_chunks']}")
        print("הסתברויות:")
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {cls:>12}  {prob:6.1%}  {bar}")


if __name__ == "__main__":
    main()
