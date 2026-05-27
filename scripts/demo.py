"""
הדגמה חיה (Gradio) — מסווג קטע אודיו ומציג את 4 ההסתברויות.

מקליטים מהמיקרופון או מעלים קובץ → המודל מסווג ל-human / ivr / music / recording
ומציג גרף עמודות של ההסתברויות. נותן קישור ציבורי (share=True) להדגמה במצגת.

הרצה (אחרי קלון/משיכה של הריפו ב-Colab):
    pip install gradio
    python scripts/demo.py --checkpoint <נתיב ל-best.pt>

דוגמה:
    python scripts/demo.py \
        --checkpoint /content/drive/MyDrive/holdie-audio/checkpoints/run-XXXX/best.pt

חשוב: מספר הפיצ'רים בקוד חייב להתאים ל-checkpoint. ה-best.pt של הריצה הטובה
(20 פיצ'רים) דורש את גרסת הקוד עם 20 הפיצ'רים. אם אימנת עם --no-aux-features,
הרץ כאן עם --no-aux-features.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASSES
from features import NUM_FEATURES
from model import AudioClassifier
from inference import load_and_chunk, classify_chunks

try:
    import gradio as gr
except ImportError:
    print("שגיאה: נדרש gradio — הרץ:  pip install gradio")
    sys.exit(1)

# תוויות תצוגה בעברית
LABELS_HE = {
    "human": "נציג אנושי 🧑",
    "ivr": "תפריט IVR 🤖",
    "music": "מוזיקת המתנה 🎵",
    "recording": "הקלטה / תא קולי 📼",
}


def build_demo(checkpoint: Path, base_model: str, use_aux: bool) -> "gr.Blocks":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"מכשיר: {device}")
    print(f"טוען מודל בסיס: {base_model}")
    model = AudioClassifier(
        base_model=base_model,
        num_classes=len(CLASSES),
        aux_features_dim=NUM_FEATURES if use_aux else 0,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"נטען checkpoint: {checkpoint}")

    def predict(audio_path):
        if not audio_path:
            return {}
        chunks = load_and_chunk(Path(audio_path))
        result = classify_chunks(model, chunks, device)
        # מיפוי לעברית עבור תצוגת gr.Label
        return {LABELS_HE.get(c, c): float(p) for c, p in result["probabilities"].items()}

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Audio(sources=["upload", "microphone"], type="filepath",
                        label="🎤 הקלט או העלה קטע אודיו"),
        outputs=gr.Label(num_top_classes=4, label="סיווג המודל"),
        title="Holdie — מסווג אודיו של שיחות טלפון",
        description="הקלט/העלה קטע אודיו, והמודל יסווג אותו לאחת מארבע קטגוריות "
                    "ויציג את רמת הביטחון לכל אחת.",
    )
    return demo


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="הדגמת Gradio למסווג האודיו")
    parser.add_argument("--checkpoint", type=Path,
                        default=project_root / "checkpoints" / "best.pt",
                        help="נתיב ל-best.pt")
    parser.add_argument("--base-model", default="facebook/wav2vec2-xls-r-300m",
                        help="מודל הבסיס (חייב להתאים ל-checkpoint)")
    parser.add_argument("--no-aux-features", action="store_true",
                        help="אם המודל אומן בלי פיצ'רים אקוסטיים")
    parser.add_argument("--no-share", action="store_true",
                        help="בלי קישור ציבורי (רק מקומי)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"שגיאה: לא נמצא checkpoint ב-{args.checkpoint}")
        sys.exit(1)

    demo = build_demo(args.checkpoint, args.base_model, not args.no_aux_features)
    demo.launch(share=not args.no_share)


if __name__ == "__main__":
    main()
