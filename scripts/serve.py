"""
שרת web להדגמת המסווג — מגיש דף HTML מעוצב ומריץ את המודל על קובץ שמעלים.

הדף (web/index.html) שולח את האודיו ל-/predict, השרת מריץ את המודל ומחזיר JSON.
מאחר שאותו שרת מגיש גם את הדף וגם את ה-API — אין בעיות CORS.

הרצה עצמאית:
    pip install fastapi uvicorn python-multipart
    python scripts/serve.py --checkpoint <נתיב ל-best.pt>
    # ואז גלישה ל-http://localhost:8000

ב-Colab (לקבלת כתובת לחיצה) — ראו את ה-build_app() שאפשר להריץ בתהליך נפרד
עם uvicorn ולחשוף את הפורט דרך google.colab proxyPort. הוראות בצ'אט.

חשוב: מספר הפיצ'רים בקוד חייב להתאים ל-checkpoint (ריצה 9 = 20 פיצ'רים).
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASSES
from features import NUM_FEATURES
from model import AudioClassifier
from inference import load_and_chunk, classify_chunks

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:
    print("שגיאה: נדרשים fastapi ו-python-multipart — הרץ:")
    print("    pip install fastapi uvicorn python-multipart")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = PROJECT_ROOT / "web" / "index.html"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".ogg", ".aac", ".webm", ".opus"}


def build_app(checkpoint: Path, base_model: str = "facebook/wav2vec2-xls-r-300m",
              use_aux: bool = True) -> "FastAPI":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"מכשיר: {device} | טוען מודל בסיס: {base_model}")
    model = AudioClassifier(
        base_model=base_model,
        num_classes=len(CLASSES),
        aux_features_dim=NUM_FEATURES if use_aux else 0,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"נטען checkpoint: {checkpoint}")

    app = FastAPI(title="Holdie Audio Classifier")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        suffix = Path(file.filename or "audio.wav").suffix.lower()
        if suffix not in AUDIO_EXTS:
            suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            chunks = load_and_chunk(tmp_path)
            result = classify_chunks(model, chunks, device)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            tmp_path.unlink(missing_ok=True)
        return JSONResponse(result)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="שרת הדגמה למסווג האודיו")
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--base-model", default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--no-aux-features", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"שגיאה: לא נמצא checkpoint ב-{args.checkpoint}")
        sys.exit(1)

    import uvicorn
    app = build_app(args.checkpoint, args.base_model, not args.no_aux_features)
    print(f"\nהשרת רץ על http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
