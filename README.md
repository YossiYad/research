# מסווג אודיו — Holdie

מחקר לאימון מודל למידת מכונה שמסווג קטעי אודיו לאחת מארבע קטגוריות:

1. **human** — נציג אנושי חי
2. **ivr** — תפריט אינטראקטיבי ("הקש 1 לעברית")
3. **music** — מוזיקת המתנה
4. **recording** — הקלטה / תא קולי / הודעה מוקלטת

## פרטים טכניים

- **אורך כל חתיכת אודיו:** 5 שניות
- **קצב דגימה:** 16 קילוהרץ
- **ערוצים:** 1 (מונו)
- **פורמט:** WAV עם דגימה של 16 ביט (PCM)
- **מודל מתוכנן:** wav2vec2 בגרסה העברית של ivrit-ai (כיוון עדין)

## דרישות התקנה

צריך Python 3.10 ומעלה, ו-ffmpeg מותקן במערכת.

```bash
# מהתיקיה הזו
pip install -r requirements.txt
```

ל-Windows, אפשר להתקין ffmpeg עם:

```powershell
winget install Gyan.FFmpeg
```

## מבנה תיקיות

```
audio-classifier/
├── dataset/
│   ├── raw/                  # קבצי אודיו גולמיים, לפני חיתוך
│   │   ├── human/
│   │   ├── ivr/
│   │   ├── music/
│   │   └── recording/
│   ├── processed/            # חתיכות של 5 שניות, מוכנות לאימון
│   │   ├── human/
│   │   ├── ivr/
│   │   ├── music/
│   │   └── recording/
│   └── metadata.csv          # רשימה של כל החתיכות והמקור שלהן
├── scripts/
│   └── chop_audio.py         # חיתוך אודיו ארוך לחתיכות של 5 שניות
├── requirements.txt
└── README.md
```

## תהליך עבודה

### שלב 1 — הוספת אודיו גולמי

מעתיקים קבצי אודיו (mp3, wav, m4a, וכו') לתיקייה המתאימה ב-`dataset/raw/`. למשל:

- שיחה אמיתית של נציג → `dataset/raw/human/`
- הקלטה של תפריט IVR → `dataset/raw/ivr/`
- הורדת מוזיקת המתנה מיוטיוב → `dataset/raw/music/`
- הקלטה של תא קולי → `dataset/raw/recording/`

### שלב 2 — חיתוך לחתיכות של 5 שניות

```bash
python scripts/chop_audio.py --input dataset/raw/human --class human
python scripts/chop_audio.py --input dataset/raw/ivr --class ivr
python scripts/chop_audio.py --input dataset/raw/music --class music
python scripts/chop_audio.py --input dataset/raw/recording --class recording
```

הסקריפט יחתוך כל קובץ לחתיכות של 5 שניות עם חפיפה של 2.5 שניות, ידלג על חתיכות שקטות, ויעדכן את `metadata.csv`.

### שלב 3 — אימון

יבוא בהמשך — סקריפט אימון ב-Google Colab.

### שלב 4 — שילוב ב-Holdie

יבוא בהמשך — שירות inference שיוטמע בתוך `packages/worker`.

## הערות

- קבצי האודיו ב-`dataset/` לא נכנסים ל-git (הם גדולים מדי וגם פרטיים). ראה `.gitignore`.
- כדאי לתחזק גיבוי חיצוני של המאגר (Google Drive או דיסק חיצוני).
