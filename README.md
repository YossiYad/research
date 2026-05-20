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
│   └── metadata.csv          # רשימה של כל החתיכות, מקור וקטגוריה
├── scripts/
│   ├── chop_audio.py         # חיתוך אודיו ארוך לחתיכות של 5 שניות
│   ├── dataset.py            # טעינה ופיצול train/val/test (stratified)
│   ├── augment.py            # augmentations אקוסטיות
│   ├── model.py              # ארכיטקטורת המסווג (wav2vec2 + attention pooling)
│   ├── train.py              # סקריפט אימון
│   └── visualize.py          # גרפים ומטריצת בלבול
├── notebooks/
│   └── train_colab.ipynb     # אימון בענן (Google Colab עם GPU)
├── requirements.txt
└── README.md
```

**הערה לגבי שפות:** אודיו IVR עשוי להכיל משפטים בשפות שונות (עברית, אנגלית, רוסית וכו'). המודל מסווג לפי מאפיינים אקוסטיים (קול סינתטי, קצב, טון) ולא לפי תוכן שפתי, ולכן אין צורך למיין את ההקלטות לפי שפה.

## תהליך עבודה

### שלב 1 — הוספת אודיו גולמי

מעתיקים קבצי אודיו (mp3, wav, m4a, וכו') לתת-התיקייה של הקטגוריה ב-`dataset/raw/`. למשל:

- שיחה של נציג אנושי → `dataset/raw/human/`
- תפריט IVR → `dataset/raw/ivr/`
- מוזיקת המתנה → `dataset/raw/music/`
- תא קולי / הקלטה → `dataset/raw/recording/`

### שלב 2 — חיתוך לחתיכות של 5 שניות

```bash
python scripts/chop_audio.py --input dataset/raw/human --class human
python scripts/chop_audio.py --input dataset/raw/ivr --class ivr
python scripts/chop_audio.py --input dataset/raw/music --class music
python scripts/chop_audio.py --input dataset/raw/recording --class recording
```

הסקריפט יחתוך כל קובץ לחתיכות של 5 שניות עם חפיפה של 2.5 שניות, ידלג על חתיכות שקטות, ויעדכן את `metadata.csv`.

ב-Colab מבצעים את כל זה אוטומטית — ראו `notebooks/train_colab.ipynb`.

### שלב 3 — אימון

#### האפשרות המומלצת: Google Colab (GPU בחינם)

1. פתחו את [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) ב-Colab.
2. החליפו ל-GPU runtime: `Runtime → Change runtime type → GPU` (T4 חינמי).
3. העלו את האודיו ל-`MyDrive/holdie-audio/raw/{class}/` (ראו "מבנה תיקיות" למעלה).
4. הריצו את התאים לפי הסדר. המחברת תקלון את הריפו, תתקין תלויות, תחתוך את האודיו, תאמן ותשמור את המודל ל-Drive.

#### אימון מקומי (אם יש GPU)

```bash
python scripts/train.py \
  --base-model facebook/wav2vec2-xls-r-300m \
  --epochs 20 --batch-size 8 --lr 1e-4
```

הפרמטרים החשובים:
- `--base-model` — מודל בסיס מ-HuggingFace. אפשרויות:
  - `facebook/wav2vec2-base` — קל ומהיר, אומן על אנגלית. מתאים למאגרים קטנים או לבדיקות.
  - `facebook/wav2vec2-xls-r-300m` — רב-לשוני (53 שפות), 300M פרמטרים.
  - `imvladikon/wav2vec2-xls-r-300m-hebrew` — מותאם עברית, **מומלץ** כשרוב המאגר בעברית.
- `--freeze-encoder` (ברירת מחדל: מופעל) — מאמן רק את ראש הסיווג. במאגר גדול אפשר לעבור ל-`--no-freeze-encoder`.
- `--val-ratio` / `--test-ratio` — שיעור הקלטות (לא חתיכות) ב-val/test.

#### על הפיצול

הפיצול ל-train/val/test מתבצע **לפי הקלטת מקור** (כל החתיכות מאותה הקלטה הולכות לאותו פיצול — אחרת המודל היה מקבל ציון מנופח) ו**עם stratification** על הקטגוריה.

אזהרה: אם לקטגוריה מסוימת יש פחות מ-3 הקלטות מקור, כולן יילכו ל-train ולא תהיה הערכה עבור הקטגוריה הזו. ה-train.py יציג אזהרה בעת ההרצה.

### שלב 4 — שילוב ב-Holdie

יבוא בהמשך — שירות inference שיוטמע בתוך `packages/worker`.

## הערות

- קבצי האודיו ב-`dataset/` לא נכנסים ל-git (הם גדולים מדי וגם פרטיים). ראה `.gitignore`.
- כדאי לתחזק גיבוי חיצוני של המאגר (Google Drive או דיסק חיצוני).
