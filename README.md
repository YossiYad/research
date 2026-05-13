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
│   │   │   ├── he/           # תת-תיקייה לכל שפה (קוד ISO 639-1)
│   │   │   ├── en/
│   │   │   └── ar/
│   │   ├── ivr/
│   │   │   ├── he/
│   │   │   └── en/
│   │   ├── music/
│   │   │   └── n_a/          # למוזיקה אין שפה
│   │   └── recording/
│   │       ├── he/
│   │       └── en/
│   ├── processed/            # חתיכות של 5 שניות, מוכנות לאימון
│   │   ├── human/
│   │   ├── ivr/
│   │   ├── music/
│   │   └── recording/
│   └── metadata.csv          # רשימה של כל החתיכות, מקור, וקטגוריה+שפה
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

## תהליך עבודה

### שלב 1 — הוספת אודיו גולמי

מעתיקים קבצי אודיו (mp3, wav, m4a, וכו') לתת-התיקייה של הקטגוריה **והשפה** ב-`dataset/raw/`. למשל:

- שיחה אמיתית של נציג בעברית → `dataset/raw/human/he/`
- שיחה של נציג באנגלית → `dataset/raw/human/en/`
- תפריט IVR בעברית → `dataset/raw/ivr/he/`
- מוזיקת המתנה → `dataset/raw/music/n_a/` (שפה לא רלוונטית)
- תא קולי בעברית → `dataset/raw/recording/he/`

קודי השפה הם ISO 639-1 (`he`, `en`, `ar`, `ru`, ...). למוזיקה השתמשו ב-`n_a` בשם התיקייה (סלאש אסור בשם תיקייה — הסקריפט מתרגם אוטומטית ל-`n/a` ב-CSV).

### שלב 2 — חיתוך לחתיכות של 5 שניות

הריצו פעם נפרדת לכל צמד (קטגוריה, שפה):

```bash
python scripts/chop_audio.py --input dataset/raw/human/he --class human --language he
python scripts/chop_audio.py --input dataset/raw/human/en --class human --language en
python scripts/chop_audio.py --input dataset/raw/ivr/he --class ivr --language he
python scripts/chop_audio.py --input dataset/raw/music/n_a --class music --language n/a
python scripts/chop_audio.py --input dataset/raw/recording/he --class recording --language he
```

הסקריפט יחתוך כל קובץ לחתיכות של 5 שניות עם חפיפה של 2.5 שניות, ידלג על חתיכות שקטות, ויעדכן את `metadata.csv` עם עמודת `language` לכל חתיכה.

ב-Colab מבצעים את כל זה אוטומטית — ראו `notebooks/train_colab.ipynb`.

### שלב 3 — אימון

#### האפשרות המומלצת: Google Colab (GPU בחינם)

1. פתחו את [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) ב-Colab.
2. החליפו ל-GPU runtime: `Runtime → Change runtime type → GPU` (T4 חינמי).
3. העלו את האודיו ל-`MyDrive/holdie-audio/raw/{class}/{language}/` (ראו "מבנה תיקיות" למעלה).
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
  - `facebook/wav2vec2-xls-r-300m` — רב-לשוני (53 שפות), **מומלץ לאימון רב-לשוני**.
  - `imvladikon/wav2vec2-xls-r-300m-hebrew` — מותאם עברית; טוב אם רוב המאגר עברית.
- `--freeze-encoder` (ברירת מחדל: מופעל) — מאמן רק את ראש הסיווג. במאגר גדול אפשר לעבור ל-`--no-freeze-encoder`.
- `--val-ratio` / `--test-ratio` — שיעור הקלטות (לא חתיכות) ב-val/test, פר-שפה.

#### על הפיצול הרב-לשוני

הפיצול ל-train/val/test מתבצע **לפי הקלטת מקור** (כל החתיכות מאותה הקלטה הולכות לאותו פיצול — אחרת המודל היה מקבל ציון מנופח) ו**עם stratification כפול** על הצמד (קטגוריה, שפה). זה אומר שכל שפה מקבלת ייצוג ב-train, ב-val וב-test — גם אם יש לה רק כמה הקלטות.

אזהרה: אם ל-(קטגוריה, שפה) מסוימים יש פחות מ-3 הקלטות מקור, כולן יילכו ל-train ולא תהיה הערכה על השפה הזו עבור הקטגוריה הזו. ה-train.py יציג אזהרה בעת ההרצה.

בסיום האימון, מודפס דיוק בדיקה **פר שפה** — כך אפשר לזהות הטיה (למשל, אם המודל טוב בעברית אבל גרוע באנגלית).

### שלב 4 — שילוב ב-Holdie

יבוא בהמשך — שירות inference שיוטמע בתוך `packages/worker`.

## הערות

- קבצי האודיו ב-`dataset/` לא נכנסים ל-git (הם גדולים מדי וגם פרטיים). ראה `.gitignore`.
- כדאי לתחזק גיבוי חיצוני של המאגר (Google Drive או דיסק חיצוני).
