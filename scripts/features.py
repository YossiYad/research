"""
חילוץ פיצ'רים אקוסטיים מותאמים-קטגוריה.

כל פיצ'ר מכוון להבחנה של קטגוריה אחת או יותר מתוך
{human, ivr, music, recording}, ומבוסס על אלגוריתם מקובל
(librosa / numpy) ולא על heuristic שרירותי.

הרציונל: הפיצ'רים מספקים "ראייה משלימה" ל-wav2vec2 — מאפיינים
גלובליים (tempo, אנרגיית DTMF, רוחב-פס) שמודל מאומן על דיבור
לא בהכרח תופס. הם מחושבים על האודיו המקורי (לפני augmentation)
ומחוברים לפלט ה-attention pooling לפני שכבת הסיווג.

הפיצ'רים מנורמלים לטווח דומה (~[0, 2]) כדי שלא ידחקו את הפיצ'רים
של wav2vec2 בשכבת הסיווג (בלי נירמול, פיצ'ר בקנה מידה גדול כמו
תדר ב-Hz גורם לגרדיאנטים גדולים פי כמה ושובר את האופטימיזציה).

רשימת הפיצ'רים (15):
    music:     tempo_bpm, beat_strength, harmonic_ratio, chroma_std
    ivr:       dtmf_energy, pitch_stability, silence_ratio, pitch_jitter, amp_shimmer
    human:     pitch_range, breath_events, voiced_ratio
    recording: bandwidth_hz, rolloff_hz, noise_floor_db,
               noise_stationarity, spectral_tilt, dynamic_range_db
    כללי:      zcr_mean, spectral_centroid_hz
"""
import warnings

import numpy as np
import librosa

FEATURE_NAMES = [
    "tempo_bpm",            # music — קצב (BPM)
    "beat_strength",        # music — חוזק onset
    "harmonic_ratio",       # music — יחס אנרגיה הרמונית
    "chroma_std",           # music — פיזור chroma (מגוון צלילים)
    "dtmf_energy",          # ivr   — אנרגיית צלילי חיוג (DTMF)
    "pitch_stability",      # ivr   — יציבות pitch (קול סינתטי)
    "silence_ratio",        # ivr   — יחס פריימים שקטים
    "pitch_jitter",         # ivr   — הפרעת F0 מחזורית (סינתטי = נמוך)
    "amp_shimmer",          # ivr   — הפרעת עוצמה מחזורית (סינתטי = נמוך)
    "pitch_range",          # human — טווח pitch דינמי
    "breath_events",        # human — ספירת אירועי נשימה
    "voiced_ratio",         # human — יחס פריימים קוליים
    "bandwidth_hz",         # recording — רוחב פס אפקטיבי (rolloff 99%)
    "rolloff_hz",           # recording — תדר גלגלת 85%
    "noise_floor_db",       # recording — רצפת רעש (dB)
    "noise_stationarity",   # recording — יציבות רעש הרקע (ערוץ)
    "spectral_tilt",        # recording — שיפוע ספקטרלי (חתימת codec)
    "dynamic_range_db",     # recording — טווח דינמי (reverb/דחיסה)
    "zcr_mean",             # כללי — קצב חציית אפס
    "spectral_centroid_hz", # כללי — מרכז כובד ספקטרלי
]
NUM_FEATURES = len(FEATURE_NAMES)

# שמות תאימות לאחור (קוד ישן שמייבא את השמות הללו)
NUM_BREATH_FEATURES = NUM_FEATURES

# סקלות נירמול — חלוקה לכל פיצ'ר לפי הטווח הפיזי שלו, להבאתו ל-~[0, 2].
_NORM_SCALES = np.array([
    100.0,    # tempo_bpm
    3.0,      # beat_strength
    1.0,      # harmonic_ratio (0-1)
    0.2,      # chroma_std
    0.5,      # dtmf_energy (max ratio לפריים)
    1.0,      # pitch_stability (0-1)
    1.0,      # silence_ratio (0-1)
    0.05,     # pitch_jitter (הפרעה יחסית)
    0.1,      # amp_shimmer (הפרעה יחסית)
    150.0,    # pitch_range (Hz)
    5.0,      # breath_events
    1.0,      # voiced_ratio (0-1)
    4000.0,   # bandwidth_hz
    4000.0,   # rolloff_hz
    40.0,     # noise_floor_db (שלילי)
    1.0,      # noise_stationarity (CV)
    5.0,      # spectral_tilt (dB/kHz)
    30.0,     # dynamic_range_db
    0.2,      # zcr_mean
    2000.0,   # spectral_centroid_hz
], dtype=np.float32)

# זוגות תדרי DTMF (צלילי חיוג טלפוני)
DTMF_FREQS = [697, 770, 852, 941, 1209, 1336, 1477, 1633]

_MIN_SAMPLES = 2048  # מתחת לזה אין מספיק פריימים לחישוב משמעותי


def extract_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    מחלץ 15 פיצ'רים אקוסטיים מקטע אודיו ומחזיר וקטור מנורמל בגודל (15,).

    Args:
        audio: מערך אודיו מונו (float).
        sr: קצב דגימה.

    Returns:
        np.ndarray בגודל (15,) — ראו FEATURE_NAMES לסדר.
    """
    audio = np.asarray(audio, dtype=np.float32)

    # אודיו קצר מדי או שקט לחלוטין — מחזירים אפסים
    if len(audio) < _MIN_SAMPLES or float(np.max(np.abs(audio))) < 1e-6:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    hop = 512
    n_fft = 2048

    # ── חישובים משותפים ──────────────────────────────────────
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))  # (freq, time)
    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # ── music ────────────────────────────────────────────────
    try:
        tempo = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop)[0]
        tempo_bpm = float(np.atleast_1d(tempo)[0])
    except Exception:
        tempo_bpm = 0.0

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    beat_strength = float(np.mean(onset_env)) if onset_env.size else 0.0

    harmonic = librosa.effects.harmonic(audio)
    harmonic_ratio = float(np.sum(harmonic ** 2) / (np.sum(audio ** 2) + 1e-9))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # chroma מתלונן על קליפים קצרים/חסרי-pitch
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop)
    chroma_std = float(np.mean(np.std(chroma, axis=1)))

    # ── ivr ──────────────────────────────────────────────────
    # אנרגיית DTMF — היחס המקסימלי לאורך הזמן בין 8 תדרי החיוג לכלל הספקטרום.
    # max (ולא ממוצע) כי צלילי חיוג קצרים בזמן ולא רוצים שהם "יידוללו".
    dtmf_idx = [int(np.argmin(np.abs(freqs - f))) for f in DTMF_FREQS]
    frame_total = np.sum(S, axis=0) + 1e-9
    dtmf_per_frame = np.sum(S[dtmf_idx, :], axis=0) / frame_total
    dtmf_energy = float(np.max(dtmf_per_frame)) if dtmf_per_frame.size else 0.0

    # pitch — מבוסס pyin (מזהה גם פריימים לא-קוליים, מחזיר NaN)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=65, fmax=500, sr=sr, hop_length=hop,
        )
    except Exception:
        f0 = np.array([np.nan])
        voiced_flag = np.array([False])

    f0_valid = f0[np.isfinite(f0) & (f0 > 0)]
    if f0_valid.size > 1:
        # יציבות = הופכי לסטיית התקן (קול סינתטי של IVR יציב מאוד)
        pitch_stability = float(1.0 / (1.0 + np.std(f0_valid)))
        # טווח = אחוזון 95 פחות 5 (דיבור אנושי דינמי)
        pitch_range = float(np.percentile(f0_valid, 95) - np.percentile(f0_valid, 5))
    else:
        pitch_stability = 0.0
        pitch_range = 0.0

    silence_thr = float(np.max(rms)) * 0.1
    silence_ratio = float(np.mean(rms < silence_thr))

    # jitter — הפרעת F0 בין פריימים קוליים. קול סינתטי (IVR) "חלק" → jitter נמוך;
    # קול אנושי טבעי → jitter גבוה. מבדיל בין IVR ל-human (ששניהם דיבור).
    if f0_valid.size > 2:
        pitch_jitter = float(np.mean(np.abs(np.diff(f0_valid))) / (np.mean(f0_valid) + 1e-9))
    else:
        pitch_jitter = 0.0

    # shimmer — הפרעת עוצמה בין פריימים קוליים. אותו רציונל כמו jitter.
    nv = min(len(rms), len(voiced_flag))
    vmask = np.asarray(voiced_flag[:nv], dtype=bool) if nv > 0 else np.array([], dtype=bool)
    voiced_amp = rms[:nv][vmask] if vmask.size else np.array([])
    if voiced_amp.size > 2:
        amp_shimmer = float(np.mean(np.abs(np.diff(voiced_amp))) / (np.mean(voiced_amp) + 1e-9))
    else:
        amp_shimmer = 0.0

    # ── human ────────────────────────────────────────────────
    # אירועי נשימה — רצפים של אנרגיה נמוכה + flatness גבוה (רעש רחב-פס)
    flatness = librosa.feature.spectral_flatness(S=S, n_fft=n_fft, hop_length=hop)[0]
    n = min(len(rms), len(flatness))
    breath_mask = (
        (rms[:n] < np.median(rms))
        & (flatness[:n] > 0.3)
        & (rms[:n] > silence_thr)
    )
    breath_events = 0
    prev = False
    for is_breath in breath_mask:
        if is_breath and not prev:
            breath_events += 1
        prev = is_breath
    breath_events = float(breath_events)

    # יחס פריימים קוליים (דיבור אנושי/IVR > מוזיקה/רעש)
    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag.size else 0.0

    # ── recording ────────────────────────────────────────────
    # הקלטות/תא קולי לרוב דחוסות (codec) — אין תכולה בתדרים גבוהים
    rolloff_99 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.99)[0]
    bandwidth_hz = float(np.mean(rolloff_99))
    rolloff_85 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    rolloff_hz = float(np.mean(rolloff_85))
    noise_floor = float(np.percentile(rms, 10))
    noise_floor_db = float(20 * np.log10(noise_floor + 1e-7))

    # יציבות רעש רקע — סטיית תקן יחסית בפריימים החלשים. הקלטה/תא-קולי:
    # רעש קבוע (CV נמוך); שיחה חיה: רעש משתנה (CV גבוה).
    noise_thr = float(np.percentile(rms, 30))
    noise_frames = rms[rms <= noise_thr]
    if noise_frames.size > 2:
        noise_stationarity = float(np.std(noise_frames) / (np.mean(noise_frames) + 1e-9))
    else:
        noise_stationarity = 0.0

    # שיפוע ספקטרלי (dB/kHz) — חתימת ערוץ/codec. דחיסת קו-טלפון → tilt תלול.
    spectrum_db = librosa.amplitude_to_db(np.mean(S, axis=1) + 1e-9)
    spectral_tilt = float(np.polyfit(freqs / 1000.0, spectrum_db, 1)[0])

    # טווח דינמי (dB) — יחס פריימים חזקים/חלשים. שיחה חיה: טווח רחב (שתיקות עמוקות);
    # הקלטה/reverb: טווח צר (הד/רעש ממלאים את השתיקות).
    loud = float(np.percentile(rms, 95))
    quiet = float(np.percentile(rms, 5))
    dynamic_range_db = float(20 * np.log10((loud + 1e-9) / (quiet + 1e-9)))

    # ── כללי ─────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop)[0]
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spectral_centroid_hz = float(np.mean(centroid)) if centroid.size else 0.0

    features = np.array([
        tempo_bpm, beat_strength, harmonic_ratio, chroma_std,
        dtmf_energy, pitch_stability, silence_ratio, pitch_jitter, amp_shimmer,
        pitch_range, breath_events, voiced_ratio,
        bandwidth_hz, rolloff_hz, noise_floor_db,
        noise_stationarity, spectral_tilt, dynamic_range_db,
        zcr_mean, spectral_centroid_hz,
    ], dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = features / _NORM_SCALES
    # חיתוך לטווח שפוי — מגן מפני קלט פתולוגי (למשל רעש לבן טהור)
    # שיכול לדחוף פיצ'ר בודד לערך קיצוני שידחק את שאר הקלטים.
    return np.clip(features, -5.0, 5.0).astype(np.float32)


# כינוי לאחור-תאימות
extract_breath_features = extract_features
