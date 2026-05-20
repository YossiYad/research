"""
חילוץ פיצ'רי נשימה מאודיו.

נשימה מאופיינת בקטעים עם אנרגיה נמוכה (לא דיבור) אבל לא שקט מוחלט,
עם spectral flatness גבוה (רעש רחב-פס, לא צליל טונלי).
"""
import numpy as np
import librosa

NUM_BREATH_FEATURES = 5


def extract_breath_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    מחלץ 5 פיצ'רי נשימה מקטע אודיו.

    Args:
        audio: מערך אודיו מונו.
        sr: קצב דגימה.

    Returns:
        np.ndarray בגודל (5,):
            [0] breath_count — מספר קטעי נשימה שזוהו
            [1] breath_ratio — חלק הפריימים שזוהו כנשימה
            [2] mean_breath_flatness — ממוצע spectral flatness בקטעי נשימה
            [3] breath_energy_contrast — יחס אנרגיה דיבור/נשימה
            [4] snr_db — יחס אות לרעש בדציבלים (אומדן מבוסס VAD)
    """
    hop_length = 512
    n_fft = 2048

    # חישוב אנרגיה (RMS) לכל פריים
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]

    # spectral flatness — מדד ל"רעשיות" (1 = רעש לבן, 0 = צליל טהור)
    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=n_fft, hop_length=hop_length,
    )[0]

    n_frames = min(len(rms), len(flatness))
    rms = rms[:n_frames]
    flatness = flatness[:n_frames]

    if n_frames == 0:
        return np.zeros(NUM_BREATH_FEATURES, dtype=np.float32)

    # סף שתיקה — פריימים עם אנרגיה אפסית
    silence_threshold = np.max(rms) * 0.02

    # סף אנרגיה — נשימה מתחת לחציון (לא דיבור/מוזיקה)
    energy_median = np.median(rms)

    # סף flatness — נשימה היא רעשית, flatness גבוה
    flatness_threshold = np.median(flatness)

    # פריים = נשימה אם: אנרגיה נמוכה + flatness גבוה + לא שתיקה
    breath_mask = (
        (rms < energy_median)
        & (flatness > flatness_threshold)
        & (rms > silence_threshold)
    )

    # ── פיצ'ר 1: ספירת קטעי נשימה (רצפים רצופים) ──
    breath_count = 0
    in_breath = False
    for is_breath in breath_mask:
        if is_breath and not in_breath:
            breath_count += 1
            in_breath = True
        elif not is_breath:
            in_breath = False

    # ── פיצ'ר 2: יחס פריימי נשימה ──
    breath_ratio = float(np.sum(breath_mask)) / n_frames

    # ── פיצ'ר 3: ממוצע spectral flatness בנשימות ──
    if np.any(breath_mask):
        mean_breath_flatness = float(np.mean(flatness[breath_mask]))
    else:
        mean_breath_flatness = 0.0

    # ── פיצ'ר 4: ניגודיות אנרגיה דיבור/נשימה ──
    non_breath_mask = ~breath_mask & (rms > silence_threshold)
    if np.any(breath_mask) and np.any(non_breath_mask):
        speech_energy = float(np.mean(rms[non_breath_mask]))
        breath_energy = float(np.mean(rms[breath_mask]))
        breath_energy_contrast = speech_energy / (breath_energy + 1e-7)
    else:
        breath_energy_contrast = 0.0

    # ── פיצ'ר 5: SNR (אומדן אות-לרעש) ──
    # הפריימים עם אנרגיה גבוהה = אות (דיבור/מוזיקה),
    # הפריימים עם אנרגיה נמוכה שאינם שתיקה = רעש רקע.
    # נציג אנושי במוקד → SNR בינוני (רעשי רקע),
    # הקלטה/IVR → SNR גבוה (אודיו נקי).
    signal_mask = rms >= energy_median
    noise_mask = (rms < energy_median) & (rms > silence_threshold)
    if np.any(signal_mask) and np.any(noise_mask):
        signal_power = float(np.mean(rms[signal_mask] ** 2))
        noise_power = float(np.mean(rms[noise_mask] ** 2))
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    else:
        snr_db = 0.0

    return np.array(
        [breath_count, breath_ratio, mean_breath_flatness, breath_energy_contrast, snr_db],
        dtype=np.float32,
    )
