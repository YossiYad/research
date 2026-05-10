"""
Augmentations אקוסטיות לאימון.

כל טרנספורמציה מקבלת numpy array (float32, 16kHz) ומחזירה מערך באותו פורמט.
ה-pipeline מפעיל כל אוגמנטציה בהסתברות נתונה — רק על קבוצת האימון.
"""
import random

import librosa
import numpy as np


def add_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """מוסיף רעש גאוסי ב-SNR נתון."""
    rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-9
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, audio.shape).astype(np.float32)
    return audio + noise


def time_stretch(audio: np.ndarray, rate_range: tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """שינוי מהירות בלי לשנות pitch."""
    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio: np.ndarray, sr: int = 16000, semitone_range: tuple[float, float] = (-2, 2)) -> np.ndarray:
    """הזזת pitch בטווח של חצאי טונים."""
    n_steps = random.uniform(*semitone_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def random_gain(audio: np.ndarray, db_range: tuple[float, float] = (-6, 6)) -> np.ndarray:
    """שינוי עוצמה אקראי ב-dB."""
    gain_db = random.uniform(*db_range)
    return audio * (10 ** (gain_db / 20))


class AudioAugmentor:
    """
    מפעיל augmentations אקראיות על אודיו.

    Args:
        p: הסתברות להפעלת כל אוגמנטציה בנפרד.
        snr_range: טווח SNR לרעש (dB).
    """

    def __init__(self, p: float = 0.5, snr_range: tuple[float, float] = (10, 30)) -> None:
        self.p = p
        self.snr_range = snr_range

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            snr = random.uniform(*self.snr_range)
            audio = add_noise(audio, snr_db=snr)

        if random.random() < self.p:
            audio = time_stretch(audio)

        if random.random() < self.p:
            audio = pitch_shift(audio)

        if random.random() < self.p:
            audio = random_gain(audio)

        return audio
