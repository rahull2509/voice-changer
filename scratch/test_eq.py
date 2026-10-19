import numpy as np
import scipy.signal as signal
import soundfile as sf
from pathlib import Path
from app.conversion.voice_reference import VoiceReferenceManager

def apply_target_warmth_eq(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Apply gentle acoustic EQ matching the WhatsApp voice warmth (slight low-mid boost, high-shelf softening)
    without ANY vocoder distortion.
    """
    # 1. Gentle low-mid warmth (peaking at 250Hz, Q=1.0, +2.5dB)
    # 2. Smooth gentle high-frequency softening (>4000Hz, -2.0dB) to match mobile voice message intimate warmth
    b, a = signal.iirpeak(250.0 / (sr / 2), 1.0)
    # Apply subtle mix
    warmth = signal.lfilter(b, a, audio)
    mixed = 0.85 * audio + 0.15 * warmth
    
    # Soften piercing highs (gentle Butterworth low-pass at 6500 Hz)
    b_lp, a_lp = signal.butter(2, 6500.0 / (sr / 2), btype='low')
    smoothed = signal.lfilter(b_lp, a_lp, mixed)
    
    # Normalize peak safely
    peak = np.max(np.abs(smoothed))
    if peak > 0.01:
        smoothed = smoothed * (0.85 / max(peak, 0.85))
    return smoothed.astype(np.float32)

if __name__ == "__main__":
    audio, sr = sf.read("cache/tts_cloned_output.wav")
    eq_audio = apply_target_warmth_eq(audio, sr)
    sf.write("cache/tts_cloned_realistic.wav", eq_audio, sr)
    print("Realistic EQ applied successfully. RMS:", np.sqrt(np.mean(eq_audio**2)))
