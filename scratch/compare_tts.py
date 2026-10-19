import numpy as np
import pyworld as pw
import soundfile as sf
from pathlib import Path

raw_mp3 = Path("cache/tts_raw.mp3")
cloned_wav = Path("cache/tts_cloned_output.wav")

if raw_mp3.exists():
    from app.conversion.voice_reference import VoiceReferenceManager
    mgr = VoiceReferenceManager("voice-clone", "cache/voice_clone")
    raw_audio, sr = mgr.load_audio_file(raw_mp3)
    raw_16k = mgr.resample_audio(raw_audio, sr, 16000)
    
    # Analyze raw edge-tts
    f0, t = pw.dio(raw_16k.astype(np.float64), 16000)
    voiced = f0 > 40.0
    print(f"RAW edge-tts: duration={len(raw_16k)/16000:.2f}s, F0 median={np.median(f0[voiced]):.1f} Hz")

if cloned_wav.exists():
    cloned_audio, sr = sf.read(str(cloned_wav))
    f0_c, t = pw.dio(cloned_audio.astype(np.float64), 16000)
    voiced_c = f0_c > 40.0
    print(f"CLONED output: duration={len(cloned_audio)/16000:.2f}s, F0 median={np.median(f0_c[voiced_c]):.1f} Hz")
