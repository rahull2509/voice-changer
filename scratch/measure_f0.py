import numpy as np
import pyworld as pw
from app.conversion.voice_reference import VoiceReferenceManager

mgr = VoiceReferenceManager("voice-clone", "cache/voice_clone")

for name in ["scratch/swara_default.mp3", "scratch/swara_pitched_60hz.mp3", "scratch/swara_pitched_75hz.mp3"]:
    audio, sr = mgr.load_audio_file(name)
    audio_16k = mgr.resample_audio(audio, sr, 16000)
    f0, t = pw.dio(audio_16k.astype(np.float64), 16000)
    voiced = f0 > 40.0
    f0_med = float(np.median(f0[voiced]))
    print(f"{name}: F0 median = {f0_med:.1f} Hz")
