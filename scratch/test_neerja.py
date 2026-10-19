import asyncio
import edge_tts
import numpy as np
import pyworld as pw
from app.conversion.voice_reference import VoiceReferenceManager

async def test_neerja():
    text = "Hello! I am speaking with your cloned AI voice."
    mgr = VoiceReferenceManager("voice-clone", "cache/voice_clone")
    for offset in [0, -10, -20, -25]:
        pitch_str = f"{offset:+d}Hz" if offset != 0 else "+0Hz"
        c = edge_tts.Communicate(text, "en-IN-NeerjaNeural", pitch=pitch_str)
        tmp_file = f"scratch/test_neerja_{offset}.mp3"
        await c.save(tmp_file)
        audio, sr = mgr.load_audio_file(tmp_file)
        audio_16k = mgr.resample_audio(audio, sr, 16000)
        f0, t = pw.dio(audio_16k.astype(np.float64), 16000)
        voiced = f0 > 40.0
        f0_med = float(np.median(f0[voiced]))
        print(f"Neerja offset {pitch_str} -> F0 median = {f0_med:.1f} Hz (target=206.6 Hz)")

if __name__ == "__main__":
    asyncio.run(test_neerja())
