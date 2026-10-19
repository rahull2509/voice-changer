import asyncio
import edge_tts
import numpy as np
import pyworld as pw
from app.conversion.voice_reference import VoiceReferenceManager

async def find_exact_pitch():
    text = "नमस्ते, आप कैसे हैं? सब ठीक है?"
    mgr = VoiceReferenceManager("voice-clone", "cache/voice_clone")
    for offset in [-15, -20, -25, -30]:
        c = edge_tts.Communicate(text, "hi-IN-SwaraNeural", pitch=f"{offset}Hz")
        tmp_file = f"scratch/test_{offset}.mp3"
        await c.save(tmp_file)
        audio, sr = mgr.load_audio_file(tmp_file)
        audio_16k = mgr.resample_audio(audio, sr, 16000)
        f0, t = pw.dio(audio_16k.astype(np.float64), 16000)
        voiced = f0 > 40.0
        f0_med = float(np.median(f0[voiced]))
        print(f"Offset {offset}Hz -> F0 median = {f0_med:.1f} Hz (target=206.6 Hz)")

if __name__ == "__main__":
    asyncio.run(find_exact_pitch())
