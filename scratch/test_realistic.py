import asyncio
import edge_tts
import numpy as np
import pyworld as pw
import soundfile as sf
from app.conversion.voice_reference import VoiceReferenceManager

async def test_realistic_synthesis():
    text = "नमस्ते, आप कैसे हैं? सब ठीक है? मैं आपकी क्लोन आवाज में बोल रही हूँ।"
    
    # Target profile has f0_median = 206.6 Hz
    # Swara default is ~247 Hz. Offset = 206.6 - 247 = -40 Hz
    c = edge_tts.Communicate(text, "hi-IN-SwaraNeural", pitch="-40Hz", rate="-3%")
    out_mp3 = "scratch/swara_calibrated_realistic.mp3"
    await c.save(out_mp3)

    mgr = VoiceReferenceManager("voice-clone", "cache/voice_clone")
    audio, sr = mgr.load_audio_file(out_mp3)
    audio_16k = mgr.resample_audio(audio, sr, 16000)

    f0, t = pw.dio(audio_16k.astype(np.float64), 16000)
    voiced = f0 > 40.0
    f0_med = float(np.median(f0[voiced]))
    print(f"Calibrated Neural Speech F0 median: {f0_med:.1f} Hz (Target is 206.6 Hz!)")

    # Save 16k wav
    sf.write("cache/tts_cloned_output.wav", audio_16k, 16000, subtype="PCM_16")
    print("Saved cache/tts_cloned_output.wav")

if __name__ == "__main__":
    asyncio.run(test_realistic_synthesis())
