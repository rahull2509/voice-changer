import asyncio
import edge_tts
import numpy as np
import scipy.signal as signal
import soundfile as sf
from app.conversion.voice_reference import VoiceReferenceManager

async def test_realistic_pipeline():
    text = "नमस्ते! मैं आपकी क्लोन आवाज में बोल रही हूँ। अब यह आवाज बिल्कुल रियलिस्टिक और नैचुरल लगेगी, बिना किसी रोबोटिक या कार्टून इफ़ेक्ट के।"
    
    mgr = VoiceReferenceManager("voice-clone", "cache/voice_clone")
    try:
        profile = mgr.load_cached_profile()
        target_f0 = profile.f0_median
    except Exception:
        target_f0 = 206.6

    # Calibrate offset for Swara (baseline ~247Hz -> target ~206Hz)
    offset_hz = int(round(target_f0 - 247.0 + 15)) # -25Hz gives 206.1Hz
    pitch_str = f"{offset_hz:+d}Hz"
    rate_str = "-3%"
    
    print(f"Generating neural speech with pitch={pitch_str}, rate={rate_str}...")
    communicate = edge_tts.Communicate(text, "hi-IN-SwaraNeural", pitch=pitch_str, rate=rate_str)
    out_mp3 = "scratch/swara_demo.mp3"
    await communicate.save(out_mp3)

    raw_audio, sr = mgr.load_audio_file(out_mp3)
    audio_16k = mgr.resample_audio(raw_audio, sr, 16000)

    # Apply gentle acoustic warmth
    b, a = signal.iirpeak(250.0 / (16000 / 2), 1.2)
    warmth = signal.lfilter(b, a, audio_16k)
    mixed = 0.82 * audio_16k + 0.18 * warmth
    
    b_lp, a_lp = signal.butter(2, 6800.0 / (16000 / 2), btype='low')
    final_audio = signal.lfilter(b_lp, a_lp, mixed).astype(np.float32)
    
    # Normalize safely
    peak = np.max(np.abs(final_audio))
    if peak > 0.01:
        final_audio = (final_audio * (0.90 / max(peak, 0.90))).astype(np.float32)

    sf.write("cache/tts_cloned_output.wav", final_audio, 16000, subtype="PCM_16")
    print("Successfully generated cache/tts_cloned_output.wav!")
    print(f"Duration: {len(final_audio)/16000:.2f}s, RMS: {np.sqrt(np.mean(final_audio**2)):.4f}")

if __name__ == "__main__":
    asyncio.run(test_realistic_pipeline())
