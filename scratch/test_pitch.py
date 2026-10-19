import asyncio
import edge_tts
import numpy as np
import pyworld as pw
from pathlib import Path

async def test_edge_tts_pitch():
    text = "नमस्ते, आप कैसे हैं? सब ठीक है?"
    
    # Test 1: Default
    c1 = edge_tts.Communicate(text, "hi-IN-SwaraNeural")
    await c1.save("scratch/swara_default.mp3")

    # Test 2: Pitch adjusted by Hz
    c2 = edge_tts.Communicate(text, "hi-IN-SwaraNeural", pitch="-60Hz", rate="-4%")
    await c2.save("scratch/swara_pitched_60hz.mp3")

    c3 = edge_tts.Communicate(text, "hi-IN-SwaraNeural", pitch="-75Hz", rate="-4%")
    await c3.save("scratch/swara_pitched_75hz.mp3")

    print("Synthesized pitch test samples successfully!")

if __name__ == "__main__":
    asyncio.run(test_edge_tts_pitch())
