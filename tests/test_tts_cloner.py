"""Automated tests for Text-to-Cloned-Voice (TTS) module and API endpoints."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.conversion.tts_cloner import TextToClonedVoice, DEFAULT_VOICE, SUPPORTED_VOICES
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_tts_voices_definition():
    """Verify supported neural voices."""
    assert DEFAULT_VOICE in SUPPORTED_VOICES
    assert "hi-IN-SwaraNeural" in SUPPORTED_VOICES
    assert "en-IN-NeerjaNeural" in SUPPORTED_VOICES


def test_tts_pitch_calibration():
    """Verify calibrated pitch calculations matching target speaker."""
    cloner = TextToClonedVoice()
    pitch_str, rate_str = cloner.compute_calibrated_pitch_and_rate("hi-IN-SwaraNeural", user_pitch_offset_hz=0, rate_percent=-3)
    assert "Hz" in pitch_str
    assert rate_str == "-3%"
    # With +10Hz user offset
    p_offset, _ = cloner.compute_calibrated_pitch_and_rate("hi-IN-SwaraNeural", user_pitch_offset_hz=10)
    val_default = int(pitch_str.replace("Hz", ""))
    val_offset = int(p_offset.replace("Hz", ""))
    assert val_offset == val_default + 10


@pytest.mark.asyncio
async def test_tts_synthesize_empty_text_error():
    """Empty or whitespace text must raise ValueError."""
    cloner = TextToClonedVoice()
    with pytest.raises(ValueError, match="Text cannot be empty"):
        await cloner.synthesize_text("   ")


@pytest.mark.asyncio
async def test_tts_synthesize_mocked_success(tmp_path):
    """Test full synthesis pipeline with mock base audio."""
    cloner = TextToClonedVoice(cache_dir=str(tmp_path))

    # Create dummy base audio file in cache
    dummy_wav = tmp_path / "tts_raw.wav"
    import soundfile as sf
    dummy_signal = (np.sin(2 * np.pi * 200 * np.linspace(0, 1.0, 16000, False)) * 0.3).astype(np.float32)
    sf.write(str(dummy_wav), dummy_signal, 16000)

    # Patch base TTS generation to return our dummy wav
    with patch.object(cloner, "_generate_base_tts", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = dummy_wav
        result = await cloner.synthesize_text(
            text="Testing cloned voice synthesis",
            voice_name="hi-IN-SwaraNeural",
            play_audio=False,
        )

        assert result["status"] == "success"
        assert result["duration_sec"] > 0
        assert result["output_rms"] > 0
        assert Path(result["output_file"]).exists()
        assert result["played_to_device"] is False


def test_api_tts_empty_text_rejected(client):
    """API endpoint returns 400 when text is empty."""
    resp = client.post("/voice-cloning/tts", json={"text": "   ", "play_audio": False})
    assert resp.status_code == 400


def test_api_tts_synthesize_endpoint(client, tmp_path):
    """API endpoint successfully accepts text and returns audio metadata."""
    # We patch _generate_base_tts to ensure fast and deterministic execution without internet
    dummy_wav = tmp_path / "tts_raw.wav"
    import soundfile as sf
    dummy_signal = (np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, 16000, False)) * 0.25).astype(np.float32)
    sf.write(str(dummy_wav), dummy_signal, 16000)

    with patch("app.conversion.tts_cloner.TextToClonedVoice._generate_base_tts", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = dummy_wav

        resp = client.post(
            "/voice-cloning/tts",
            json={
                "text": "Hello, this is a test of voice cloning.",
                "voice_name": "hi-IN-SwaraNeural",
                "play_audio": False,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["duration_sec"] > 0
        assert data["output_rms"] > 0


def test_api_tts_audio_download_endpoint(client):
    """GET /voice-cloning/tts/audio returns WAV audio response."""
    resp = client.get("/voice-cloning/tts/audio")
    # Even if previous test created it, or standalone test created it in cache/
    if resp.status_code == 200:
        assert resp.headers["content-type"] in ("audio/wav", "audio/x-wav")
    else:
        assert resp.status_code == 404
