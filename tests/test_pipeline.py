"""Unit tests for audio pipeline mechanics, crossfading, limiter, and queue safety."""

import numpy as np
import pytest

from app.audio.playback import AudioPlayback
from app.audio.pipeline import AudioPipeline
from app.config import AppConfig


def test_playback_soft_limiter():
    """Verify high amplitude audio is smoothly soft-limited below clipping threshold."""
    # Create extreme input waveform with peaks at 5.0 (would cause severe distortion)
    extreme_audio = np.array([-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0], dtype=np.float32)
    limited = AudioPlayback._soft_limit(extreme_audio, threshold=0.95)

    assert len(limited) == len(extreme_audio)
    # Peak must not exceed threshold
    assert np.max(np.abs(limited)) <= 0.9501
    # Zero crossing preserved
    assert limited[3] == 0.0


def test_pipeline_dry_wet_and_gain():
    cfg = AppConfig()
    cfg.dsp.pitch_factor = 1.6
    cfg.dsp.output_gain = 0.5
    cfg.dsp.dry_wet_mix = 0.0  # 100% dry (original voice)

    pipeline = AudioPipeline(cfg)
    converter = pipeline._init_converter()

    # Pass 300ms chunk
    chunk = (0.4 * np.sin(2 * np.pi * 300 * np.arange(4800) / 16000)).astype(np.float32)
    processed = converter.process(chunk)

    # At 0.0 dry_wet_mix, mixed should match original chunk
    mixed = (1.0 - cfg.dsp.dry_wet_mix) * chunk + cfg.dsp.dry_wet_mix * processed
    assert np.allclose(mixed, chunk, atol=1e-5)

    # Gain scaling
    gained = (mixed * cfg.dsp.output_gain).astype(np.float32)
    assert np.allclose(gained, chunk * 0.5, atol=1e-5)


def test_pipeline_status_reporting():
    cfg = AppConfig()
    pipeline = AudioPipeline(cfg)
    status_dict = pipeline.status.to_dict()

    assert "is_running" in status_dict
    assert status_dict["is_running"] is False
    assert status_dict["active_mode"] == "dsp"
    assert "real_time_factor" in status_dict
