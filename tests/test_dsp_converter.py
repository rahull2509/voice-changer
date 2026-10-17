"""Unit tests for DSP voice conversion (WORLD vocoder and fallback)."""

import numpy as np
import pytest

from app.config import DSPConfig
from app.conversion.dsp_converter import DSPConverter, WORLD_AVAILABLE


def test_dsp_converter_initialization():
    cfg = DSPConfig(engine="world", pitch_factor=1.6, formant_shift=1.15)
    conv = DSPConverter(cfg, sample_rate=16000)
    assert conv.name.startswith("DSP")
    assert conv.sample_rate == 16000


def test_dsp_silence_handling():
    cfg = DSPConfig(engine="world", pitch_factor=1.6)
    conv = DSPConverter(cfg, sample_rate=16000)

    silence = np.zeros(4800, dtype=np.float32)
    out = conv.process(silence)

    assert out.shape == silence.shape
    assert out.dtype == np.float32
    assert np.all(out == 0.0)


def test_dsp_audio_processing_shape_and_finite():
    cfg = DSPConfig(engine="world", pitch_factor=1.6, formant_shift=1.15)
    conv = DSPConverter(cfg, sample_rate=16000)

    # 300ms 200 Hz harmonic test tone
    t = np.arange(4800) / 16000.0
    audio = (0.5 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    out = conv.process(audio)

    # Must preserve exact length for streaming chunks
    assert len(out) == len(audio)
    assert out.dtype == np.float32

    # No NaN or Inf
    assert np.all(np.isfinite(out))

    # Output energy should be present
    rms = np.sqrt(np.mean(out**2))
    assert rms > 0.01


def test_phase_vocoder_fallback():
    cfg = DSPConfig(engine="phase_vocoder", pitch_factor=1.5)
    conv = DSPConverter(cfg, sample_rate=16000)

    t = np.arange(4800) / 16000.0
    audio = (0.4 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)

    out = conv.process(audio)
    assert len(out) == len(audio)
    assert np.all(np.isfinite(out))
