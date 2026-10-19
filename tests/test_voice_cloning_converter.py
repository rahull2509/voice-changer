"""Automated tests for VoiceCloningConverter."""

import numpy as np
import pytest
import soundfile as sf
import torch

from app.config import VoiceCloningConfig
from app.conversion.voice_cloning_converter import (
    NeuralTimbreConditionedGenerator,
    VoiceCloningConverter,
)
from app.conversion.voice_reference import VoiceReferenceManager


@pytest.fixture
def prepared_cloning_env(tmp_path):
    """Create a temporary environment with prepared reference clips."""
    ref_dir = tmp_path / "voice-clone"
    cache_dir = tmp_path / "cache-clone"
    ref_dir.mkdir()
    cache_dir.mkdir()

    sr = 16000
    t = np.arange(sr * 2) / float(sr)
    # Synthetic target speech clip
    clip = (0.4 * np.sin(2 * np.pi * 215 * t)).astype(np.float32)
    sf.write(str(ref_dir / "target_sample.wav"), clip, sr)

    mgr = VoiceReferenceManager(str(ref_dir), str(cache_dir), target_sample_rate=sr)
    mgr.prepare_reference()

    cfg = VoiceCloningConfig(
        reference_dir=str(ref_dir),
        cache_dir=str(cache_dir),
        target_pitch_scale=1.0,
        timbre_mix=1.0,
    )
    return cfg


def test_voice_cloning_unprepared_raises_actionable_error(tmp_path):
    """When reference is not prepared, process() must raise clear instructions."""
    cfg = VoiceCloningConfig(
        reference_dir=str(tmp_path / "nonexistent"),
        cache_dir=str(tmp_path / "empty_cache"),
    )
    conv = VoiceCloningConverter(cfg, sample_rate=16000, warmup=False)
    assert "Reference not prepared" in conv.name

    dummy_audio = np.ones(1600, dtype=np.float32) * 0.1
    with pytest.raises(RuntimeError) as exc_info:
        conv.process(dummy_audio)
    assert "Voice Cloning mode is not ready" in str(exc_info.value)
    assert "Prepare Voice Reference" in str(exc_info.value)


def test_voice_cloning_processing_invariants(prepared_cloning_env):
    cfg = prepared_cloning_env
    conv = VoiceCloningConverter(cfg, sample_rate=16000, warmup=True)

    assert conv.is_loaded
    assert "Voice Cloning" in conv.name
    assert conv.target_profile is not None
    assert 200.0 <= conv.target_profile.f0_median <= 230.0

    # Input: 300ms speech-like chunk
    sr = 16000
    chunk_len = int(sr * 0.3)
    t = np.linspace(0, 0.3, chunk_len, endpoint=False)
    input_chunk = (0.35 * np.sin(2 * np.pi * 125 * t)).astype(np.float32)  # Male fundamental ~125 Hz

    out_chunk = conv.process(input_chunk)

    # Invariants
    assert len(out_chunk) == chunk_len
    assert out_chunk.dtype == np.float32
    assert np.all(np.isfinite(out_chunk))
    assert np.max(np.abs(out_chunk)) <= 1.0

    conv.cleanup()
    assert not conv.is_loaded


def test_voice_cloning_silence_gate(prepared_cloning_env):
    cfg = prepared_cloning_env
    conv = VoiceCloningConverter(cfg, sample_rate=16000, warmup=False)

    sr = 16000
    silent_chunk = np.zeros(int(sr * 0.3), dtype=np.float32)
    out = conv.process(silent_chunk)

    assert len(out) == len(silent_chunk)
    # Output must be perfect silence
    assert np.all(out == 0.0)


def test_voice_cloning_handles_nan_and_inf(prepared_cloning_env):
    cfg = prepared_cloning_env
    conv = VoiceCloningConverter(cfg, sample_rate=16000, warmup=False)

    sr = 16000
    corrupt_chunk = np.ones(int(sr * 0.2), dtype=np.float32) * 0.1
    corrupt_chunk[100] = np.nan
    corrupt_chunk[200] = np.inf

    # Should not crash; returns safe fallback
    out = conv.process(corrupt_chunk)
    assert len(out) == len(corrupt_chunk)


def test_neural_generator_architecture_on_cpu():
    """Verify NeuralTimbreConditionedGenerator executes forward pass on CPU."""
    model = NeuralTimbreConditionedGenerator(in_channels=80, emb_dim=256, hidden_dim=64)
    model.eval()

    B, T = 1, 18
    mel = torch.randn(B, 80, T)
    spk_emb = torch.randn(B, 256)
    pitch = torch.randn(B, 1, T)

    with torch.no_grad():
        out = model(mel, spk_emb, pitch)

    assert out.ndim == 3
    assert out.shape[0] == B
    assert out.shape[1] == 1
    assert out.shape[2] == T * 256
    assert torch.all(torch.isfinite(out))
