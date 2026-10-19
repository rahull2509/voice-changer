"""Automated tests for VoiceReferenceManager and reference audio preparation."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.conversion.voice_reference import (
    AudioClipMetadata,
    TargetVoiceProfile,
    VoiceReferenceManager,
)


@pytest.fixture
def temp_ref_env(tmp_path):
    """Create a temporary directory with synthetic reference clips and cache."""
    ref_dir = tmp_path / "voice-clone-test"
    cache_dir = tmp_path / "cache-test"
    ref_dir.mkdir()
    cache_dir.mkdir()

    sr = 16000
    t = np.arange(sr * 2) / float(sr)

    # 1. Clip 1: Sine at 210 Hz (target female pitch range)
    clip1 = (0.5 * np.sin(2 * np.pi * 210 * t)).astype(np.float32)
    # Add leading and trailing silence
    clip1_padded = np.pad(clip1, (sr // 2, sr // 2))
    sf.write(str(ref_dir / "clip1.wav"), clip1_padded, sr)

    # 2. Clip 2: Sine at 220 Hz
    clip2 = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(ref_dir / "clip2.wav"), clip2, sr)

    # 3. Clip 3: Sine at 200 Hz
    clip3 = (0.6 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    sf.write(str(ref_dir / "clip3.wav"), clip3, sr)

    return ref_dir, cache_dir


def test_reference_manager_discovery_and_readiness(temp_ref_env):
    ref_dir, cache_dir = temp_ref_env
    mgr = VoiceReferenceManager(str(ref_dir), str(cache_dir))

    status, msg = mgr.get_readiness()
    assert status == "needs_preparation"
    assert "3 reference files" in msg

    fp = mgr.compute_fingerprint()
    assert len(fp) == 64  # SHA256 hex


def test_reference_manager_prepare_and_caching(temp_ref_env):
    ref_dir, cache_dir = temp_ref_env
    mgr = VoiceReferenceManager(str(ref_dir), str(cache_dir))

    profile = mgr.prepare_reference()
    assert profile.total_clips == 3
    assert profile.total_duration_sec > 4.0
    assert 190.0 <= profile.f0_median <= 230.0

    # Verify cache files were created
    assert (cache_dir / "profile.json").is_file()
    assert (cache_dir / "reference_clean_16k.wav").is_file()
    assert (cache_dir / "speaker_embedding.npy").is_file()

    # Readiness should now be 'ready'
    status, msg = mgr.get_readiness()
    assert status == "ready"

    # Reload from cache
    reloaded = mgr.load_cached_profile()
    assert reloaded.total_clips == profile.total_clips
    assert reloaded.f0_median == profile.f0_median
    assert mgr.speaker_embedding is not None
    assert mgr.speaker_embedding.shape == (256,)


def test_reference_manager_handles_corrupt_and_silent_files(tmp_path):
    ref_dir = tmp_path / "bad-refs"
    cache_dir = tmp_path / "bad-cache"
    ref_dir.mkdir()
    cache_dir.mkdir()

    # Empty file
    (ref_dir / "empty.wav").write_bytes(b"")

    # Corrupt header file
    (ref_dir / "corrupt.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    # Silent audio file (1 sec of zeros)
    sr = 16000
    zeros = np.zeros(sr, dtype=np.float32)
    sf.write(str(ref_dir / "silent.wav"), zeros, sr)

    mgr = VoiceReferenceManager(str(ref_dir), str(cache_dir))
    with pytest.raises(RuntimeError) as exc_info:
        mgr.prepare_reference()
    assert "No valid audio could be prepared" in str(exc_info.value)


def test_reference_manager_silence_trimming(temp_ref_env):
    ref_dir, cache_dir = temp_ref_env
    mgr = VoiceReferenceManager(str(ref_dir), str(cache_dir))

    sr = 16000
    # 1 second of silence + 1 second of tone + 1 second of silence
    silence = np.zeros(sr, dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    audio = np.concatenate([silence, tone, silence])

    trimmed = mgr.trim_silence(audio, sr)
    # Trimmed audio should be significantly shorter than 3 seconds
    assert len(trimmed) < len(audio)
    assert len(trimmed) >= sr * 0.9  # preserves the speech tone


def test_speaker_embedding_properties(temp_ref_env):
    ref_dir, cache_dir = temp_ref_env
    mgr = VoiceReferenceManager(str(ref_dir), str(cache_dir))

    sr = 16000
    test_audio = (0.5 * np.sin(2 * np.pi * 210 * np.arange(sr * 2) / sr)).astype(np.float32)
    emb = mgr.extract_speaker_embedding(test_audio, sr)

    assert emb.shape == (256,)
    assert emb.dtype == np.float32
    # Should be unit normalized (L2 norm ~ 1.0)
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, rel=1e-3) == 1.0
