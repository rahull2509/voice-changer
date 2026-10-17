"""Unit tests for configuration management."""

import json
from pathlib import Path

import pytest
from app.config import AppConfig, AudioConfig, DSPConfig, RVCConfig


def test_default_config():
    cfg = AppConfig()
    assert cfg.mode == "dsp"
    assert cfg.dsp.pitch_factor == 1.6
    assert cfg.dsp.formant_shift == 1.15
    assert cfg.audio.internal_sample_rate == 16000
    assert cfg.audio.chunk_duration_ms == 300


def test_config_serialization(tmp_path: Path):
    cfg = AppConfig()
    cfg.dsp.pitch_factor = 1.75
    cfg.audio.input_device = 9
    cfg.audio.output_device = 1

    file_path = tmp_path / "test_config.json"
    cfg.save(str(file_path))

    loaded = AppConfig.load(str(file_path))
    assert loaded.dsp.pitch_factor == 1.75
    assert loaded.audio.input_device == 9
    assert loaded.audio.output_device == 1
    assert loaded.mode == "dsp"


def test_from_dict():
    data = {
        "mode": "rvc",
        "audio": {"chunk_duration_ms": 500, "internal_sample_rate": 16000},
        "dsp": {"pitch_factor": 1.5, "formant_shift": 1.1},
        "rvc": {"pitch_shift_semitones": 12},
    }
    cfg = AppConfig.from_dict(data)
    assert cfg.mode == "rvc"
    assert cfg.audio.chunk_duration_ms == 500
    assert cfg.dsp.pitch_factor == 1.5
    assert cfg.rvc.pitch_shift_semitones == 12
