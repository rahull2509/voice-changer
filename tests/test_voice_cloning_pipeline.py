"""Automated tests for AudioPipeline with Voice Cloning mode."""

import time
import numpy as np
import pytest

from app.audio.pipeline import AudioPipeline
from app.config import AppConfig


def test_pipeline_instantiates_voice_cloning_converter():
    cfg = AppConfig.load()
    cfg.mode = "voice_cloning"
    pipeline = AudioPipeline(cfg)

    converter = pipeline._init_converter()
    assert converter is not None
    assert "Voice Cloning" in converter.name


def test_pipeline_mode_switching_and_restart(monkeypatch):
    """Test switching between dsp and voice_cloning modes."""
    cfg = AppConfig.load()
    cfg.mode = "dsp"
    pipeline = AudioPipeline(cfg)

    conv_dsp = pipeline._init_converter()
    assert "DSP" in conv_dsp.name

    # Switch config to voice_cloning
    cfg_cloning = AppConfig.load()
    cfg_cloning.mode = "voice_cloning"
    conv_cloning = AudioPipeline(cfg_cloning)._init_converter()
    assert "Voice Cloning" in conv_cloning.name


def test_pipeline_status_reporting():
    cfg = AppConfig.load()
    cfg.mode = "voice_cloning"
    pipeline = AudioPipeline(cfg)

    status_dict = pipeline.status.to_dict()
    assert status_dict["active_mode"] == "voice_cloning"
    assert "overload" in status_dict
    assert "overload_count" in status_dict
    assert "real_time_factor" in status_dict
