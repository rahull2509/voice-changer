"""Unit tests for RVC models, loader, and neural architecture."""

import numpy as np
import pytest
import torch

from app.config import RVCConfig
from app.conversion.model_loader import check_hubert_available, get_available_models
from app.conversion.rvc_converter import RVCConverter
from app.conversion.rvc_models import SynthesizerTrn


def test_model_loader_discovers_dsp_presets():
    models = get_available_models("models")
    assert len(models) >= 3
    ids = [m.id for m in models]
    assert "dsp_indian_female" in ids
    assert "dsp_young_female" in ids
    assert "dsp_warm_female" in ids


def test_rvc_converter_missing_files_error_diagnostics():
    cfg = RVCConfig(model_path="models/nonexistent_model.pth", hubert_path="models/hubert_base.pt")
    conv = RVCConverter(cfg, sample_rate=16000)

    assert conv.name == "RVC AI (No checkpoint loaded)"

    # Calling process on unloaded model must raise actionable RuntimeError, not crash silently
    dummy_audio = np.zeros(1600, dtype=np.float32)
    with pytest.raises(RuntimeError) as exc_info:
        conv.process(dummy_audio)
    assert "RVC AI Model is not loaded" in str(exc_info.value)


def test_synthesizer_trn_architecture_on_cpu():
    """Verify neural generator initializes and executes on CPU."""
    model = SynthesizerTrn(
        spec_channels=1025,
        segment_size=32,
        inter_channels=64,
        hidden_channels=64,
        filter_channels=256,
        n_heads=2,
        n_layers=2,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=128,
        upsample_kernel_sizes=[16, 16, 4, 4],
        spk_embed_dim=0,
        gin_channels=0,
        sr=40000,
        f0=True,
        version="v2",
    )
    model.eval()

    # Pass dummy HuBERT features [B, T, 768] and pitch [B, T]
    B, T = 1, 15
    dummy_phone = torch.randn(B, T, 768)
    dummy_pitch = torch.randint(0, 255, (B, T))

    with torch.no_grad():
        out_wav = model.infer(dummy_phone, pitch=dummy_pitch)

    # Output should be raw audio tensor [B, 1, samples]
    assert out_wav.ndim == 3
    assert out_wav.shape[0] == 1
    assert out_wav.shape[1] == 1
    assert out_wav.shape[2] > 0
    assert torch.all(torch.isfinite(out_wav))
