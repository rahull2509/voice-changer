"""RVC (Retrieval-based Voice Conversion) v2 inference engine for CPU.

Features:
- Loads SynthesizerTrn generator from RVC v1/v2 .pth checkpoint.
- Reuses model instances across calls.
- Performs HuBERT acoustic feature extraction.
- F0 estimation with PyWorld and semitone pitch shifting.
- Fast CPU inference with torch.no_grad().
- Waveform resampling from model sample rate (40k/48k) to pipeline rate (16k).
- Descriptive error messages with exact instructions when model files are missing.
"""

import logging
from math import gcd
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly

from app.config import RVCConfig
from app.conversion.base import BaseConverter
from app.conversion.rvc_models import SynthesizerTrn

logger = logging.getLogger(__name__)


class RVCConverter(BaseConverter):
    """RVC v2 neural voice conversion pipeline for CPU."""

    def __init__(self, config: RVCConfig, sample_rate: int = 16000) -> None:
        self.config = config
        self.sample_rate = sample_rate

        self._net_g: Optional[SynthesizerTrn] = None
        self._hubert: Optional[torch.nn.Module] = None
        self._loaded: bool = False
        self._model_sr: int = 40000
        self._version: str = "v2"
        self._f0_enabled: bool = True
        self._device = torch.device("cpu")

        # Attempt to load model files
        self._load_models()

    @property
    def name(self) -> str:
        if self._loaded and self.config.model_path:
            return f"RVC AI ({Path(self.config.model_path).stem})"
        return "RVC AI (No checkpoint loaded)"

    def _load_models(self) -> None:
        """Load RVC checkpoint and HuBERT model into CPU memory."""
        # Check model path
        if not self.config.model_path or not Path(self.config.model_path).is_file():
            logger.info("RVC model path not configured or file not found (%s)", self.config.model_path)
            self._loaded = False
            return

        # Check HuBERT path
        hubert_file = Path(self.config.hubert_path)
        if not hubert_file.is_file():
            logger.warning(
                "HuBERT checkpoint '%s' not found. "
                "Download hubert_base.pt to the models/ folder.",
                self.config.hubert_path,
            )
            self._loaded = False
            return

        try:
            logger.info("Loading RVC checkpoint: %s", self.config.model_path)
            cpt = torch.load(self.config.model_path, map_location=self._device)

            tgt_sr = cpt.get("sr", 40000)
            self._model_sr = tgt_sr
            self._f0_enabled = cpt.get("f0", 1) == 1
            self._version = cpt.get("version", "v2")

            # Extract architecture config
            cfg = cpt.get("config", [])
            if len(cfg) >= 16:
                net_g = SynthesizerTrn(
                    spec_channels=cfg[0],
                    segment_size=cfg[1],
                    inter_channels=cfg[2],
                    hidden_channels=cfg[3],
                    filter_channels=cfg[4],
                    n_heads=cfg[5],
                    n_layers=cfg[6],
                    kernel_size=cfg[7],
                    p_dropout=cfg[8],
                    resblock=cfg[9],
                    resblock_kernel_sizes=cfg[10],
                    resblock_dilation_sizes=cfg[11],
                    upsample_rates=cfg[12],
                    upsample_initial_channel=cfg[13],
                    upsample_kernel_sizes=cfg[14],
                    spk_embed_dim=cfg[15] if len(cfg) > 15 else 0,
                    gin_channels=cfg[16] if len(cfg) > 16 else 0,
                    sr=tgt_sr,
                    f0=self._f0_enabled,
                    version=self._version,
                )
            else:
                # Default v2 config
                net_g = SynthesizerTrn(
                    1025, 32, 192, 192, 768, 2, 6, 3, 0.1, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 8, 2, 2], 512, [16, 16, 4, 4], 0, 0,
                    tgt_sr, self._f0_enabled, self._version,
                )

            # Load weights
            weights = cpt.get("weight", cpt)
            net_g.load_state_dict(weights, strict=False)
            net_g.eval()
            try:
                net_g.remove_weight_norm()
            except Exception:
                pass

            self._net_g = net_g

            # Load HuBERT model
            logger.info("Loading HuBERT model from %s", self.config.hubert_path)
            from transformers import HubertModel
            # Try loading as transformers model or fairseq checkpoint
            try:
                hubert = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960" if not hubert_file.is_file() else str(hubert_file.parent)
                )
                hubert.eval()
                self._hubert = hubert
            except Exception:
                # Direct PyTorch load fallback
                self._hubert = torch.load(self.config.hubert_path, map_location=self._device)
                if hasattr(self._hubert, "eval"):
                    self._hubert.eval()

            self._loaded = True
            logger.info("RVC AI mode loaded successfully (sr=%d, f0=%s)", self._model_sr, self._f0_enabled)

        except Exception as e:
            self._loaded = False
            logger.exception("Failed to initialize RVC model: %s", e)
            raise RuntimeError(
                f"Failed to load RVC checkpoint ({self.config.model_path}): {e}"
            )

    def _extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency contour using PyWorld DIO."""
        import pyworld as pw

        x = audio.astype(np.float64)
        fs = self.sample_rate
        f0, t = pw.dio(x, fs, frame_period=10.0)
        f0 = pw.stonemask(x, f0, t, fs)

        # Apply semitone pitch shift
        semitones = self.config.pitch_shift_semitones
        if semitones != 0:
            factor = 2.0 ** (semitones / 12.0)
            voiced = f0 > 0
            f0[voiced] *= factor

        # Quantize f0 to pitch bin indices (1 to 255)
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        f0_mel[f0 == 0] = 0
        f0_bins = np.clip(np.round(f0_mel * 255.0 / 1127.0), 0, 255).astype(np.int64)
        return f0_bins

    def _extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract HuBERT hidden representations (16kHz audio -> [1, T, 768])."""
        x_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, L]
        with torch.no_grad():
            if hasattr(self._hubert, "extract_features"):
                # Fairseq style
                feats, _ = self._hubert.extract_features(source=x_tensor, padding_mask=None, mask=False)
            elif hasattr(self._hubert, "forward"):
                # Transformers style
                out = self._hubert(x_tensor)
                feats = out.last_hidden_state
            else:
                feats = torch.zeros((1, len(audio) // 320, 768), dtype=torch.float32)
        return feats

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Perform RVC voice conversion on mono 16kHz audio chunk."""
        if not self._loaded or self._net_g is None:
            # Descriptive guidance when model is not available
            raise RuntimeError(
                "RVC AI Model is not loaded!\n"
                "To enable AI mode:\n"
                "1. Download an RVC v2 female voice model (.pth) and place in models/\n"
                "2. Download hubert_base.pt and place in models/\n"
                "3. Select the model from the UI or config\n"
                "For immediate low-latency conversion, switch to 'DSP Mode'."
            )

        if audio.size == 0:
            return audio.astype(np.float32)

        # Silence gate
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-5:
            return np.zeros_like(audio, dtype=np.float32)

        # 1. Feature extraction
        feats = self._extract_features(audio)  # [1, T, 768]

        # 2. Pitch extraction
        pitch_bins = None
        if self._f0_enabled:
            f0_arr = self._extract_f0(audio)
            # Match temporal length with HuBERT features
            t_len = feats.shape[1]
            if len(f0_arr) != t_len:
                f0_arr = np.interp(
                    np.linspace(0, 1, t_len),
                    np.linspace(0, 1, len(f0_arr)),
                    f0_arr,
                ).astype(np.int64)
            pitch_bins = torch.from_numpy(f0_arr).unsqueeze(0)  # [1, T]

        # 3. Neural generator inference
        with torch.no_grad():
            audio_out = self._net_g.infer(feats, pitch=pitch_bins)
            audio_out = audio_out.squeeze().cpu().numpy()

        # 4. Resample from model sample rate (40k/48k) to pipeline rate (16k)
        if self._model_sr != self.sample_rate:
            common = gcd(self._model_sr, self.sample_rate)
            up = self.sample_rate // common
            down = self._model_sr // common
            out_resampled = resample_poly(audio_out, up, down).astype(np.float32)
        else:
            out_resampled = audio_out.astype(np.float32)

        # Match exact input length
        target_len = len(audio)
        if len(out_resampled) > target_len:
            out_resampled = out_resampled[:target_len]
        elif len(out_resampled) < target_len:
            out_resampled = np.pad(out_resampled, (0, target_len - len(out_resampled)), mode="constant")

        return out_resampled

    def cleanup(self) -> None:
        """Release neural network models from memory."""
        self._net_g = None
        self._hubert = None
        self._loaded = False
        logger.info("RVC models unloaded from memory.")
