"""DSP-based voice conversion using WORLD vocoder or phase-vocoder fallback.

WORLD mode
----------
Uses the WORLD vocoder (pyworld) for high-quality pitch shifting with
independent formant control.  Requires chunks of >= ~200 ms for reliable
f0 estimation; the pipeline supplies padded analysis windows so this is
satisfied.

Phase-vocoder mode
------------------
STFT-based pitch shifting via time-stretch + resample.  Works on any chunk
size but does NOT preserve formants — the ``formant_shift`` parameter is
ignored.  Clearly labelled as a lower-quality fallback.

Neither mode is neural voice conversion — both are signal-processing
transforms.  They are labelled "DSP Mode" in the UI.
"""

import logging
import numpy as np
from typing import Optional

from app.config import DSPConfig
from app.conversion.base import BaseConverter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional WORLD import
# ---------------------------------------------------------------------------
try:
    import pyworld as pw

    WORLD_AVAILABLE = True
except ImportError:
    WORLD_AVAILABLE = False
    logger.info("pyworld not installed — WORLD engine unavailable")


class DSPConverter(BaseConverter):
    """Voice feminisation via pitch + formant DSP processing."""

    def __init__(self, config: DSPConfig, sample_rate: int = 16000) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self._engine = config.engine

        if self._engine == "world" and not WORLD_AVAILABLE:
            logger.warning(
                "WORLD engine requested but pyworld is not installed. "
                "Falling back to phase_vocoder."
            )
            self._engine = "phase_vocoder"

        logger.info(
            "DSPConverter ready  engine=%s  pitch=%.2f  formant=%.2f  gain=%.2f",
            self._engine,
            config.pitch_factor,
            config.formant_shift,
            config.output_gain,
        )

    # ------------------------------------------------------------------
    # BaseConverter interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"DSP ({self._engine})"

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process one analysis window (with padding) of mono audio."""
        if audio.size == 0:
            return audio.astype(np.float32)

        # Silence gate — skip processing on very quiet input
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-6:
            return np.zeros_like(audio, dtype=np.float32)

        try:
            if self._engine == "world":
                out = self._process_world(audio)
            else:
                out = self._process_phase_vocoder(audio)
            return out.astype(np.float32)
        except Exception:
            logger.exception("DSP processing error — returning original audio")
            return audio.astype(np.float32)

    # ------------------------------------------------------------------
    # WORLD engine
    # ------------------------------------------------------------------
    def _process_world(self, audio: np.ndarray) -> np.ndarray:
        fs = self.sample_rate
        x = audio.astype(np.float64)

        # --- analysis ---------------------------------------------------
        f0, t = pw.dio(x, fs, frame_period=5.0)
        f0 = pw.stonemask(x, f0, t, fs)
        sp = pw.cheaptrick(x, f0, t, fs)
        ap = pw.d4c(x, f0, t, fs)

        # --- pitch modification -----------------------------------------
        f0_mod = f0.copy()
        voiced = f0_mod > 0
        f0_mod[voiced] *= self.config.pitch_factor
        f0_mod[voiced] = np.clip(f0_mod[voiced], 80.0, 880.0)

        # --- formant modification ----------------------------------------
        if abs(self.config.formant_shift - 1.0) > 0.01:
            sp = self._warp_spectral_envelope(sp, self.config.formant_shift)

        # --- synthesis ---------------------------------------------------
        sp = np.ascontiguousarray(sp, dtype=np.float64)
        ap = np.ascontiguousarray(ap, dtype=np.float64)
        f0_mod = np.ascontiguousarray(f0_mod, dtype=np.float64)
        out = pw.synthesize(f0_mod, sp, ap, fs, frame_period=5.0)

        # WORLD synthesis frame quantization can produce slightly differing sample count.
        # Align exact length to match input window:
        target_len = len(audio)
        if len(out) > target_len:
            out = out[:target_len]
        elif len(out) < target_len:
            out = np.pad(out, (0, target_len - len(out)), mode="constant")

        return out

    @staticmethod
    def _warp_spectral_envelope(sp: np.ndarray, factor: float) -> np.ndarray:
        """Warp the spectral envelope along the frequency axis.

        ``factor > 1`` shifts formants upward (male → female direction).
        """
        _nframes, nbins = sp.shape
        indices = np.arange(nbins, dtype=np.float64)
        query = indices / factor  # source indices for each target bin
        warped = np.empty_like(sp)
        for i in range(_nframes):
            warped[i] = np.interp(query, indices, sp[i], left=sp[i, 0], right=1e-16)
        return np.maximum(warped, 1e-16)

    # ------------------------------------------------------------------
    # Phase-vocoder fallback
    # ------------------------------------------------------------------
    def _process_phase_vocoder(self, audio: np.ndarray) -> np.ndarray:
        """Pitch-shift via STFT time-stretch + resample.

        Does NOT preserve formants — this is a simple pitch shifter, not
        a neural voice converter.
        """
        from scipy.signal import stft as _stft, istft as _istft, resample

        factor = self.config.pitch_factor
        if abs(factor - 1.0) < 0.01:
            return audio.astype(np.float32)

        fs = self.sample_rate
        n_fft = 2048
        hop = n_fft // 4

        # ---- STFT -------------------------------------------------------
        _f, _t, Zxx = _stft(audio, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
        n_freq, n_time = Zxx.shape

        # ---- Phase-vocoder time-stretch by 1 / factor -------------------
        rate = 1.0 / factor
        time_steps = np.arange(0, n_time - 1, rate)
        mag = np.abs(Zxx)
        phi = np.angle(Zxx)

        phase_adv = np.linspace(0, np.pi * hop, n_freq, endpoint=False)

        out_mag = np.zeros((n_freq, len(time_steps)), dtype=np.float64)
        out_phi = np.zeros_like(out_mag)
        out_phi[:, 0] = phi[:, 0]

        for i, step in enumerate(time_steps):
            idx = int(step)
            frac = step - idx
            if idx + 1 >= n_time:
                out_mag[:, i] = mag[:, -1]
                if i > 0:
                    out_phi[:, i] = out_phi[:, i - 1]
                break
            out_mag[:, i] = (1 - frac) * mag[:, idx] + frac * mag[:, idx + 1]
            if i > 0:
                dphi = phi[:, idx + 1] - phi[:, idx] - phase_adv
                dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
                out_phi[:, i] = out_phi[:, i - 1] + phase_adv + dphi

        stretched_stft = out_mag * np.exp(1j * out_phi)
        _, stretched = _istft(stretched_stft, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)

        # ---- Resample to original length → changes pitch -----------------
        if stretched.size == 0:
            return audio.astype(np.float32)
        result = resample(stretched, len(audio)).astype(np.float32)
        return result
