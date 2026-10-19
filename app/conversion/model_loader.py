"""Model discovery, validation, and metadata extraction for voice models."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata describing an available voice conversion model."""

    id: str
    name: str
    model_type: str  # "dsp_preset" or "rvc_checkpoint"
    path: Optional[str] = None
    index_path: Optional[str] = None
    sample_rate: int = 16000
    pitch_guidance: bool = True
    size_mb: float = 0.0
    description: str = ""
    is_ready: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "path": self.path,
            "index_path": self.index_path,
            "sample_rate": self.sample_rate,
            "pitch_guidance": self.pitch_guidance,
            "size_mb": round(self.size_mb, 1),
            "description": self.description,
            "is_ready": self.is_ready,
        }


def get_available_models(models_dir: str = "models") -> List[ModelInfo]:
    """Scan models directory and return list of DSP presets and RVC models."""
    results: List[ModelInfo] = []

    # 1. Built-in DSP Presets (Always ready, zero latency, no download needed)
    results.append(
        ModelInfo(
            id="dsp_indian_female",
            name="Indian Female (DSP Real-Time)",
            model_type="dsp_preset",
            sample_rate=16000,
            description="Algorithmic WORLD vocoder feminization (Pitch 1.6x, Formant 1.15x). Zero model downloads, ultra-low latency.",
            is_ready=True,
        )
    )
    results.append(
        ModelInfo(
            id="dsp_young_female",
            name="Young Female (DSP Higher Pitch)",
            model_type="dsp_preset",
            sample_rate=16000,
            description="Algorithmic WORLD vocoder feminization (Pitch 1.75x, Formant 1.20x). Bright energetic timbre.",
            is_ready=True,
        )
    )
    results.append(
        ModelInfo(
            id="dsp_warm_female",
            name="Warm Female (DSP Subtle)",
            model_type="dsp_preset",
            sample_rate=16000,
            description="Algorithmic WORLD vocoder feminization (Pitch 1.45x, Formant 1.10x). Natural lower register feminine tone.",
            is_ready=True,
        )
    )

    # 2. Check for RVC .pth checkpoints in models directory
    p = Path(models_dir)
    if p.exists() and p.is_dir():
        pth_files = list(p.glob("*.pth")) + list(p.glob("*.pt"))
        for f in pth_files:
            if f.name.startswith("hubert"):
                continue  # skip hubert base model from voice target list

            size_mb = f.stat().st_size / (1024 * 1024)
            # Find matching .index file if present
            index_file = p / f"{f.stem}.index"
            index_path = str(index_file) if index_file.is_file() else None

            results.append(
                ModelInfo(
                    id=f.stem,
                    name=f.stem.replace("_", " ").title(),
                    model_type="rvc_checkpoint",
                    path=str(f.resolve()),
                    index_path=index_path,
                    sample_rate=40000,
                    size_mb=size_mb,
                    description=f"Neural RVC v2 checkpoint ({size_mb:.1f} MB). High naturalness, requires CPU neural inference.",
                    is_ready=True,
                )
            )

    # 3. Check for prepared Voice Cloning target profile
    try:
        from app.conversion.voice_reference import VoiceReferenceManager
        mgr = VoiceReferenceManager()
        readiness, _ = mgr.get_readiness()
        if readiness == "ready":
            prof = mgr.load_cached_profile()
            results.append(
                ModelInfo(
                    id="voice_clone_target",
                    name=f"Cloned Female Voice ({prof.f0_median:.0f} Hz Target)",
                    model_type="voice_cloning",
                    sample_rate=prof.sample_rate,
                    size_mb=round(prof.total_duration_sec * 0.03, 1),
                    description=f"Neural reference match conditioned on {prof.total_clips} voice messages ({prof.total_duration_sec:.1f}s, target F0 median {prof.f0_median:.0f} Hz).",
                    is_ready=True,
                )
            )
    except Exception as e:
        logger.debug("Voice cloning profile check in model loader: %s", e)

    return results


def check_hubert_available(hubert_path: str = "models/hubert_base.pt") -> bool:
    """Check if the HuBERT feature extractor weights file exists."""
    return Path(hubert_path).is_file()
