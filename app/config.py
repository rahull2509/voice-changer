"""Configuration management for the voice changer application."""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class AudioConfig:
    """Audio device and stream configuration."""

    input_device: Optional[int] = None    # None = system default
    output_device: Optional[int] = None   # None = system default
    internal_sample_rate: int = 16000     # Processing sample rate (Hz)
    chunk_duration_ms: int = 300          # Chunk duration for processing
    pad_duration_ms: int = 150            # Context padding on each side
    crossfade_ms: int = 10                # Crossfade between output chunks
    input_channels: int = 1
    output_channels: int = 1
    queue_max_size: int = 30              # Max chunks in output buffer
    blocksize: int = 1024                 # Sounddevice callback block size


@dataclass
class DSPConfig:
    """DSP mode configuration."""

    engine: str = "world"                 # "world" or "phase_vocoder"
    pitch_factor: float = 1.6             # f0 multiplier (>1 = higher pitch)
    formant_shift: float = 1.15           # Spectral envelope warp (>1 = up)
    output_gain: float = 0.85            # Output volume (0.0–2.0)
    dry_wet_mix: float = 1.0             # 0.0 = dry, 1.0 = fully processed


@dataclass
class RVCConfig:
    """RVC AI mode configuration (Phase 2)."""

    model_path: Optional[str] = None
    index_path: Optional[str] = None
    hubert_path: str = "models/hubert_base.pt"
    f0_method: str = "pm"                 # pm, harvest, dio
    pitch_shift_semitones: int = 12
    index_rate: float = 0.5
    chunk_duration_ms: int = 2000


@dataclass
class VoiceCloningConfig:
    """Voice Cloning reference-conditioned mode configuration."""

    reference_dir: str = "voice-clone"
    cache_dir: str = "cache/voice_clone"
    model_path: Optional[str] = None
    target_pitch_scale: float = 1.0       # Pitch adjustment factor relative to reference (0.8–1.3)
    timbre_mix: float = 1.0               # 0.0–1.0 reference timbre strength
    f0_method: str = "world"              # "world"


@dataclass
class AppConfig:
    """Top-level application configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    rvc: RVCConfig = field(default_factory=RVCConfig)
    voice_cloning: VoiceCloningConfig = field(default_factory=VoiceCloningConfig)
    mode: str = "dsp"                     # "dsp", "rvc", or "voice_cloning"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AppConfig":
        return cls(
            audio=AudioConfig(**{k: v for k, v in d.get("audio", {}).items()
                                 if k in AudioConfig.__dataclass_fields__}),
            dsp=DSPConfig(**{k: v for k, v in d.get("dsp", {}).items()
                             if k in DSPConfig.__dataclass_fields__}),
            rvc=RVCConfig(**{k: v for k, v in d.get("rvc", {}).items()
                             if k in RVCConfig.__dataclass_fields__}),
            voice_cloning=VoiceCloningConfig(**{k: v for k, v in d.get("voice_cloning", {}).items()
                                               if k in VoiceCloningConfig.__dataclass_fields__}),
            mode=d.get("mode", "dsp"),
            host=d.get("host", "127.0.0.1"),
            port=d.get("port", 8000),
            log_level=d.get("log_level", "info"),
        )

    def save(self, path: str = "config.json") -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str = "config.json") -> "AppConfig":
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except FileNotFoundError:
            return cls()

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        cfg = cls()

        if v := os.getenv("INPUT_DEVICE"):
            cfg.audio.input_device = int(v)
        if v := os.getenv("OUTPUT_DEVICE"):
            cfg.audio.output_device = int(v)
        if v := os.getenv("INTERNAL_SAMPLE_RATE"):
            cfg.audio.internal_sample_rate = int(v)
        if v := os.getenv("CHUNK_DURATION_MS"):
            cfg.audio.chunk_duration_ms = int(v)
        if v := os.getenv("PAD_DURATION_MS"):
            cfg.audio.pad_duration_ms = int(v)
        if v := os.getenv("CROSSFADE_MS"):
            cfg.audio.crossfade_ms = int(v)

        if v := os.getenv("MODE"):
            cfg.mode = v
        if v := os.getenv("HOST"):
            cfg.host = v
        if v := os.getenv("PORT"):
            cfg.port = int(v)

        if v := os.getenv("DSP_ENGINE"):
            cfg.dsp.engine = v
        if v := os.getenv("DSP_PITCH_FACTOR"):
            cfg.dsp.pitch_factor = float(v)
        if v := os.getenv("DSP_FORMANT_SHIFT"):
            cfg.dsp.formant_shift = float(v)
        if v := os.getenv("DSP_OUTPUT_GAIN"):
            cfg.dsp.output_gain = float(v)
        if v := os.getenv("DSP_DRY_WET_MIX"):
            cfg.dsp.dry_wet_mix = float(v)

        if v := os.getenv("RVC_MODEL_PATH"):
            cfg.rvc.model_path = v
        if v := os.getenv("RVC_INDEX_PATH"):
            cfg.rvc.index_path = v
        if v := os.getenv("RVC_HUBERT_PATH"):
            cfg.rvc.hubert_path = v
        if v := os.getenv("RVC_PITCH_SHIFT_SEMITONES"):
            cfg.rvc.pitch_shift_semitones = int(v)

        if v := os.getenv("VOICE_CLONE_REF_DIR"):
            cfg.voice_cloning.reference_dir = v
        if v := os.getenv("VOICE_CLONE_PITCH_SCALE"):
            cfg.voice_cloning.target_pitch_scale = float(v)
        if v := os.getenv("VOICE_CLONE_TIMBRE_MIX"):
            cfg.voice_cloning.timbre_mix = float(v)

        return cfg
