"""FastAPI route definitions for the voice changer backend."""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.audio.devices import get_input_devices, get_output_devices, list_all_devices
from app.audio.pipeline import AudioPipeline
from app.config import AppConfig
from app.conversion.model_loader import get_available_models

logger = logging.getLogger(__name__)

router = APIRouter()

# Global pipeline reference set by main.py
_pipeline: Optional[AudioPipeline] = None
_start_time = time.time()


def set_pipeline(pipeline: AudioPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> AudioPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="Audio pipeline not initialized")
    return _pipeline


# --- Pydantic Request Models ---

class SettingsUpdateRequest(BaseModel):
    mode: Optional[str] = Field(None, description="'dsp' or 'rvc'")
    input_device: Optional[int] = Field(None, description="Input device index (None for system default)")
    output_device: Optional[int] = Field(None, description="Output device index (None for system default)")
    chunk_duration_ms: Optional[int] = Field(None, description="Processing chunk size (ms)")
    dsp_pitch_factor: Optional[float] = Field(None, description="DSP pitch multiplier (>1 = higher pitch)")
    dsp_formant_shift: Optional[float] = Field(None, description="DSP formant warp factor (>1 = feminine)")
    dsp_output_gain: Optional[float] = Field(None, description="Output volume gain")
    dsp_dry_wet_mix: Optional[float] = Field(None, description="0.0 = original voice, 1.0 = fully converted")
    dsp_engine: Optional[str] = Field(None, description="'world' or 'phase_vocoder'")
    rvc_model_path: Optional[str] = Field(None, description="Path to .pth voice model")
    rvc_pitch_shift_semitones: Optional[int] = Field(None, description="Semitones to shift pitch in RVC")


# --- Endpoints ---

@router.get("/health")
def get_health() -> Dict[str, Any]:
    """Health check endpoint exposing runtime uptime and backend status."""
    pipe = get_pipeline()
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "is_running": pipe.status.is_running,
        "mode": pipe.config.mode,
        "platform": "Windows 10 (CPU)",
    }


@router.get("/audio/devices")
def get_devices() -> Dict[str, Any]:
    """Enumerate all available physical and virtual audio devices."""
    inputs = [d.to_dict() for d in get_input_devices()]
    outputs = [d.to_dict() for d in get_output_devices()]
    return {
        "input_devices": inputs,
        "output_devices": outputs,
        "total_devices": len(inputs) + len(outputs),
    }


@router.get("/audio/status")
def get_status() -> Dict[str, Any]:
    """Live processing telemetry, latency measurements, and buffer statistics."""
    pipe = get_pipeline()
    return pipe.status.to_dict()


@router.get("/voice/models")
def list_models() -> Dict[str, Any]:
    """List available voice models (DSP presets and detected RVC checkpoints)."""
    models = get_available_models("models")
    return {
        "models": [m.to_dict() for m in models],
        "active_mode": get_pipeline().config.mode,
    }


@router.get("/settings")
def get_settings() -> Dict[str, Any]:
    """Retrieve the current active configuration."""
    pipe = get_pipeline()
    return pipe.config.to_dict()


@router.put("/settings")
def update_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
    """Update settings live. Restarts audio streams if device/mode changed."""
    pipe = get_pipeline()
    cfg = pipe.config

    if req.mode is not None:
        cfg.mode = req.mode
    if req.input_device is not None:
        cfg.audio.input_device = req.input_device
    if req.output_device is not None:
        cfg.audio.output_device = req.output_device
    if req.chunk_duration_ms is not None:
        cfg.audio.chunk_duration_ms = req.chunk_duration_ms

    if req.dsp_pitch_factor is not None:
        cfg.dsp.pitch_factor = req.dsp_pitch_factor
    if req.dsp_formant_shift is not None:
        cfg.dsp.formant_shift = req.dsp_formant_shift
    if req.dsp_output_gain is not None:
        cfg.dsp.output_gain = req.dsp_output_gain
    if req.dsp_dry_wet_mix is not None:
        cfg.dsp.dry_wet_mix = req.dsp_dry_wet_mix
    if req.dsp_engine is not None:
        cfg.dsp.engine = req.dsp_engine

    if req.rvc_model_path is not None:
        cfg.rvc.model_path = req.rvc_model_path
    if req.rvc_pitch_shift_semitones is not None:
        cfg.rvc.pitch_shift_semitones = req.rvc_pitch_shift_semitones

    try:
        pipe.update_settings(cfg)
        cfg.save("config.json")
        return {"status": "success", "message": "Settings updated", "config": cfg.to_dict()}
    except Exception as e:
        logger.exception("Failed to update settings: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/audio/start")
def start_audio() -> Dict[str, Any]:
    """Start real-time voice conversion audio pipeline."""
    pipe = get_pipeline()
    if pipe.status.is_running:
        return {"status": "already_running", "message": "Audio conversion is already active"}

    try:
        pipe.start()
        return {"status": "started", "message": "Audio conversion pipeline started"}
    except Exception as e:
        logger.exception("Failed to start audio: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to start audio: {e}")


@router.post("/audio/stop")
def stop_audio() -> Dict[str, Any]:
    """Stop real-time voice conversion audio pipeline."""
    pipe = get_pipeline()
    pipe.stop()
    return {"status": "stopped", "message": "Audio conversion pipeline stopped"}


@router.post("/audio/test-sample")
def play_test_sample() -> Dict[str, Any]:
    """Play a natural Hindi speech sample converted to female voice in headphones."""
    from pathlib import Path
    import numpy as np
    import sounddevice as sd
    from scipy.signal import resample_poly
    import scipy.io.wavfile as wavfile
    from app.conversion.dsp_converter import DSPConverter

    pipe = get_pipeline()
    cfg = pipe.config
    sr = 16000

    speech_path = Path(__file__).parent.parent / "ui" / "test_speech.wav"
    if speech_path.exists():
        file_sr, data = wavfile.read(str(speech_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            test_audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            test_audio = data.astype(np.float32) / 2147483648.0
        else:
            test_audio = data.astype(np.float32)
        if file_sr != sr:
            test_audio = resample_poly(test_audio, sr, file_sr).astype(np.float32)
    else:
        # Fallback harmonic vocalization
        t = np.arange(sr * 2) / float(sr)
        test_audio = (0.5 * np.sin(2 * np.pi * 130 * t) + 0.25 * np.sin(2 * np.pi * 260 * t)).astype(np.float32)

    # Convert with active DSP settings
    conv = DSPConverter(cfg.dsp, sample_rate=sr)
    converted = conv.process(test_audio)
    converted = (converted * cfg.dsp.output_gain).astype(np.float32)

    # If pipeline is running, queue it directly to playback stream
    if pipe.status.is_running and pipe.playback:
        # Split into chunks and push to playback queue
        chunk_len = int(sr * 0.3)
        for i in range(0, len(converted), chunk_len):
            sub = converted[i : i + chunk_len]
            pipe.playback.put_chunk(sub)
    else:
        # Play directly to resolved output device
        from app.audio.devices import resolve_output_device
        out_dev = resolve_output_device(cfg.audio.output_device)
        sd.play(converted, samplerate=sr, device=out_dev)

    return {
        "status": "success",
        "message": "Playing 2-second voice converted sample into headphones",
        "pitch_factor": cfg.dsp.pitch_factor,
        "formant_shift": cfg.dsp.formant_shift,
    }
