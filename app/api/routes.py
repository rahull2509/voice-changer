"""FastAPI route definitions for the voice changer backend."""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
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
    mode: Optional[str] = Field(None, description="'dsp', 'rvc', or 'voice_cloning'")
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
    voice_clone_ref_dir: Optional[str] = Field(None, description="Relative path to voice reference folder")
    voice_clone_pitch_scale: Optional[float] = Field(None, description="Target pitch scale factor (0.8–1.3)")
    voice_clone_timbre_mix: Optional[float] = Field(None, description="Target timbre intensity (0.0–1.0)")


class TTSVoiceRequest(BaseModel):
    text: str = Field(..., description="Text to speak in cloned voice (Hindi, Hinglish, English)")
    voice_name: Optional[str] = Field("hi-IN-SwaraNeural", description="Base neural voice model")
    pitch_offset_hz: Optional[int] = Field(0, description="Fine-tuning pitch offset in Hz (-40 to +40)")
    rate_percent: Optional[int] = Field(-3, description="Speaking rate adjustment in percent (-15 to +15)")
    play_audio: Optional[bool] = Field(True, description="Immediately play to configured output device / virtual mic")


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
        valid_modes = {"dsp", "rvc", "voice_cloning"}
        if req.mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode '{req.mode}'. Must be one of {valid_modes}")
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

    if req.voice_clone_ref_dir is not None:
        # Prevent path traversal
        clean_path = Path(req.voice_clone_ref_dir)
        if clean_path.is_absolute() or ".." in clean_path.parts:
            raise HTTPException(status_code=400, detail="Invalid reference directory path.")
        cfg.voice_cloning.reference_dir = str(clean_path)
    if req.voice_clone_pitch_scale is not None:
        cfg.voice_cloning.target_pitch_scale = float(req.voice_clone_pitch_scale)
    if req.voice_clone_timbre_mix is not None:
        cfg.voice_cloning.timbre_mix = float(req.voice_clone_timbre_mix)

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


@router.get("/voice-cloning/status")
def get_voice_cloning_status() -> Dict[str, Any]:
    """Inspect readiness of target speaker reference and neural model."""
    pipe = get_pipeline()
    from app.conversion.voice_reference import VoiceReferenceManager
    mgr = VoiceReferenceManager(
        reference_dir=pipe.config.voice_cloning.reference_dir,
        cache_dir=pipe.config.voice_cloning.cache_dir,
    )
    readiness, message = mgr.get_readiness()
    profile_dict = None
    if readiness == "ready":
        try:
            profile_dict = mgr.load_cached_profile().to_dict()
        except Exception:
            pass

    return {
        "status": readiness,
        "message": message,
        "fingerprint": mgr.compute_fingerprint(),
        "reference_dir": pipe.config.voice_cloning.reference_dir,
        "is_active_mode": pipe.config.mode == "voice_cloning",
        "profile": profile_dict,
    }


@router.post("/voice-cloning/prepare-reference")
def prepare_voice_reference() -> Dict[str, Any]:
    """Scan reference folder, validate, silence-trim, profile speaker, and save cache."""
    pipe = get_pipeline()
    from app.conversion.voice_reference import VoiceReferenceManager
    mgr = VoiceReferenceManager(
        reference_dir=pipe.config.voice_cloning.reference_dir,
        cache_dir=pipe.config.voice_cloning.cache_dir,
    )
    try:
        profile = mgr.prepare_reference(force_reload=True)
        # If pipeline is running in voice_cloning mode, restart to use new reference
        if pipe.status.is_running and pipe.config.mode == "voice_cloning":
            pipe.stop()
            pipe.start()
        return {
            "status": "success",
            "message": f"Successfully prepared target speaker reference ({profile.total_clips} clips, {profile.total_duration_sec:.1f}s)",
            "profile": profile.to_dict(),
        }
    except Exception as e:
        logger.exception("Failed to prepare voice reference: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/voice-cloning/reload-reference")
def reload_voice_reference() -> Dict[str, Any]:
    """Reload cached target speaker reference."""
    pipe = get_pipeline()
    from app.conversion.voice_reference import VoiceReferenceManager
    mgr = VoiceReferenceManager(
        reference_dir=pipe.config.voice_cloning.reference_dir,
        cache_dir=pipe.config.voice_cloning.cache_dir,
    )
    try:
        profile = mgr.load_cached_profile()
        return {
            "status": "success",
            "message": "Target speaker profile reloaded from cache",
            "profile": profile.to_dict(),
        }
    except Exception as e:
        logger.exception("Failed to reload cached reference: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/voice-cloning/tts")
async def generate_cloned_tts(req: TTSVoiceRequest) -> Dict[str, Any]:
    """Synthesize arbitrary text into speech sounding like the target cloned voice."""
    pipe = get_pipeline()
    from app.conversion.tts_cloner import TextToClonedVoice
    cloner = TextToClonedVoice(
        config=pipe.config.voice_cloning,
        sample_rate=pipe.config.audio.internal_sample_rate,
        cache_dir=pipe.config.voice_cloning.cache_dir,
    )
    try:
        result = await cloner.synthesize_text(
            text=req.text,
            voice_name=req.voice_name or "hi-IN-SwaraNeural",
            pitch_offset_hz=int(req.pitch_offset_hz or 0),
            rate_percent=int(req.rate_percent if req.rate_percent is not None else -3),
            play_audio=bool(req.play_audio),
            output_device_id=pipe.config.audio.output_device,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Cloned TTS generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice-cloning/tts/audio")
def get_cloned_tts_audio():
    """Retrieve the latest synthesized cloned audio WAV file."""
    pipe = get_pipeline()
    out_file = Path(pipe.config.voice_cloning.cache_dir) / "tts_cloned_output.wav"
    if not out_file.exists():
        raise HTTPException(status_code=404, detail="No synthesized audio available yet")
    return FileResponse(
        str(out_file),
        media_type="audio/wav",
        filename="tts_cloned_output.wav",
    )

