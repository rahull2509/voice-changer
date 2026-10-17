"""Audio device enumeration and validation using sounddevice."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    """Describes a single audio device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi_name: str
    is_input: bool
    is_output: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "max_input_channels": self.max_input_channels,
            "max_output_channels": self.max_output_channels,
            "default_samplerate": self.default_samplerate,
            "hostapi_name": self.hostapi_name,
            "is_input": self.is_input,
            "is_output": self.is_output,
        }


def list_all_devices() -> List[AudioDevice]:
    """Return every audio device visible to PortAudio."""
    devices: List[AudioDevice] = []
    try:

        hostapis = sd.query_hostapis()
        raw_devices = sd.query_devices()
        # query_devices returns a single dict when there's only one device
        if isinstance(raw_devices, dict):
            raw_devices = [raw_devices]
        for i, dev in enumerate(raw_devices):
            ha = hostapis[dev["hostapi"]]
            devices.append(
                AudioDevice(
                    index=i,
                    name=dev["name"],
                    max_input_channels=dev["max_input_channels"],
                    max_output_channels=dev["max_output_channels"],
                    default_samplerate=dev["default_samplerate"],
                    hostapi_name=ha["name"],
                    is_input=dev["max_input_channels"] > 0,
                    is_output=dev["max_output_channels"] > 0,
                )
            )
    except Exception as e:
        logger.error(f"Failed to enumerate audio devices: {e}")
    return devices


def get_input_devices() -> List[AudioDevice]:
    """Return only devices that can capture audio."""
    return [d for d in list_all_devices() if d.is_input]


def get_output_devices() -> List[AudioDevice]:
    """Return only devices that can play audio."""
    return [d for d in list_all_devices() if d.is_output]


def validate_device(
    device_id: Optional[int],
    need_input: bool = False,
    need_output: bool = False,
) -> bool:
    """Check that *device_id* exists and has the requested capabilities."""
    if device_id is None:
        return True  # system default is always valid
    if device_id < 0:
        return False
    try:
        info = sd.query_devices(device_id)
        if need_input and info["max_input_channels"] == 0:
            return False
        if need_output and info["max_output_channels"] == 0:
            return False
        return True
    except Exception:
        return False


def is_device_openable(device_id: int, is_input: bool) -> bool:
    """Test if a device can actually open a stream on Windows without OS error."""
    try:
        info = sd.query_devices(device_id)
        if is_input:
            if info["max_input_channels"] <= 0:
                return False
            ch = int(info["max_input_channels"])
            sr = int(info["default_samplerate"])
            s = sd.InputStream(device=device_id, channels=ch, samplerate=sr, callback=lambda *a: None)
            s.start()
            s.stop()
            s.close()
            return True
        else:
            if info["max_output_channels"] <= 0:
                return False
            ch = min(int(info["max_output_channels"]), 2)
            sr = int(info["default_samplerate"])
            s = sd.OutputStream(device=device_id, channels=ch, samplerate=sr, callback=lambda outdata, *a: outdata.fill(0))
            s.start()
            s.stop()
            s.close()
            return True
    except Exception:
        return False


def resolve_input_device(device_id: Optional[int]) -> Optional[int]:
    """Resolve input device ID without terminating running streams."""
    if device_id is not None and validate_device(device_id, need_input=True):
        return device_id
    try:
        def_in = sd.default.device[0]
        if def_in >= 0 and validate_device(def_in, need_input=True):
            return def_in
    except Exception:
        pass
    inputs = get_input_devices()
    for d in inputs:
        if d.is_input and d.max_input_channels > 0:
            return d.index
    return None


def resolve_output_device(device_id: Optional[int]) -> Optional[int]:
    """Resolve output device ID without terminating running streams."""
    if device_id is not None and validate_device(device_id, need_output=True):
        return device_id
    try:
        def_out = sd.default.device[1]
        if def_out >= 0 and validate_device(def_out, need_output=True):
            return def_out
    except Exception:
        pass
    outputs = get_output_devices()
    for d in outputs:
        if d.is_output and d.max_output_channels > 0:
            return d.index
    return None


def get_device_samplerate(device_id: Optional[int], is_input: bool) -> int:
    """Return the default sample rate for *device_id*."""
    try:
        dev_id = resolve_input_device(device_id) if is_input else resolve_output_device(device_id)
        if dev_id is not None:
            info = sd.query_devices(dev_id)
            return int(info["default_samplerate"])
        return 44100
    except Exception:
        return 44100
