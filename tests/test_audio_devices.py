"""Unit tests for audio device discovery and validation."""

import pytest
from app.audio.devices import (
    get_device_samplerate,
    get_input_devices,
    get_output_devices,
    list_all_devices,
    validate_device,
)


def test_list_all_devices():
    devices = list_all_devices()
    assert isinstance(devices, list)
    # The machine should have detected audio devices
    assert len(devices) > 0
    first = devices[0]
    assert hasattr(first, "name")
    assert hasattr(first, "index")
    assert hasattr(first, "is_input")
    assert hasattr(first, "is_output")


def test_get_input_devices():
    inputs = get_input_devices()
    assert isinstance(inputs, list)
    for dev in inputs:
        assert dev.is_input is True
        assert dev.max_input_channels > 0


def test_get_output_devices():
    outputs = get_output_devices()
    assert isinstance(outputs, list)
    assert len(outputs) > 0  # Headphones and speakers detected earlier
    for dev in outputs:
        assert dev.is_output is True
        assert dev.max_output_channels > 0


def test_validate_device():
    # None is valid (system default)
    assert validate_device(None) is True

    # Invalid negative device index
    assert validate_device(-999) is False
