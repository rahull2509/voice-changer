"""Unit and integration tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.api.routes import set_pipeline
from app.audio.pipeline import AudioPipeline
from app.config import AppConfig
from app.main import app


@pytest.fixture(scope="module")
def client():
    cfg = AppConfig()
    pipeline = AudioPipeline(cfg)
    set_pipeline(pipeline)
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "mode" in data
    assert data["mode"] == "dsp"


def test_devices_endpoint(client):
    res = client.get("/audio/devices")
    assert res.status_code == 200
    data = res.json()
    assert "input_devices" in data
    assert "output_devices" in data
    assert isinstance(data["input_devices"], list)
    assert isinstance(data["output_devices"], list)


def test_models_endpoint(client):
    res = client.get("/voice/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert len(data["models"]) >= 3
    assert any(m["id"] == "dsp_indian_female" for m in data["models"])


def test_settings_get_and_put(client):
    res = client.get("/settings")
    assert res.status_code == 200
    cfg = res.json()
    assert "dsp" in cfg

    # Update setting
    res_update = client.put(
        "/settings",
        json={"dsp_pitch_factor": 1.70, "dsp_formant_shift": 1.20, "dsp_output_gain": 0.90},
    )
    assert res_update.status_code == 200
    data = res_update.json()
    assert data["status"] == "success"

    # Verify updated
    res2 = client.get("/settings")
    cfg2 = res2.json()
    assert cfg2["dsp"]["pitch_factor"] == 1.70
    assert cfg2["dsp"]["formant_shift"] == 1.20


def test_audio_status_endpoint(client):
    res = client.get("/audio/status")
    assert res.status_code == 200
    data = res.json()
    assert "is_running" in data
    assert "processing_time_ms" in data
    assert "total_latency_ms" in data
    assert "real_time_factor" in data
    assert "input_rms" in data
    assert "output_rms" in data
