"""Automated tests for Voice Cloning API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.audio.pipeline import AudioPipeline
from app.config import AppConfig
from app.main import app


@pytest.fixture
def client():
    # Ensure fresh test client
    return TestClient(app)


def test_api_voice_cloning_status(client):
    response = client.get("/voice-cloning/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "message" in data
    assert "fingerprint" in data
    assert "reference_dir" in data
    assert data["reference_dir"] == "voice-clone"


def test_api_voice_cloning_prepare_reference(client):
    response = client.post("/voice-cloning/prepare-reference")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "profile" in data
    assert data["profile"]["total_clips"] >= 3
    assert data["profile"]["f0_median"] > 100.0


def test_api_voice_cloning_reload_reference(client):
    response = client.post("/voice-cloning/reload-reference")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "profile" in data


def test_api_update_settings_voice_cloning_mode(client):
    response = client.put("/settings", json={"mode": "voice_cloning", "voice_clone_pitch_scale": 1.05})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["config"]["mode"] == "voice_cloning"
    assert data["config"]["voice_cloning"]["target_pitch_scale"] == 1.05

    # Switch back to dsp to preserve default state
    client.put("/settings", json={"mode": "dsp"})


def test_api_update_settings_invalid_mode_rejected(client):
    response = client.put("/settings", json={"mode": "invalid_mode_xyz"})
    assert response.status_code == 400
    data = response.json()
    assert "Invalid mode" in data["detail"]


def test_api_path_traversal_rejected(client):
    response = client.put("/settings", json={"voice_clone_ref_dir": "../../../etc/passwd"})
    assert response.status_code == 400
    data = response.json()
    assert "Invalid reference directory" in data["detail"]
