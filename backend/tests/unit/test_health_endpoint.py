"""Regression tests for the health endpoint reliability."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_responds_promptly_without_model_load(client):
    """Health must not hang on model download or model initialization."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_promoted"] is False
    assert "model_bundle_found" in data
    assert "model_loaded" in data


def test_health_does_not_expose_secrets(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    # Should not leak bundle paths, keys, or filesystem details.
    for forbidden in ("SIGNSAGE", "bundle_dir", "secret", "key", "password", "token"):
        json_str = str(data).lower()
        assert forbidden not in json_str, f"Health leaked forbidden term: {forbidden}"
