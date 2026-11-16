"""POST /api/v1/speech/transcribe — size/type limits, codes, happy path."""
from __future__ import annotations

import io

import pytest

from app.services import stt_service
from app.core.errors import SttUnavailableError


def test_transcribe_happy_path(make_client, monkeypatch):
    async def fake(data, filename, content_type, settings, transport=None):
        assert data == b"RIFFDATA"
        assert content_type == "audio/webm"
        return "hello world"

    monkeypatch.setattr(stt_service, "transcribe_audio", fake)
    c = make_client()
    r = c.post("/api/v1/speech/transcribe", files={"file": ("clip.webm", b"RIFFDATA", "audio/webm")})
    assert r.status_code == 200
    assert r.json() == {"text": "hello world"}


def test_transcribe_rejects_over_1mb(make_client, monkeypatch):
    called = []

    async def fake(data, filename, content_type, settings, transport=None):
        called.append(len(data))
        return "x"

    monkeypatch.setattr(stt_service, "transcribe_audio", fake)
    c = make_client()
    big = b"\x00" * (1_048_576 + 1)
    r = c.post("/api/v1/speech/transcribe", files={"file": ("clip.webm", big, "audio/webm")})
    assert r.status_code == 503
    assert r.json()["code"] == "STT_BAD_AUDIO"
    assert called == []  # rejected before the service ran


def test_transcribe_rejects_disallowed_type(make_client, monkeypatch):
    async def fake(data, filename, content_type, settings, transport=None):
        return "x"

    monkeypatch.setattr(stt_service, "transcribe_audio", fake)
    c = make_client()
    r = c.post("/api/v1/speech/transcribe", files={"file": ("img.png", b"\x89PNG", "image/png")})
    assert r.status_code == 503
    assert r.json()["code"] == "STT_BAD_AUDIO"


def test_transcribe_rejects_empty_file(make_client, monkeypatch):
    async def fake(data, filename, content_type, settings, transport=None):
        return "x"

    monkeypatch.setattr(stt_service, "transcribe_audio", fake)
    c = make_client()
    r = c.post("/api/v1/speech/transcribe", files={"file": ("clip.webm", b"", "audio/webm")})
    assert r.status_code == 503
    assert r.json()["code"] == "STT_BAD_AUDIO"


def test_transcribe_service_failure_maps_code(make_client, monkeypatch):
    async def fake(data, filename, content_type, settings, transport=None):
        raise SttUnavailableError("Groq rate limit reached, try again shortly.", code="STT_RATE_LIMITED")

    monkeypatch.setattr(stt_service, "transcribe_audio", fake)
    c = make_client()
    r = c.post("/api/v1/speech/transcribe", files={"file": ("clip.ogg", b"oggdata", "audio/ogg")})
    assert r.status_code == 503
    assert r.json()["code"] == "STT_RATE_LIMITED"


def test_transcribe_rate_limited_after_10_per_minute(make_client, monkeypatch):
    from app.core import ratelimit

    ratelimit._LIMITERS.clear()  # limiter registry is process-global; isolate this test

    async def fake(data, filename, content_type, settings, transport=None):
        return "ok"

    monkeypatch.setattr(stt_service, "transcribe_audio", fake)
    c = make_client(rate_limit_stt="2/min")
    for i in range(2):
        r = c.post("/api/v1/speech/transcribe", files={"file": ("clip.webm", b"d", "audio/webm")})
        assert r.status_code == 200, i
    r = c.post("/api/v1/speech/transcribe", files={"file": ("clip.webm", b"d", "audio/webm")})
    assert r.status_code == 429
    assert r.json()["code"] == "RATE_LIMITED"
