"""End-to-end integration tests.

Unlike the other test modules (which exercise models/vocab/preprocessing in
isolation), this module boots the *real* FastAPI app -- including the
lifespan handler that loads the actual bundled checkpoints and vocabularies
-- and drives it exactly the way a frontend client would: HTTP requests
against `/api/v1/*` and a real WebSocket session against
`/api/v1/ws/translate`.

This is the test category that was previously missing: every other test
exercised a service or schema directly, so a wiring mistake in main.py,
routes.py, or websocket.py (dependency injection, response shape, CORS,
error codes) could pass the rest of the suite while still being broken for
an actual client. These tests would catch that class of bug.

The whole module shares one app instance (module-scoped fixture) so the
checkpoints are loaded exactly once, not once per test.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


LANDMARK_DIM = 1629


def _frame(seed: int = 0) -> list[float]:
    """A deterministic, small-magnitude fake landmark frame.

    Values are not meant to be anatomically plausible -- these tests check
    plumbing (shapes, routing, error codes), not translation quality, which
    is already documented as unreliable in research/outputs/phase0/RESULTS.md.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(LANDMARK_DIM) * 0.01).astype(float).tolist()


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager runs FastAPI's lifespan handler,
    # which loads the real checkpoints/vocab from disk -- a genuine
    # end-to-end smoke test rather than a mocked one.
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


def test_root_advertises_websocket_path(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["websocket"] == "/api/v1/ws/translate"


def test_health_reports_models_loaded(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["device"] in {"cpu", "cuda", "cuda:0"}


def test_models_inventory_excludes_gru_baseline(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    body = response.json()["models"]
    names = {item["name"] for item in body}
    task_by_name = {item["name"]: item["task"] for item in body}

    # "text_to_pose_gru" legitimately contains "gru" -- it IS a GRU and it
    # IS served. What must never be true is a *sign_to_text* model backed by
    # the GRU sanity baseline: it fully mode-collapsed in evaluation (0.4%
    # unique outputs across 1,500 test rows -- see
    # research/outputs/phase0/RESULTS.md), so it is excluded from serving.
    assert "transformer_full" in names
    assert "text_to_pose_gru" in names
    assert "gru_baseline" not in names

    sign_to_text_models = {n for n, t in task_by_name.items() if t == "sign_to_text"}
    assert sign_to_text_models <= {"transformer_full", "transformer_no_face"}


def test_models_endpoint_surfaces_research_provenance(client):
    """The knowledge-repository manifest (signsage_knowledge_manifest.json)
    is loaded into memory at startup specifically to be 'consumed by the
    backend for model/version metadata and research transparency' (its
    documented purpose). This asserts that data is actually reachable via
    the API, not just parsed and discarded."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    provenance = response.json()["provenance"]
    assert provenance is not None
    assert provenance["run_id"]
    assert provenance["phase_0_status"] == "GO-CANDIDATE"
    assert "not automatically" in provenance["warning"]


def test_models_endpoint_surfaces_pose_normalization_stats(client):
    """`POSE_NORMALIZATION_STATS` mirrors the notebook's own counter, which
    research/outputs/phase0/RESULTS.md flags as tracked-but-never-surfaced.
    This asserts the backend actually exposes it (shape/keys), rather than
    silently accumulating counts nobody can read."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    stats = response.json()["pose_normalization"]
    assert set(stats.keys()) == {
        "total_sequences",
        "bad_center_sequences",
        "degenerate_scale_sequences",
    }
    assert all(isinstance(v, int) and v >= 0 for v in stats.values())


def test_pose_normalization_stats_increment_in_raw_mode():
    """Confirms the counter this endpoint reports is actually live: running
    normalize_pose_sequence (the raw-mode path) must move the counter."""
    from app.services.pose_preprocessing import (
        POSE_NORMALIZATION_STATS,
        normalize_pose_sequence,
    )

    before = POSE_NORMALIZATION_STATS.snapshot()["total_sequences"]
    normalize_pose_sequence(np.zeros((4, 543, 3), dtype=np.float32))
    after = POSE_NORMALIZATION_STATS.snapshot()["total_sequences"]
    assert after == before + 1


def test_capabilities_endpoint_reports_contract_and_limits(client):
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    body = response.json()

    assert body["websocket_endpoint"] == "/api/v1/ws/translate"
    assert set(body["message_types"]["client"]) == {
        "landmarks.frame",
        "sentence.boundary",
        "text.input",
    }
    assert set(body["message_types"]["server"]) == {
        "translation.text",
        "translation.pose",
        "error",
    }

    limits = body["limits"]
    assert limits["max_frames"] == get_settings().max_frames
    assert limits["pose_fps"] == 25

    # No REST endpoint exists for the core translation loop -- WebSocket is
    # the sole transport (see docs/API.md and DECISIONS.md).
    assert client.post("/api/v1/sign-to-text", json={}).status_code == 404
    assert client.post("/api/v1/text-to-pose", json={}).status_code == 404


def test_capabilities_reflects_disabled_gloss_providers_by_default(client):
    """Groq/Ollama are both disabled in .env.example; the capabilities
    endpoint must report that accurately rather than always claiming a
    feature is on."""
    response = client.get("/api/v1/capabilities")
    features = response.json()["features"]
    assert features["groq_gloss_enabled"] is False
    assert features["ollama_gloss_enabled"] is False
    assert features["landmark_input_mode"] == "normalized"


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


def test_ws_sign_to_text_round_trip(client):
    with client.websocket_connect("/api/v1/ws/translate") as ws:
        session_id = "test-session-sign"
        for i in range(6):
            ws.send_json(
                {
                    "type": "landmarks.frame",
                    "session_id": session_id,
                    "timestamp_ms": i * 33,
                    "landmarks": _frame(i),
                }
            )
        ws.send_json(
            {
                "type": "sentence.boundary",
                "session_id": session_id,
                "trigger": "manual",
            }
        )
        response = ws.receive_json()

    assert response["type"] == "translation.text"
    assert response["session_id"] == session_id
    assert isinstance(response["text"], str)
    assert response["latency_ms"] >= 0


def test_ws_text_to_pose_round_trip(client):
    with client.websocket_connect("/api/v1/ws/translate") as ws:
        session_id = "test-session-pose"
        ws.send_json(
            {
                "type": "text.input",
                "session_id": session_id,
                "text": "thank you very much",
            }
        )
        response = ws.receive_json()

    assert response["type"] == "translation.pose"
    assert response["session_id"] == session_id
    assert response["fps"] == 25
    assert len(response["frames"]) > 0
    assert len(response["frames"][0]) == LANDMARK_DIM
    assert response["latency_ms"] >= 0


def test_ws_unknown_message_type_returns_structured_error(client):
    with client.websocket_connect("/api/v1/ws/translate") as ws:
        ws.send_json({"type": "bogus", "session_id": "s"})
        response = ws.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "UNKNOWN_MESSAGE_TYPE"


def test_ws_invalid_json_returns_structured_error(client):
    with client.websocket_connect("/api/v1/ws/translate") as ws:
        ws.send_text("not json")
        response = ws.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "INVALID_JSON"


def test_ws_session_id_change_mid_connection_is_rejected(client):
    with client.websocket_connect("/api/v1/ws/translate") as ws:
        # First frame silently locks in the session (no reply expected).
        ws.send_json(
            {
                "type": "landmarks.frame",
                "session_id": "session-a",
                "timestamp_ms": 0,
                "landmarks": _frame(0),
            }
        )
        # Second frame under a different session_id on the same connection
        # must be rejected -- one WebSocket connection is one session (see
        # app/services/session.py).
        ws.send_json(
            {
                "type": "landmarks.frame",
                "session_id": "session-b",
                "timestamp_ms": 33,
                "landmarks": _frame(1),
            }
        )
        response = ws.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "SESSION_ERROR"


def test_ws_unexpected_field_is_rejected(client):
    # `extra="forbid"` on every WebSocket schema -- a client typo should be
    # a loud structured error, not a silently-ignored field.
    with client.websocket_connect("/api/v1/ws/translate") as ws:
        ws.send_json(
            {
                "type": "text.input",
                "session_id": "s",
                "text": "hi",
                "unexpected": True,
            }
        )
        response = ws.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "INVALID_MESSAGE"


def test_ws_sentence_boundary_without_frames_is_rejected(client):
    with client.websocket_connect("/api/v1/ws/translate") as ws:
        ws.send_json(
            {
                "type": "sentence.boundary",
                "session_id": "session-empty",
                "trigger": "manual",
            }
        )
        response = ws.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "SESSION_ERROR"


def test_ws_rejects_malformed_landmarks_frame(client):
    with client.websocket_connect("/api/v1/ws/translate") as ws:
        ws.send_json(
            {
                "type": "landmarks.frame",
                "session_id": "s",
                "timestamp_ms": 0,
                "landmarks": [0.0] * 10,  # wrong length: must be exactly 1629
            }
        )
        response = ws.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "INVALID_MESSAGE"


def test_ws_connection_survives_a_bad_message(client):
    """A rejected message must not kill the connection -- the client should
    be able to keep talking on the same session afterward."""
    with client.websocket_connect("/api/v1/ws/translate") as ws:
        ws.send_json({"type": "bogus", "session_id": "s"})
        error = ws.receive_json()
        assert error["type"] == "error"

        ws.send_json(
            {
                "type": "text.input",
                "session_id": "s",
                "text": "still alive",
            }
        )
        response = ws.receive_json()

    assert response["type"] == "translation.pose"
