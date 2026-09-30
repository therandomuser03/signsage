"""WebSocket: auth-in-first-message, history ids, and hostile-input handling."""
import json

import numpy as np
import pytest


def _frames(ws, n=25, seed=5):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        ws.send_json({"type": "landmarks.frame", "landmarks": rng.normal(size=(543, 3)).tolist()})


def test_authenticated_socket_saves_history_and_returns_its_id(make_client, fake_bundle):
    c = make_client(bundle=fake_bundle, promoted=True)
    with c.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "auth", "access_token": "good-u1"})
        assert ws.receive_json() == {"type": "auth.ok", "user_id": "u1"}
        _frames(ws)
        ws.send_json({"type": "sentence.boundary"})
        msg = ws.receive_json()
    assert msg["type"] == "translation.text" and msg["result_type"] == "model" and msg["history_id"]
    items = c.get("/api/v1/history", headers={"Authorization": "Bearer good-u1"}).json()["items"]
    assert items[0]["id"] == msg["history_id"]


def test_guest_socket_translates_without_history(make_client, fake_bundle):
    c = make_client(bundle=fake_bundle, promoted=True)
    with c.websocket_connect("/ws/translate") as ws:
        _frames(ws)
        ws.send_json({"type": "sentence.boundary"})
        msg = ws.receive_json()
    assert msg["accepted"] and msg["history_id"] is None


def test_save_history_false_is_honoured(make_client, fake_bundle):
    c = make_client(bundle=fake_bundle, promoted=True)
    with c.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "auth", "access_token": "good-u1"})
        ws.receive_json()
        _frames(ws)
        ws.send_json({"type": "sentence.boundary", "save_history": False})
        assert ws.receive_json()["history_id"] is None


def test_bad_token_gets_unauthorized_and_socket_stays_usable(make_client, fake_bundle):
    c = make_client(bundle=fake_bundle, promoted=True)
    with c.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "auth", "access_token": "bad"})
        assert ws.receive_json()["code"] == "UNAUTHORIZED"
        _frames(ws)
        ws.send_json({"type": "sentence.boundary"})
        assert ws.receive_json()["type"] == "translation.text"     # still works as a guest


def test_auth_required_needs_auth_before_translating(make_client, fake_bundle):
    c = make_client(bundle=fake_bundle, promoted=True, auth_required="true")
    with c.websocket_connect("/ws/translate") as ws:
        _frames(ws)
        ws.send_json({"type": "sentence.boundary"})
        assert ws.receive_json()["code"] == "UNAUTHORIZED"


@pytest.mark.parametrize("payload,code", [
    ("not json at all", "INVALID_JSON"),
    (json.dumps([1, 2, 3]), "INVALID_MESSAGE"),
    (json.dumps({"type": "nope"}), "UNKNOWN_MESSAGE_TYPE"),
    (json.dumps({"type": "landmarks.frame", "landmarks": [[0, 0, 0]]}), "INVALID_MESSAGE"),
    (json.dumps({"type": "landmarks.frame", "landmarks": [[0, 0]] * 543}), "INVALID_MESSAGE"),
    (json.dumps({"type": "auth"}), "INVALID_MESSAGE"),
])
def test_hostile_or_malformed_messages_get_a_coded_error_and_do_not_kill_the_socket(make_client, fake_bundle, payload, code):
    c = make_client(bundle=fake_bundle, promoted=True)
    with c.websocket_connect("/ws/translate") as ws:
        ws.send_text(payload)
        assert ws.receive_json()["code"] == code
        ws.send_json({"type": "sentence.boundary"})            # socket must still answer
        assert ws.receive_json()["code"] == "EMPTY_SEQUENCE"


def test_oversized_message_is_rejected(make_client, fake_bundle):
    c = make_client(bundle=fake_bundle, promoted=True, max_message_bytes=1000)
    with c.websocket_connect("/ws/translate") as ws:
        ws.send_text(json.dumps({"type": "landmarks.frame", "landmarks": [[0.5, 0.5, 0.5]] * 543}))
        assert ws.receive_json()["code"] == "MESSAGE_TOO_LARGE"


def test_unpromoted_model_closes_with_model_unavailable(make_client):
    c = make_client(promoted=False)
    with c.websocket_connect("/ws/translate") as ws:
        assert ws.receive_json()["code"] == "MODEL_UNAVAILABLE"
