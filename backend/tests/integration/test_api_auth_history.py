"""REST behaviour of auth, history, capabilities and the avatar route, through the real app."""
import numpy as np


def bearer(uid):
    return {"Authorization": f"Bearer good-{uid}"}


def test_me_requires_a_valid_token(make_client):
    c = make_client()
    assert c.get("/api/v1/auth/me").status_code == 401
    assert c.get("/api/v1/auth/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/api/v1/auth/me", headers={"Authorization": "Basic abc"}).status_code == 401
    ok = c.get("/api/v1/auth/me", headers=bearer("u1"))
    assert ok.status_code == 200 and ok.json() == {"id": "u1", "email": "u1@example.com"}


def test_unauthorized_response_shape_and_header(make_client):
    r = make_client().get("/api/v1/auth/me")
    assert r.json()["code"] == "UNAUTHORIZED" and r.headers["www-authenticate"] == "Bearer"


def test_me_reports_auth_not_configured_when_supabase_is_unset(make_client):
    c = make_client(supabase_url="", supabase_publishable_key="")
    r = c.get("/api/v1/auth/me")
    assert r.status_code == 503 and r.json()["code"] == "AUTH_NOT_CONFIGURED"


def test_guest_can_use_avatar_route_but_nothing_is_saved(make_client):
    r = make_client().post("/api/v1/text-to-sign", json={"text": "Hello thank you"})
    assert r.status_code == 200
    assert r.json() == {"gloss": ["HELLO", "THANK_YOU"], "unmatched_words": [], "source": "deterministic", "history_id": None}


def test_signed_in_user_history_roundtrip_with_isolation(make_client):
    c = make_client()
    saved = c.post("/api/v1/text-to-sign", json={"text": "hello water"}, headers=bearer("u1")).json()
    assert saved["history_id"]

    mine = c.get("/api/v1/history", headers=bearer("u1")).json()
    assert mine["total"] == 1
    item = mine["items"][0]
    assert item["direction"] == "text_to_sign" and item["result_type"] == "avatar_gloss"
    assert item["sentence_count"] == 1
    assert item["preview"] == "HELLO WATER"
    assert c.get("/api/v1/history", headers=bearer("u2")).json()["total"] == 0        # isolation

    assert c.delete(f"/api/v1/history/{saved['history_id']}", headers=bearer("u2")).status_code == 404
    assert c.delete(f"/api/v1/history/{saved['history_id']}", headers=bearer("u1")).status_code == 200
    assert c.get("/api/v1/history", headers=bearer("u1")).json()["total"] == 0


def test_save_history_false_and_empty_gloss_are_not_persisted(make_client):
    c = make_client()
    c.post("/api/v1/text-to-sign", json={"text": "hello", "save_history": False}, headers=bearer("u1"))
    c.post("/api/v1/text-to-sign", json={"text": "xyzzy"}, headers=bearer("u1"))
    assert c.get("/api/v1/history", headers=bearer("u1")).json()["total"] == 0


def test_clear_history_and_pagination_validation(make_client):
    c = make_client()
    for _ in range(3):
        c.post("/api/v1/text-to-sign", json={"text": "hello"}, headers=bearer("u1"))
    page = c.get("/api/v1/history?limit=2&offset=0", headers=bearer("u1")).json()
    assert len(page["items"]) == 2 and page["total"] == 3
    assert c.get("/api/v1/history?limit=0", headers=bearer("u1")).status_code == 422
    assert c.delete("/api/v1/history", headers=bearer("u1")).json() == {"deleted": 3}


def test_history_requires_auth_and_reports_disabled(make_client):
    assert make_client().get("/api/v1/history").status_code == 401
    assert make_client().get("/api/v1/history/selfcheck").status_code == 401
    c = make_client(history_enabled="false")
    r = c.get("/api/v1/history", headers=bearer("u1"))
    assert r.status_code == 503 and r.json()["code"] == "HISTORY_UNAVAILABLE"


def test_history_selfcheck_returns_ok_for_signed_in_user(make_client):
    c = make_client()
    assert c.get("/api/v1/history/selfcheck").status_code == 401
    res = c.get("/api/v1/history/selfcheck", headers=bearer("u1"))
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert [s["step"] for s in data["steps"]] == ["session_insert", "sentence_insert", "read_back", "delete"]


def test_auth_required_blocks_guest_translation(make_client):
    c = make_client(auth_required="true")
    assert c.post("/api/v1/text-to-sign", json={"text": "hello"}).status_code == 401
    assert c.post("/api/v1/text-to-sign", json={"text": "hello"}, headers=bearer("u1")).status_code == 200


def test_capabilities_and_models_are_honest_about_the_unpromoted_model(make_client):
    c = make_client(fallback_enabled="false")
    caps = c.get("/api/v1/capabilities").json()
    assert caps["sign_to_text_available"] is False and caps["sign_to_text_unavailable_reason"] == "model_not_promoted"
    assert caps["sign_to_text_source"] == "unavailable" and caps["fallback_vocabulary"] == []
    assert caps["auth_enabled"] is True and caps["history_enabled"] is True
    assert caps["landmark_count"] == 543 and caps["text_to_sign_mode"] == "avatar_gloss"
    assert caps["avatar_vocabulary_size"] == 14
    models = c.get("/api/v1/models").json()["models"][0]
    assert models["promoted"] is False and models["loaded"] is False


def test_capabilities_disclose_the_heuristic_fallback_by_default(make_client):
    caps = make_client().get("/api/v1/capabilities").json()
    assert caps["sign_to_text_available"] is True
    assert caps["sign_to_text_unavailable_reason"] is None
    assert caps["sign_to_text_source"] == "heuristic-fallback"
    assert set(caps["fallback_vocabulary"]) == {"HELLO", "YES", "NO", "YOU", "HELP"}


def test_capabilities_when_promoted_but_bundle_missing(make_client, tmp_path):
    caps = make_client(promoted=True, model_bundle_dir=str(tmp_path / "nope"), fallback_enabled="false").get("/api/v1/capabilities").json()
    assert caps["sign_to_text_unavailable_reason"] == "model_bundle_missing"


def test_promoted_with_missing_bundle_is_a_clean_503_not_a_500(make_client, tmp_path):
    c = make_client(promoted=True, model_bundle_dir=str(tmp_path / "nope"), fallback_enabled="false")
    frames = [{"landmarks": np.zeros((543, 3)).tolist()} for _ in range(3)]
    r = c.post("/api/v1/translation/sign-to-text", json={"frames": frames})
    assert r.status_code == 503 and r.json()["code"] == "MODEL_UNAVAILABLE"


def test_with_a_loadable_bundle_capabilities_models_and_rest_translation(make_client, fake_bundle):
    c = make_client(bundle=fake_bundle, promoted=True)
    assert c.get("/api/v1/capabilities").json()["sign_to_text_available"] is True
    rng = np.random.default_rng(3)
    frames = [{"landmarks": rng.normal(size=(543, 3)).tolist()} for _ in range(20)]

    guest = c.post("/api/v1/translation/sign-to-text", json={"frames": frames}).json()
    assert guest["accepted"] and guest["result_type"] == "model" and guest["history_id"] is None

    user = c.post("/api/v1/translation/sign-to-text", json={"frames": frames}, headers=bearer("u1")).json()
    assert user["history_id"]
    item = c.get("/api/v1/history", headers=bearer("u1")).json()["items"][0]
    assert item["direction"] == "sign_to_text" and item["result_type"] == "model"
    # Confidence lives on the sentence (session level does not average):
    detail = c.get(f"/api/v1/history/{item['id']}", headers=bearer("u1")).json()
    assert detail["sentences"][0]["confidence"] is not None
    assert c.get("/api/v1/health").json()["model_loaded"] is True


def test_ragged_landmarks_are_rejected_with_422_not_a_crash(make_client, fake_bundle):
    c = make_client(bundle=fake_bundle, promoted=True)
    bad = [{"landmarks": [[0.0, 0.0]] * 543} for _ in range(3)]  # 2 coords, not 3
    assert c.post("/api/v1/translation/sign-to-text", json={"frames": bad}).status_code == 422


def test_caller_confidence_floor_returns_unsupported_without_text(make_client, fake_bundle):
    """accepted=false must carry text "" and a reason -- the wire equivalent
    of {status: "uncertain", text: null}. No plausible English on uncertainty."""
    c = make_client(bundle=fake_bundle, promoted=True)
    rng = np.random.default_rng(3)
    frames = [{"landmarks": rng.normal(size=(543, 3)).tolist()} for _ in range(20)]
    body = c.post(
        "/api/v1/translation/sign-to-text",
        json={"frames": frames, "min_confidence": 1.0},
    ).json()
    assert body["accepted"] is False
    assert body["text"] == ""
    assert body["reason"] == "low_model_confidence"


def test_inference_timeout_is_a_clean_503_not_a_hang(make_client, fake_bundle):
    c = make_client(bundle=fake_bundle, promoted=True, inference_timeout_s=0)
    rng = np.random.default_rng(3)
    frames = [{"landmarks": rng.normal(size=(543, 3)).tolist()} for _ in range(20)]
    r = c.post("/api/v1/translation/sign-to-text", json={"frames": frames})
    assert r.status_code == 503 and r.json()["code"] == "SESSION_ERROR"


# ---------------------------------------------------------------- session model

def _compose(c, bearer_uid, words, *, session_key=None, client_key=None):
    body = {"words": words, "confidences": [0.9] * len(words), "save_history": True}
    if session_key:
        body["client_session_key"] = session_key
    if client_key:
        body["client_key"] = client_key
    return c.post("/api/v1/translation/compose", json=body, headers=bearer(bearer_uid))


def test_sentences_of_one_interaction_group_into_one_session(make_client):
    c = make_client()
    assert _compose(c, "u1", ["hello"], session_key="interaction-1", client_key="s-1").json()["history_saved"] is True
    assert _compose(c, "u1", ["water"], session_key="interaction-1", client_key="s-2").json()["history_saved"] is True

    mine = c.get("/api/v1/history", headers=bearer("u1")).json()
    assert mine["total"] == 1
    item = mine["items"][0]
    assert item["sentence_count"] == 2

    detail = c.get(f"/api/v1/history/{item['id']}", headers=bearer("u1")).json()
    assert [s["sequence_number"] for s in detail["sentences"]] == [1, 2]
    assert detail["sentences"][0]["gloss"] == "hello"
    assert detail["sentences"][1]["gloss"] == "water"
    assert detail["sentences"][0]["output_text"]
    assert detail["sentences"][1]["output_text"]

    # A new interaction creates a separate session.
    assert _compose(c, "u1", ["hello"], session_key="interaction-2", client_key="s-3").status_code == 200
    assert c.get("/api/v1/history", headers=bearer("u1")).json()["total"] == 2


def test_repeated_save_of_the_same_sentence_is_idempotent(make_client):
    c = make_client()
    first = _compose(c, "u1", ["hello"], session_key="interaction-dup", client_key="dup-1")
    second = _compose(c, "u1", ["hello"], session_key="interaction-dup", client_key="dup-1")
    assert first.json()["history_saved"] is True and second.json()["history_saved"] is True

    mine = c.get("/api/v1/history", headers=bearer("u1")).json()
    assert mine["total"] == 1
    assert mine["items"][0]["sentence_count"] == 1


def test_session_detail_is_owner_only(make_client):
    c = make_client()
    _compose(c, "u1", ["hello"], session_key="interaction-own", client_key="own-1")
    item = c.get("/api/v1/history", headers=bearer("u1")).json()["items"][0]

    assert c.get(f"/api/v1/history/{item['id']}", headers=bearer("u2")).status_code == 404
    assert c.get(f"/api/v1/history/{item['id']}", headers=bearer("u1")).status_code == 200


def test_bulk_delete_removes_only_the_callers_sessions(make_client):
    c = make_client()
    for i in range(3):
        _compose(c, "u1", ["hello"], session_key=f"bulk-{i}", client_key=f"bulk-s-{i}")
    _compose(c, "u2", ["hello"], session_key="bulk-u2", client_key="bulk-s-u2")

    ids = [item["id"] for item in c.get("/api/v1/history", headers=bearer("u1")).json()["items"]]
    assert len(ids) == 3

    # Mixing in someone else's id deletes only what belongs to the caller.
    victim = c.get("/api/v1/history", headers=bearer("u2")).json()["items"][0]["id"]
    r = c.post(
        "/api/v1/history/bulk-delete",
        json={"ids": ids[:2] + [victim]},
        headers=bearer("u1"),
    )
    assert r.status_code == 200 and r.json()["deleted"] == 2

    assert c.get("/api/v1/history", headers=bearer("u1")).json()["total"] == 1
    assert c.get("/api/v1/history", headers=bearer("u2")).json()["total"] == 1      # untouched

    # Owner of the remaining id can still delete it; the other account's cannot.
    remaining = c.get("/api/v1/history", headers=bearer("u1")).json()["items"][0]["id"]
    assert c.post("/api/v1/history/bulk-delete", json={"ids": [remaining]}, headers=bearer("u2")).json()["deleted"] == 0
    assert c.get("/api/v1/history", headers=bearer("u1")).json()["total"] == 1


def test_bulk_delete_requires_a_nonempty_id_list(make_client):
    c = make_client()
    assert c.post("/api/v1/history/bulk-delete", json={"ids": []}, headers=bearer("u1")).status_code == 422


def test_clear_all_deletes_only_the_authenticated_users_history(make_client):
    c = make_client()
    _compose(c, "u1", ["hello"], session_key="clear-u1-a", client_key="clear-a")
    _compose(c, "u1", ["hello"], session_key="clear-u1-b", client_key="clear-b")
    _compose(c, "u2", ["hello"], session_key="clear-u2-a", client_key="clear-c")

    assert c.delete("/api/v1/history", headers=bearer("u1")).json() == {"deleted": 2}
    assert c.get("/api/v1/history", headers=bearer("u1")).json()["total"] == 0
    assert c.get("/api/v1/history", headers=bearer("u2")).json()["total"] == 1


def test_history_search_direction_sort_and_validation(make_client):
    c = make_client()
    _compose(c, "u1", ["hello"], session_key="filter-a", client_key="filter-a-1")
    c.post("/api/v1/text-to-sign", json={"text": "hello water"}, headers=bearer("u1"))

    r = c.get("/api/v1/history?direction=sign_to_text", headers=bearer("u1")).json()
    assert r["total"] == 1 and r["items"][0]["direction"] == "sign_to_text"
    r = c.get("/api/v1/history?direction=text_to_sign", headers=bearer("u1")).json()
    assert r["total"] == 1 and r["items"][0]["direction"] == "text_to_sign"

    r = c.get("/api/v1/history?q=HELLO", headers=bearer("u1")).json()
    assert r["total"] == 2                                   # case-insensitive substring search
    r = c.get("/api/v1/history?q=zzzznope", headers=bearer("u1")).json()
    assert r["total"] == 0 and r["items"] == []

    r = c.get("/api/v1/history?sort=oldest", headers=bearer("u1")).json()
    assert r["items"][0]["direction"] == "sign_to_text"       # compose ran before the avatar call
    r = c.get("/api/v1/history?sort=newest", headers=bearer("u1")).json()
    assert r["items"][0]["direction"] == "text_to_sign"

    assert c.get("/api/v1/history?direction=banana", headers=bearer("u1")).status_code == 422
    assert c.get("/api/v1/history?sort=banana", headers=bearer("u1")).status_code == 422
    assert c.get("/api/v1/history?limit=0", headers=bearer("u1")).status_code == 422
    assert c.get("/api/v1/history?offset=-1", headers=bearer("u1")).status_code == 422
