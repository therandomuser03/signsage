"""History write paths against a mocked PostgREST layer.

Every path that saves history goes through HistoryService.try_record_sentence ->
SupabaseTranslationRepository.record_sentence: get-or-create ONE
translation_sessions row for the interaction, then one translation_messages
row per completed sentence, always on the caller's JWT. Idempotency is by
client keys (session scope + per-sentence key).
"""
import json
import logging

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import HistoryUnavailableError, NotFoundError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistorySentence, ResultType
from app.infrastructure.database.repositories.translation_repository import (
    SupabaseTranslationRepository,
    _session_to_record,
)
from app.infrastructure.database.supabase import SupabaseRestClient
from app.services.history_service import HistoryService

USER = AuthenticatedUser(id="u1", email="a@b.c", access_token="user-jwt")
REC = NewHistorySentence(
    Direction.SIGN_TO_TEXT, "Hello, help, me.", ResultType.MODEL, "word_v3_8w_s123",
    input_text="hello help me", gloss="hello help me", confidence=0.8,
    title="Word recognition: Hello, help, me.", status="completed",
)
REC_KEYS = NewHistorySentence(
    Direction.SIGN_TO_TEXT, "Hello.", ResultType.MODEL, "m",
    session_key="interaction-1", client_key="sentence-1", title="Word recognition: Hello.",
)

EMBED_SELECT = (
    "id,user_id,title,direction,status,created_at,ended_at,updated_at,"
    "translation_messages(sequence,id,session_id,direction,input_text,output_text,gloss,"
    "confidence,model_version,result_type,created_at)"
)


def _repo(handler):
    settings = Settings(supabase_url="https://x.supabase.co", supabase_publishable_key="pub")
    return SupabaseTranslationRepository(SupabaseRestClient(settings, transport=httpx.MockTransport(handler)))


def _service(handler):
    return HistoryService(_repo(handler))


def _table(req):
    return str(req.url).rsplit("/rest/v1/", 1)[1].split("?", 1)[0]


def _msg(seq=1, text="Hello."):
    return {
        "sequence": seq, "id": f"m-{seq}", "session_id": "s-1", "direction": "sign_to_text",
        "input_text": "hi", "output_text": text, "gloss": "hi", "confidence": 0.9,
        "model_version": "mv", "result_type": "model",
        "created_at": "2026-01-02T00:00:00+00:00",
    }


def _full_handler(calls):
    def handler(req):
        table = _table(req)
        params = dict(req.url.params)
        calls.append({
            "table": table,
            "method": req.method,
            "params": params,
            "auth": req.headers["authorization"],
            "apikey": req.headers["apikey"],
            "prefer": req.headers.get("prefer"),
            "body": json.loads(req.content) if req.content else None,
        })
        if req.method == "PATCH":
            return httpx.Response(200, json=[{"id": "s-1"}])
        if table == "translation_sessions":
            return httpx.Response(201, json=[{"id": "s-1"}])
        if req.method == "GET":
            return httpx.Response(200, json=[])  # sequence lookup / sentence lookup
        return httpx.Response(201, json=[{"id": "m-1"}])
    return handler


async def test_record_sentence_creates_session_then_sentence_with_columns_and_headers():
    calls = []
    session_id = await _repo(_full_handler(calls)).record_sentence(USER, REC)

    assert session_id[0] == "s-1"
    tables = [c["table"] for c in calls]
    assert tables == ["translation_sessions", "translation_messages", "translation_messages", "translation_sessions"]
    assert calls[0]["method"] == "POST"
    # sequence lookup happens on the messages table before the insert
    assert calls[1] == {**calls[1], "method": "GET"}
    assert calls[1]["params"]["select"] == "sequence"
    assert calls[2]["method"] == "POST"
    assert all(c["auth"] == "Bearer user-jwt" for c in calls)
    assert all(c["apikey"] == "pub" for c in calls)
    # POSTs ask for the inserted row back; the sequence GET carries no Prefer
    assert [c["prefer"] for c in calls] == ["return=representation", None, "return=representation", "return=minimal"]

    session_body, message_body = calls[0]["body"], calls[2]["body"]
    assert session_body == {
        "user_id": "u1",
        "direction": "sign_to_text",
        "status": "completed",
        "title": "Word recognition: Hello, help, me.",
        "result_type": "model",
        "model_version": "word_v3_8w_s123",
        "input_text": "hello help me",
        "output_text": "Hello, help, me.",
        "confidence": 0.8,
        "client_key": None,
    }
    assert message_body == {
        "session_id": "s-1",
        "sequence": 1,
        "direction": "sign_to_text",
        "input_text": "hello help me",
        "output_text": "Hello, help, me.",
        "gloss": "hello help me",
        "confidence": 0.8,
        "model_version": "word_v3_8w_s123",
        "result_type": "model",
        "client_key": None,
    }


async def test_record_sentence_with_keys_looks_up_session_and_sentence_first():
    calls = []

    def handler(req):
        calls.append((_table(req), req.method, dict(req.url.params)))
        table = _table(req)
        if req.method == "PATCH":
            return httpx.Response(200, json=[{"id": "s-existing"}])
        if table == "translation_sessions" and req.method == "GET":
            return httpx.Response(200, json=[{"id": "s-existing"}])
        if table == "translation_messages" and req.method == "GET":
            return httpx.Response(200, json=[])  # no sentence with this key yet
        if table == "translation_sessions":
            return httpx.Response(201, json=[{"id": "s-new"}])
        return httpx.Response(201, json=[{"id": "m-1"}])

    session_id, sentence_id, duplicated = await _repo(handler).record_sentence(USER, REC_KEYS)

    assert (session_id, sentence_id, duplicated) == ("s-existing", "m-1", False)
    # get-or-create probe ran; no new session was created; sentence was inserted into the existing one
    assert calls[0] == ("translation_sessions", "GET", {"select": "id", "user_id": "eq.u1", "client_key": "eq.interaction-1", "limit": "1"})
    assert all(c[:2] != ("translation_sessions", "POST") for c in calls)
    msg_insert = [c for c in calls if c[0] == "translation_messages" and c[1] == "POST"]
    assert len(msg_insert) == 1


async def test_record_sentence_recovers_from_unique_key_race():
    state = {"session_gets": 0}

    def handler(req):
        table = _table(req)
        if table == "translation_sessions" and req.method == "GET":
            state["session_gets"] += 1
            # The first probe misses (the row is not visible yet); after our
            # insert loses the race to a peer, the winning row is visible.
            return httpx.Response(200, json=[] if state["session_gets"] == 1 else [{"id": "s-winner"}])
        if table == "translation_sessions" and req.method == "POST":
            # Lost the race: the peer's row holds the unique (user_id, client_key).
            return httpx.Response(409, json={"code": "23505", "message": "duplicate key value"})
        if table == "translation_messages" and req.method == "GET":
            return httpx.Response(200, json=[])
        if table == "translation_messages" and req.method == "POST":
            return httpx.Response(201, json=[{"id": "m-1"}])
        if req.method == "PATCH":
            return httpx.Response(200, json=[{"id": "s-winner"}])
        return httpx.Response(200, json=[])

    session_id, _, _ = await _repo(handler).record_sentence(USER, REC_KEYS)
    assert session_id == "s-winner"


async def test_duplicate_sentence_key_is_idempotent():
    def handler(req):
        table = _table(req)
        if req.method == "PATCH":
            return httpx.Response(200, json=[{"id": "s-1"}])
        if table == "translation_sessions":
            return httpx.Response(200, json=[{"id": "s-1"}])
        if table == "translation_messages" and req.method == "GET":
            # Only the per-sentence lookup reports the existing row.
            params = dict(req.url.params)
            if params.get("client_key") == "eq.sentence-1":
                return httpx.Response(200, json=[{"id": "m-1"}])
            return httpx.Response(200, json=[])
        if table == "translation_messages" and req.method == "POST":
            raise AssertionError("duplicate sentence must not be inserted")
        return httpx.Response(200, json=[])

    session_id, sentence_id, duplicated = await _repo(handler).record_sentence(USER, REC_KEYS)
    assert (session_id, sentence_id, duplicated) == ("s-1", "m-1", True)


async def test_session_insert_failure_logs_status_body_and_path_without_secrets(caplog):
    def handler(req):
        return httpx.Response(
            400,
            json={"code": "23502", "message": 'null value in column "user_id" violates not-null constraint'},
        )

    with caplog.at_level(logging.WARNING):
        result = await _service(handler).try_record_sentence(USER, REC)

    assert result is None
    assert any(
        "session insert" in r.getMessage() and "status=400" in r.getMessage() and "pg_code=23502" in r.getMessage()
        for r in caplog.records
    ), caplog.text
    assert any("/rest/v1/translation_sessions" in r.getMessage() for r in caplog.records), caplog.text
    assert "user-jwt" not in caplog.text
    assert "Hello, help, me." not in caplog.text


async def test_sentence_insert_failure_logs_step_status_and_path_without_secrets(caplog):
    def handler(req):
        table = _table(req)
        if table == "translation_sessions":
            if req.method == "GET":
                return httpx.Response(200, json=[])
            if req.method == "PATCH":
                return httpx.Response(200, json=[])
            return httpx.Response(201, json=[{"id": "s-1"}])
        if req.method == "GET":
            return httpx.Response(200, json=[])
        return httpx.Response(
            403,
            json={"code": "42501", "message": "row-level security policy violated for table translation_messages"},
        )

    with caplog.at_level(logging.WARNING):
        result = await _service(handler).try_record_sentence(USER, REC)

    assert result is None
    assert any(
        "sentence insert" in r.getMessage() and "status=403" in r.getMessage() and "pg_code=42501" in r.getMessage()
        for r in caplog.records
    ), caplog.text
    assert any("/rest/v1/translation_messages" in r.getMessage() for r in caplog.records), caplog.text
    assert "user-jwt" not in caplog.text
    assert "Hello, help, me." not in caplog.text


async def test_expired_jwt_logs_401_without_the_token(caplog):
    def handler(req):
        return httpx.Response(401)

    with caplog.at_level(logging.WARNING):
        result = await _service(handler).try_record_sentence(USER, REC)

    assert result is None
    assert any("session insert" in r.getMessage() and "401" in r.getMessage() for r in caplog.records), caplog.text
    assert "user-jwt" not in caplog.text


async def test_list_falls_back_to_two_requests_when_embed_fk_is_missing(caplog):
    session_row = {
        "id": "s-1", "user_id": "u1", "direction": "sign_to_text", "status": "completed",
        "title": "t", "created_at": "2026-01-02T00:00:00+00:00",
        "ended_at": None, "updated_at": "2026-01-02T00:00:00+00:00",
    }
    messages = [_msg(1, "Hi.")]
    seen = []

    def handler(req):
        params = dict(req.url.params)
        seen.append((_table(req), params))
        if params.get("select", "") == EMBED_SELECT:  # embed attempt
            return httpx.Response(400, json={"code": "PGRST200", "message": "Could not find a relationship between the tables"})
        if _table(req) == "translation_sessions":
            return httpx.Response(200, json=[session_row], headers={"content-range": "0-0/1"})
        return httpx.Response(200, json=messages)

    with caplog.at_level(logging.WARNING):
        rows, total = await _repo(handler).list_sessions(USER, 5, 0)

    assert total == 1
    assert len(rows) == 1
    assert rows[0].id == "s-1"
    assert rows[0].sentences[0].output_text == "Hi." and rows[0].sentences[0].confidence == 0.9
    assert seen[0][0] == "translation_sessions"
    assert seen[0][1]["select"] == EMBED_SELECT
    assert seen[1][0] == "translation_sessions"      # fallback: page of sessions
    assert seen[2][0] == "translation_messages"      # ... then its messages
    assert seen[2][1]["session_id"] == "in.(s-1)"
    assert any("embed failed" in r.getMessage() for r in caplog.records), caplog.text


async def test_list_does_not_fall_back_on_unrelated_errors():
    def handler(req):
        return httpx.Response(500, json={"code": "XX000", "message": "internal error"})

    with pytest.raises(HistoryUnavailableError):
        await _repo(handler).list_sessions(USER, 5, 0)


async def test_list_with_working_embed_keeps_single_request():
    def handler(req):
        return httpx.Response(200, json=[{
            "id": "s-1", "user_id": "u1", "direction": "sign_to_text", "status": "completed",
            "title": "t", "created_at": "2026-01-02T00:00:00+00:00",
            "ended_at": None, "updated_at": "2026-01-02T00:00:00+00:00",
            "translation_messages": [_msg(1, "A")],
        }], headers={"content-range": "0-0/7"})

    rows, total = await _repo(handler).list_sessions(USER, 1, 0)
    assert total == 7
    assert rows[0].sentences[0].output_text == "A"


async def test_list_orders_oldest_when_asked():
    seen = {}

    def handler(req):
        seen.update(dict(req.url.params))
        return httpx.Response(200, json=[], headers={"content-range": "*/0"})

    await _repo(handler).list_sessions(USER, 5, 0, order="oldest")
    assert seen["order"] == "created_at.asc"


async def test_delete_removes_sentences_then_session():
    calls = []

    def handler(req):
        calls.append((_table(req), dict(req.url.params)))
        return httpx.Response(200, json=[{"id": "s-1"}])

    ok = await _repo(handler).delete_session(USER, "s-1")
    assert ok is True
    assert calls == [
        ("translation_messages", {"session_id": "eq.s-1"}),
        ("translation_sessions", {"id": "eq.s-1", "user_id": "eq.u1"}),
    ]


async def test_delete_missing_session_raises_not_found_at_service_layer():
    service = _service(lambda req: httpx.Response(200, json=[]))
    with pytest.raises(NotFoundError):
        await service.delete(USER, "nope")


async def test_delete_many_deletes_owned_chunks_and_counts_sessions():
    calls = []

    def handler(req):
        calls.append((_table(req), dict(req.url.params)))
        if _table(req) == "translation_sessions" and req.method == "DELETE":
            # Only one of the three belongs to the caller (RLS + user_id filter).
            return httpx.Response(200, json=[{"id": "s-1"}])
        return httpx.Response(200, json=[])

    deleted = await _repo(handler).delete_sessions(USER, ["s-1", "s-2", "s-3", "s-1"])
    assert deleted == 1
    msg_delete, session_delete = calls
    assert msg_delete[0] == "translation_messages" and msg_delete[1]["session_id"] == "in.(s-1,s-2,s-3)"
    assert session_delete[0] == "translation_sessions" and session_delete[1]["id"] == "in.(s-1,s-2,s-3)"
    assert session_delete[1]["user_id"] == "eq.u1"


async def test_clear_removes_everything_the_user_owns():
    calls = []

    def handler(req):
        calls.append((_table(req), dict(req.url.params)))
        if _table(req) == "translation_sessions" and req.method == "GET":
            return httpx.Response(200, json=[{"id": "s-1"}, {"id": "s-2"}])
        return httpx.Response(200, json=[{"id": "x"}, {"id": "y"}])

    deleted = await _repo(handler).clear(USER)
    assert deleted == 2
    assert calls[0][0] == "translation_sessions"
    assert calls[1] == ("translation_messages", {"session_id": "in.(s-1,s-2)"})
    assert calls[2][0] == "translation_sessions" and calls[2][1]["user_id"] == "eq.u1"


def test_session_to_record_mapping():
    row = {
        "id": "r1", "user_id": "u1", "direction": "sign_to_text",
        "title": "orphan", "status": "completed", "created_at": "2026-01-01T00:00:00+00:00",
        "ended_at": None, "updated_at": None,
        "translation_messages": [],
    }
    rec = _session_to_record(row)
    assert rec.preview == "orphan" and rec.title == "orphan" and rec.sentence_count == 0
    assert rec.sentences == []


async def test_selfcheck_success_walks_the_session_lifecycle():
    def handler(req):
        table = _table(req)
        params = dict(req.url.params)
        if table == "translation_sessions" and req.method == "POST":
            return httpx.Response(201, json=[{"id": "s-1"}])
        if table == "translation_messages" and req.method == "GET":
            return httpx.Response(200, json=[])
        if table == "translation_messages" and req.method == "POST":
            return httpx.Response(201, json=[{"id": "m-1", "session_id": "s-1", "sequence": 1,
                                              "direction": "sign_to_text", "input_text": "selfcheck",
                                              "output_text": "selfcheck", "gloss": None,
                                              "confidence": None, "model_version": "selfcheck",
                                              "result_type": "model",
                                              "created_at": "2026-01-02T00:00:00+00:00"}])
        if req.method == "PATCH":
            return httpx.Response(200, json=[{"id": "s-1"}])
        if table == "translation_sessions" and req.method == "GET":
            return httpx.Response(200, json=[{
                "id": "s-1", "user_id": "u1", "direction": "sign_to_text", "status": "completed",
                "title": "selfcheck", "created_at": "2026-01-02T00:00:00+00:00",
                "ended_at": None, "updated_at": None,
                "translation_messages": [_msg(1, "selfcheck")],
            }])
        if req.method == "DELETE":
            return httpx.Response(200, json=[{"id": "s-1"}])
        return httpx.Response(200, json=[])

    result = await _service(handler).selfcheck(USER)
    assert result == {
        "ok": True,
        "steps": [
            {"step": "session_insert", "status": 201, "code": None},
            {"step": "sentence_insert", "status": 201, "code": None},
            {"step": "read_back", "status": 200, "code": None},
            {"step": "delete", "status": 200, "code": None},
        ],
    }


async def test_selfcheck_fails_at_session_insert():
    def handler(req):
        return httpx.Response(400, json={"code": "23502", "message": "null user_id"})

    result = await _service(handler).selfcheck(USER)
    assert result["ok"] is False
    assert result["steps"][0]["status"] == 400
    assert result["steps"][0]["code"] == "23502"


async def test_selfcheck_fails_at_sentence_insert():
    def handler(req):
        table = _table(req)
        if table == "translation_sessions" and req.method == "POST":
            return httpx.Response(201, json=[{"id": "s-1"}])
        if table == "translation_messages" and req.method == "GET":
            return httpx.Response(200, json=[])
        if table == "translation_messages" and req.method == "POST":
            return httpx.Response(403, json={"code": "42501", "message": "RLS error"})
        if req.method == "PATCH":
            return httpx.Response(200, json=[])
        if req.method == "DELETE":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[])

    result = await _service(handler).selfcheck(USER)
    assert result["ok"] is False
    assert result["steps"][-1]["status"] == 403
    assert result["steps"][-1]["code"] == "42501"


async def test_unknown_column_400_pgrst204_degrades_to_history_not_saved(caplog):
    """Live schema without migration 002: PostgREST 400 PGRST204 (unknown
    column) must never raise -- the write path reports history_saved=false
    with a logged reason instead."""
    def handler(req):
        return httpx.Response(
            400,
            json={
                "code": "PGRST204",
                "message": "Could not find the 'client_key' column of 'translation_sessions' in the schema cache",
            },
        )

    with caplog.at_level(logging.WARNING):
        ok, err = await _service(handler).try_record_sentence_checked(USER, REC_KEYS)

    assert ok is False
    assert err == "session_lookup:400:PGRST204"
    assert any("PGRST204" in r.getMessage() and "status=400" in r.getMessage() for r in caplog.records)
    assert "user-jwt" not in caplog.text


async def test_unknown_column_400_never_reaches_the_insert():
    """With the live schema, the 400 is raised at the client_key lookup --
    before any insert is attempted -- and it is an HistoryUnavailableError
    tagged with the failing step."""
    def handler(req):
        return httpx.Response(
            400,
            json={"code": "PGRST204", "message": "Could not find the 'client_key' column of 'translation_sessions' in the schema cache"},
        )

    with pytest.raises(HistoryUnavailableError) as ei:
        await _repo(handler).record_sentence(USER, REC_KEYS)

    assert ei.value.status == 400 and ei.value.pg_code == "PGRST204"
    assert ei.value.step == "session lookup"
