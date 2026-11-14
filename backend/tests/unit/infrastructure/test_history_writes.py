"""History write paths against a mocked PostgREST layer (C7).

Every path that saves history goes through HistoryService.try_record_word ->
SupabaseTranslationRepository.add_word_recognition: one translation_sessions
row, then one translation_messages row, always on the caller's JWT.
"""
import json
import logging

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import HistoryUnavailableError, NotFoundError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistoryRecord, ResultType
from app.infrastructure.database.repositories.translation_repository import (
    SupabaseTranslationRepository,
    _session_to_record,
)
from app.infrastructure.database.supabase import SupabaseRestClient
from app.services.history_service import HistoryService

USER = AuthenticatedUser(id="u1", email="a@b.c", access_token="user-jwt")
REC = NewHistoryRecord(
    Direction.SIGN_TO_TEXT, "Hello, help, me.", ResultType.MODEL, "word_v3_8w_s123",
    input_text="hello help me", gloss="hello help me", confidence=0.8,
    sequence=1, title="Word recognition: Hello, help, me.", status="completed",
)


def _repo(handler):
    settings = Settings(supabase_url="https://x.supabase.co", supabase_publishable_key="pub")
    return SupabaseTranslationRepository(SupabaseRestClient(settings, transport=httpx.MockTransport(handler)))


def _service(handler):
    return HistoryService(_repo(handler))


def _table(req):
    return str(req.url).rsplit("/rest/v1/", 1)[1].split("?", 1)[0]


def _full_handler(calls):
    def handler(req):
        calls.append({
            "table": _table(req),
            "method": req.method,
            "params": dict(req.url.params),
            "auth": req.headers["authorization"],
            "apikey": req.headers["apikey"],
            "prefer": req.headers.get("prefer"),
            "body": json.loads(req.content) if req.content else None,
        })
        if _table(req) == "translation_sessions":
            return httpx.Response(201, json=[{"id": "s-1"}])
        return httpx.Response(201, json=[{"id": "m-1"}])
    return handler


async def test_word_path_writes_session_then_message_with_columns_and_headers():
    calls = []
    session_id = await _repo(_full_handler(calls)).add_word_recognition(USER, REC)

    assert session_id == "m-1"
    assert [c["table"] for c in calls] == ["translation_sessions", "translation_messages"]
    assert all(c["method"] == "POST" for c in calls)
    assert all(c["auth"] == "Bearer user-jwt" for c in calls)
    assert all(c["apikey"] == "pub" for c in calls)
    assert all(c["prefer"] == "return=representation" for c in calls)
    session_body, message_body = calls[0]["body"], calls[1]["body"]
    assert session_body == {
        "user_id": "u1",
        "direction": "sign_to_text",
        "status": "completed",
        "title": "Word recognition: Hello, help, me.",
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
    }


async def test_session_insert_failure_logs_status_body_and_path_without_secrets(caplog):
    def handler(req):
        return httpx.Response(
            400,
            json={"code": "23502", "message": 'null value in column "user_id" violates not-null constraint'},
        )

    with caplog.at_level(logging.WARNING):
        result = await _service(handler).try_record_word(USER, REC)

    assert result is None
    assert any(
        "session insert" in r.getMessage() and "status=400" in r.getMessage() and "pg_code=23502" in r.getMessage()
        for r in caplog.records
    ), caplog.text
    assert any("/rest/v1/translation_sessions" in r.getMessage() for r in caplog.records), caplog.text
    assert "user-jwt" not in caplog.text
    assert "Hello, help, me." not in caplog.text


async def test_message_insert_failure_logs_step_status_and_path_without_secrets(caplog):
    def handler(req):
        if _table(req) == "translation_sessions":
            return httpx.Response(201, json=[{"id": "s-1"}])
        return httpx.Response(
            403,
            json={"code": "42501", "message": "row-level security policy violated for table translation_messages"},
        )

    with caplog.at_level(logging.WARNING):
        result = await _service(handler).try_record_word(USER, REC)

    assert result is None
    assert any(
        "message insert" in r.getMessage() and "status=403" in r.getMessage() and "pg_code=42501" in r.getMessage()
        for r in caplog.records
    ), caplog.text
    assert any("/rest/v1/translation_messages" in r.getMessage() for r in caplog.records), caplog.text
    assert "user-jwt" not in caplog.text
    assert "Hello, help, me." not in caplog.text


async def test_expired_jwt_logs_401_without_the_token(caplog):
    def handler(req):
        return httpx.Response(401)

    with caplog.at_level(logging.WARNING):
        result = await _service(handler).try_record_word(USER, REC)

    assert result is None
    assert any("session insert" in r.getMessage() and "401" in r.getMessage() for r in caplog.records), caplog.text
    assert "user-jwt" not in caplog.text


async def test_list_falls_back_to_two_requests_when_embed_fk_is_missing(caplog):
    EMBED = (
        "id,title,direction,status,created_at,"
        "translation_messages(sequence,input_text,output_text,gloss,"
        "confidence,model_version,result_type)"
    )
    messages = [
        {"session_id": "s-1", "sequence": 1, "input_text": "hi", "output_text": "Hi.",
         "gloss": "hello", "confidence": 0.9, "model_version": "mv", "result_type": "model"},
    ]
    seen = []

    def handler(req):
        params = dict(req.url.params)
        seen.append((_table(req), params))
        if params.get("select", "").startswith("id,title"):  # embed attempt
            return httpx.Response(400, json={"code": "PGRST200", "message": "Could not find a relationship between the tables"})
        if _table(req) == "translation_sessions":
            return httpx.Response(200, json=[{"id": "s-1", "user_id": "u1", "direction": "sign_to_text",
                                              "status": "completed", "title": "t",
                                              "created_at": "2026-01-02T00:00:00+00:00"}],
                                  headers={"content-range": "0-0/1"})
        return httpx.Response(200, json=messages)

    with caplog.at_level(logging.WARNING):
        rows, total = await _repo(handler).list(USER, 5, 0)

    assert total == 1
    assert len(rows) == 1
    assert rows[0].id == "s-1"
    assert rows[0].output_text == "Hi." and rows[0].confidence == 0.9
    assert seen[0][0] == "translation_sessions"
    assert seen[0][1]["select"] == EMBED
    assert seen[1][0] == "translation_sessions"      # fallback: page of sessions
    assert seen[2][0] == "translation_messages"      # ... then its messages
    assert seen[2][1]["session_id"] == "in.(s-1)"
    assert any("embed failed" in r.getMessage() for r in caplog.records), caplog.text


async def test_list_does_not_fall_back_on_unrelated_errors():
    def handler(req):
        return httpx.Response(500, json={"code": "XX000", "message": "internal error"})

    with pytest.raises(HistoryUnavailableError):
        await _repo(handler).list(USER, 5, 0)


async def test_list_with_working_embed_keeps_single_request():
    def handler(req):
        return httpx.Response(200, json=[{
            "id": "s-1", "user_id": "u1", "direction": "sign_to_text", "status": "completed",
            "title": "t", "created_at": "2026-01-02T00:00:00+00:00",
            "translation_messages": [
                {"sequence": 1, "input_text": "a", "output_text": "A", "gloss": "A", "confidence": 0.9,
                 "model_version": "mv", "result_type": "model"},
            ],
        }], headers={"content-range": "0-0/7"})

    rows, total = await _repo(handler).list(USER, 1, 0)
    assert total == 7
    assert rows[0].output_text == "A"


async def test_delete_removes_messages_then_session():
    calls = []

    def handler(req):
        calls.append(_table(req))
        return httpx.Response(200, json=[{"id": "s-1"}])

    ok = await _repo(handler).delete(USER, "s-1")
    assert ok is True
    assert calls == ["translation_messages", "translation_sessions"]


async def test_delete_missing_session_raises_not_found_at_service_layer():
    service = _service(lambda req: httpx.Response(200, json=[]))
    with pytest.raises(NotFoundError):
        await service.delete(USER, "nope")


async def test_clear_removes_messages_before_sessions():
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
    assert calls[2][0] == "translation_sessions"


def test_row_to_record_mapping_untouched():
    row = {"id": "r1", "user_id": "u1", "direction": "sign_to_text",
           "title": "orphan", "status": "completed", "created_at": "2026-01-01T00:00:00+00:00",
           "translation_messages": []}
    rec = _session_to_record(row)
    assert rec.output_text == "orphan" and rec.title == "orphan"
