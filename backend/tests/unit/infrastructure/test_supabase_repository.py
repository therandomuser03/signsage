"""The Supabase repository must send exactly the PostgREST requests RLS expects."""
import json

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import AuthError, HistoryUnavailableError, NotFoundError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistoryRecord, ResultType
from app.infrastructure.database.repositories.translation_repository import (
    InMemoryTranslationRepository, SupabaseTranslationRepository,
)
from app.infrastructure.database.supabase import SupabaseRestClient
from app.services.history_service import HistoryService

USER = AuthenticatedUser(id="u1", email="a@b.c", access_token="user-jwt")
ROW = {"id": "r1", "user_id": "u1", "direction": "sign_to_text", "input_text": None, "output_text": "hello",
       "confidence": 0.4, "model_version": "m", "result_type": "model", "created_at": "2026-09-28T07:00:00.123456+00:00"}

EMBED_SELECT = (
    "id,title,direction,status,created_at,"
    "translation_messages(sequence,input_text,output_text,gloss,"
    "confidence,model_version,result_type)"
)


def _session(session_id, title, created_at, messages=None):
    return {"id": session_id, "user_id": "u1", "direction": "sign_to_text", "status": "completed",
            "title": title, "created_at": created_at,
            "translation_messages": messages or []}


def _repo(handler):
    settings = Settings(supabase_url="https://x.supabase.co", supabase_publishable_key="pub")
    return SupabaseTranslationRepository(SupabaseRestClient(settings, transport=httpx.MockTransport(handler)))


async def test_add_posts_row_with_user_jwt_and_returns_record():
    seen = {}
    def handler(req):
        seen.update(method=req.method, url=str(req.url), auth=req.headers["authorization"],
                    apikey=req.headers["apikey"], prefer=req.headers["prefer"], body=json.loads(req.content))
        return httpx.Response(201, json=[ROW])
    rec = await _repo(handler).add(USER, NewHistoryRecord(Direction.SIGN_TO_TEXT, "hello", ResultType.MODEL, "m", confidence=0.4))
    assert seen["method"] == "POST" and seen["url"].endswith("/rest/v1/translation_sessions")
    assert seen["auth"] == "Bearer user-jwt" and seen["apikey"] == "pub"   # user's JWT, never a service key
    assert seen["prefer"] == "return=representation"
    assert seen["body"]["user_id"] == "u1" and seen["body"]["result_type"] == "model"
    assert rec.id == "r1" and rec.direction == Direction.SIGN_TO_TEXT and rec.created_at.year == 2026


async def test_list_uses_pagination_and_parses_total_from_content_range():
    seen = {}
    def handler(req):
        seen["q"] = dict(req.url.params)
        return httpx.Response(200, json=[ROW], headers={"content-range": "10-10/42"})
    rows, total = await _repo(handler).list(USER, limit=1, offset=10)
    assert seen["q"]["order"] == "created_at.desc" and seen["q"]["limit"] == "1" and seen["q"]["offset"] == "10"
    assert len(rows) == 1 and total == 42


async def test_delete_reports_whether_a_row_was_removed():
    seen = {}
    def handler(req):
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json=[ROW])
    assert await _repo(handler).delete(USER, "r1") is True
    # Delete-by-id still rides the caller's JWT: Postgres RLS is what turns a
    # cross-user id into an empty (→404) response instead of a deletion.
    assert seen["auth"] == "Bearer user-jwt"
    assert await _repo(lambda r: httpx.Response(200, json=[])).delete(USER, "nope") is False


async def test_clear_filters_on_own_user_id():
    seen = {}
    def handler(req):
        seen["q"] = dict(req.url.params)
        return httpx.Response(200, json=[ROW, ROW])
    assert await _repo(handler).clear(USER) == 2
    assert seen["q"]["user_id"] == "eq.u1"


async def test_expired_jwt_maps_to_auth_error_and_server_errors_to_unavailable():
    with pytest.raises(AuthError):
        await _repo(lambda r: httpx.Response(401)).list(USER, 5, 0)
    with pytest.raises(HistoryUnavailableError):
        await _repo(lambda r: httpx.Response(500)).list(USER, 5, 0)


async def test_list_embeds_messages_selects_only_live_columns_and_maps_first_message_by_sequence():
    seen = {}
    def handler(req):
        seen["q"] = dict(req.url.params)
        seen["prefer"] = req.headers["prefer"]
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json=[
            _session("s-2", "t2", "2026-01-02T00:00:00+00:00", [
                # Deliberately out of sequence: sequence 2 first in the array.
                {"sequence": 2, "input_text": "b", "output_text": "B", "gloss": "B", "confidence": 0.4,
                 "model_version": "mv", "result_type": "avatar_gloss"},
                {"sequence": 1, "input_text": "a", "output_text": "A", "gloss": "A", "confidence": 0.9,
                 "model_version": "mv", "result_type": "avatar_gloss"},
            ]),
            _session("s-1", "orphan session", "2026-01-01T00:00:00+00:00", []),
        ], headers={"content-range": "0-1/42"})
    rows, total = await _repo(handler).list(USER, limit=5, offset=0)

    assert seen["q"]["select"] == EMBED_SELECT
    assert seen["q"]["order"] == "created_at.desc"
    assert seen["q"]["limit"] == "5" and seen["q"]["offset"] == "0"
    assert seen["prefer"] == "count=exact"
    assert seen["auth"] == "Bearer user-jwt"                       # user's JWT, never a service key
    assert total == 42
    assert [r.id for r in rows] == ["s-2", "s-1"]                  # newest first

    first = rows[0]
    assert first.direction == Direction.SIGN_TO_TEXT
    assert first.output_text == "A"                                # first message by sequence (1)
    assert first.input_text == "a" and first.confidence == 0.9
    assert first.model_version == "mv" and first.result_type == ResultType.AVATAR_GLOSS
    assert first.title == "t2"

    legacy = rows[1]                                               # no message: fall back to title
    assert legacy.output_text == "orphan session"
    assert legacy.input_text is None and legacy.confidence is None
    assert legacy.model_version == "" and legacy.title == "orphan session"


async def test_delete_removes_messages_then_session_on_the_caller_jwt():
    calls = []
    def handler(req):
        table = str(req.url).rsplit("/rest/v1/", 1)[1].split("?", 1)[0]
        calls.append((table, dict(req.url.params), req.headers["authorization"]))
        return httpx.Response(200, json=[ROW])
    assert await _repo(handler).delete(USER, "r1") is True
    assert [c[0] for c in calls] == ["translation_messages", "translation_sessions"]
    assert calls[0][1] == {"session_id": "eq.r1"}
    assert calls[1][1] == {"id": "eq.r1"}
    assert calls[0][2] == calls[1][2] == "Bearer user-jwt"


async def test_delete_tolerates_no_rows():
    assert await _repo(lambda r: httpx.Response(200, json=[])).delete(USER, "missing") is False


async def test_delete_missing_session_raises_not_found_at_service_layer():
    service = HistoryService(_repo(lambda r: httpx.Response(200, json=[])))
    with pytest.raises(NotFoundError):
        await service.delete(USER, "missing")


async def test_in_memory_repository_isolates_users_and_orders_newest_first():
    repo = InMemoryTranslationRepository()
    other = AuthenticatedUser(id="u2")
    a = await repo.add(USER, NewHistoryRecord(Direction.SIGN_TO_TEXT, "one", ResultType.MODEL, "m"))
    b = await repo.add(USER, NewHistoryRecord(Direction.SIGN_TO_TEXT, "two", ResultType.MODEL, "m"))
    await repo.add(other, NewHistoryRecord(Direction.SIGN_TO_TEXT, "theirs", ResultType.MODEL, "m"))
    rows, total = await repo.list(USER, 10, 0)
    assert [r.output_text for r in rows] == ["two", "one"] and total == 2
    assert await repo.delete(other, a.id) is False          # cannot delete someone else's entry
    assert await repo.delete(USER, a.id) is True
    assert await repo.clear(USER) == 1
