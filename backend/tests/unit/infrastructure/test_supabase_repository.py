"""The Supabase repository must send exactly the PostgREST requests RLS expects."""
import json

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import AuthError, HistoryUnavailableError
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistoryRecord, ResultType
from app.infrastructure.database.repositories.translation_repository import (
    InMemoryTranslationRepository, SupabaseTranslationRepository,
)
from app.infrastructure.database.supabase import SupabaseRestClient

USER = AuthenticatedUser(id="u1", email="a@b.c", access_token="user-jwt")
ROW = {"id": "r1", "user_id": "u1", "direction": "sign_to_text", "input_text": None, "output_text": "hello",
       "confidence": 0.4, "model_version": "m", "result_type": "model", "created_at": "2026-09-28T07:00:00.123456+00:00"}


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
