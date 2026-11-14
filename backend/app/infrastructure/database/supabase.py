"""Thin PostgREST client for Supabase (no supabase-py dependency).

Every request forwards the *user's* access token, so Postgres row-level
security (see migrations/001_translation_sessions.sql) enforces per-user
isolation even if this service had a bug. The service never holds a
service-role key.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from app.core.config import Settings
from app.core.errors import AuthError, HistoryUnavailableError
from app.core.security import AuthenticatedUser


def _postgrest_error(resp: httpx.Response) -> tuple[Optional[str], Optional[str]]:
    """Extract the PostgREST {code, message} pair from an error body.

    Returns (None, None) when the body is not the expected shape. Bounded
    length; never includes request data, headers or tokens.
    """
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 -- best-effort error enrichment only
        return None, None
    if not isinstance(body, dict):
        return None, None
    code = body.get("code") if isinstance(body.get("code"), str) else None
    message = body.get("message") if isinstance(body.get("message"), str) else None
    if message is not None and len(message) > 300:
        message = message[:300] + "…"
    return code, message


class SupabaseRestClient:
    def __init__(self, settings: Settings, transport: Optional[httpx.AsyncBaseTransport] = None):
        if not (settings.supabase_url and settings.supabase_publishable_key):
            raise HistoryUnavailableError("Supabase is not configured.")
        self._base = settings.supabase_url.rstrip("/") + "/rest/v1"
        self._key = settings.supabase_publishable_key
        self._transport = transport

    def _headers(self, user: AuthenticatedUser, prefer: Optional[str]) -> Dict[str, str]:
        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {user.access_token}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    async def request(
        self,
        method: str,
        table: str,
        user: AuthenticatedUser,
        params: Optional[Dict[str, Any]] = None,
        json: Any = None,
        prefer: Optional[str] = None,
    ) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=10.0, transport=self._transport) as client:
                resp = await client.request(
                    method, f"{self._base}/{table}", params=params, json=json,
                    headers=self._headers(user, prefer),
                )
        except httpx.HTTPError as exc:
            err = HistoryUnavailableError(
                f"Could not reach the database: {type(exc).__name__}",
                path=f"/rest/v1/{table}",
            )
            raise err from exc
        if resp.status_code == 401:
            err = AuthError("Invalid or expired access token.")
            err.path = f"/rest/v1/{table}"  # type: ignore[attr-defined]
            err.status = 401  # type: ignore[attr-defined]
            raise err
        if resp.status_code >= 400:
            pg_code, pg_message = _postgrest_error(resp)
            raise HistoryUnavailableError(
                f"Database request failed (HTTP {resp.status_code}).",
                path=f"/rest/v1/{table}",
                status=resp.status_code,
                pg_code=pg_code,
                pg_message=pg_message,
            )
        return resp
