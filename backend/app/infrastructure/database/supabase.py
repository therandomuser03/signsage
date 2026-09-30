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
            raise HistoryUnavailableError(f"Could not reach the database: {type(exc).__name__}") from exc
        if resp.status_code == 401:
            raise AuthError("Invalid or expired access token.")
        if resp.status_code >= 400:
            raise HistoryUnavailableError(f"Database request failed (HTTP {resp.status_code}).")
        return resp
