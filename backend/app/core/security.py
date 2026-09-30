"""Access-token verification against Supabase Auth.

Supabase owns identity; this backend never issues tokens. A bearer token is
verified by asking Supabase (GET /auth/v1/user), which works regardless of how
the project signs JWTs (legacy secret or asymmetric keys) and needs no JWT
secret in this service. Results are cached for a few seconds
(SIGNSAGE_AUTH_CACHE_TTL_SECONDS) so history calls don't each pay a round trip.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import httpx

from app.core.config import Settings
from app.core.errors import AuthError, AuthNotConfiguredError, AuthUnavailableError

_CACHE_MAX_ENTRIES = 1000
_cache: Dict[str, Tuple[float, "AuthenticatedUser"]] = {}


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: Optional[str] = None
    # Kept so the database layer can forward it and let Postgres RLS enforce
    # per-user isolation. Excluded from repr so it never lands in logs.
    access_token: str = field(default="", repr=False)


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    if not authorization_header:
        return None
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must be 'Bearer <access token>'.")
    return token.strip()


def clear_auth_cache() -> None:
    _cache.clear()


async def verify_access_token(
    token: str, settings: Settings, transport: Optional[httpx.AsyncBaseTransport] = None
) -> AuthenticatedUser:
    if not settings.auth_enabled:
        raise AuthNotConfiguredError(
            "Authentication is not configured (SIGNSAGE_SUPABASE_URL / "
            "SIGNSAGE_SUPABASE_PUBLISHABLE_KEY)."
        )

    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1]

    url = f"{settings.supabase_url.rstrip('/')}/auth/v1/user"
    try:
        async with httpx.AsyncClient(timeout=5.0, transport=transport) as client:
            resp = await client.get(
                url,
                headers={"apikey": settings.supabase_publishable_key, "Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError as exc:
        raise AuthUnavailableError(f"Could not reach the auth provider: {type(exc).__name__}") from exc

    if resp.status_code in (401, 403):
        raise AuthError("Invalid or expired access token.")
    if resp.status_code != 200:
        raise AuthUnavailableError(f"Auth provider returned HTTP {resp.status_code}.")

    data = resp.json()
    user_id = data.get("id")
    if not user_id:
        raise AuthUnavailableError("Auth provider response did not include a user id.")
    user = AuthenticatedUser(id=str(user_id), email=data.get("email"), access_token=token)

    if settings.auth_cache_ttl_seconds > 0:
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            _cache.clear()
        _cache[key] = (now + settings.auth_cache_ttl_seconds, user)
    return user
