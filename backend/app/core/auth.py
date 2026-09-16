from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("signsage.auth")

async def verify_supabase_token(token: str, supabase_url: str, anon_key: str | None) -> str | None:
    """Verify Supabase JWT by calling Auth getUser. Returns user_id or None.

    ponytail: no PyJWT dep; uses Supabase HTTP API directly — upgrade to local JWT verify if latency matters.
    """
    if not token or not supabase_url:
        return None
    url = supabase_url.rstrip("/") + "/auth/v1/user"
    headers = {"Authorization": f"Bearer {token}", "apikey": anon_key or token}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("id") or data.get("user", {}).get("id")
        logger.warning("supabase auth verify failed %s %s", resp.status_code, resp.text[:200])
        return None
    except Exception:
        logger.exception("supabase auth verify error")
        return None
