from __future__ import annotations

import logging

logger = logging.getLogger("signsage.store")

# ponytail: no DB writes yet — Supabase RLS table exists (supabase/migrations/001…sql)
# Frontend inserts directly via Supabase client (authenticated) or localStorage fallback.
# Upgrade path: backend inserts here when SIGNSAGE_ENABLE_AUTH=true and service role is configured.

async def record_turn(*, user_id: str | None, direction: str, text: str, latency_ms: float | None = None) -> None:
    if user_id is None:
        return
    # no-op server-side for now; frontend will INSERT via Supabase PostgREST with RLS
    logger.info("turn user=%s dir=%s latency=%s text=%.80s", user_id, direction, latency_ms, text)
