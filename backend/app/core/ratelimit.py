"""In-memory sliding-window rate limiter for the live demo.

One uvicorn worker = one window store, so limits are per-process. That is
fine for a single-worker demo; graduate to Redis-shared windows if the
service ever runs multiple workers. Not a security boundary: slows down
abuse, doesn't stop a determined attacker.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Optional

from fastapi import Depends, Request

from app.core.config import Settings
from app.core.dependencies import get_request_user, get_settings
from app.core.errors import RateLimitedError
from app.core.security import AuthenticatedUser

_RATE_RE = re.compile(r"(\d+)\s*/\s*([a-zA-Z]+)")
_UNIT_SECONDS = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hour": 3600}


def parse_rate(rate: str) -> tuple[int, float]:
    match = _RATE_RE.fullmatch(rate.strip())
    if not match:
        raise ValueError(f"invalid rate limit '{rate!r}' (expected e.g. '30/min')")
    return int(match.group(1)), _UNIT_SECONDS[match.group(2).lower()]


class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._stamps: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> Optional[float]:
        """Return retry-after seconds when `key` is over its limit, else None."""
        now = time.monotonic()
        window = self._stamps[key]
        cutoff = now - self.window_seconds
        window[:] = [t for t in window if t > cutoff]
        if len(window) >= self.max_requests:
            return max(window[0] + self.window_seconds - now, 0.0)
        window.append(now)
        return None


_LIMITERS: dict[str, SlidingWindowLimiter] = {}


def rate_limit_dependency(scope: str):
    """FastAPI dependency: 429 RATE_LIMITED when `scope`'s limit is exceeded.

    Keyed by the authenticated user id when present, else the client IP
    (behind Render's proxy this is the real client IP once uvicorn runs with
    --proxy-headers --forwarded-allow-ips='*').
    """

    async def _dependency(
        request: Request,
        settings: Settings = Depends(get_settings),
        user: Optional[AuthenticatedUser] = Depends(get_request_user),
    ) -> None:
        if not settings.rate_limit_enabled:
            return
        if scope not in _LIMITERS:
            max_requests, window_seconds = parse_rate(getattr(settings, f"rate_limit_{scope}"))
            _LIMITERS[scope] = SlidingWindowLimiter(max_requests, window_seconds)
        key = user.id if user else (request.client.host if request.client else "anonymous")
        retry_after = _LIMITERS[scope].check(key)
        if retry_after is not None:
            raise RateLimitedError(
                "Server is busy. Please wait a moment, then try again.",
                int(retry_after) + 1,
            )

    return _dependency
