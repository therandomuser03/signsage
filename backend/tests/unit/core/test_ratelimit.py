"""Unit tests for the sliding-window rate limiter."""
from __future__ import annotations

import time

import pytest

from app.core.ratelimit import SlidingWindowLimiter, parse_rate


def test_parse_rate():
    assert parse_rate("30/min") == (30, 60.0)
    assert parse_rate("1/s") == (1, 1.0)
    assert parse_rate("2/hour") == (2, 3600.0)
    with pytest.raises(ValueError):
        parse_rate("30 per minute")
    with pytest.raises(ValueError):
        parse_rate("abc")


def test_allows_up_to_limit_and_blocks_next():
    lim = SlidingWindowLimiter(max_requests=3, window_seconds=60)
    assert lim.check("user-1") is None
    assert lim.check("user-1") is None
    assert lim.check("user-1") is None
    retry = lim.check("user-1")
    assert retry is not None and retry > 0
    # Other keys are unaffected
    assert lim.check("user-2") is None


def test_window_resets():
    lim = SlidingWindowLimiter(max_requests=1, window_seconds=0.05)
    assert lim.check("k") is None
    assert lim.check("k") is not None
    time.sleep(0.1)
    assert lim.check("k") is None


def test_keys_are_isolated():
    lim = SlidingWindowLimiter(max_requests=1, window_seconds=60)
    assert lim.check("a") is None
    assert lim.check("a") is not None
    assert lim.check("b") is None
