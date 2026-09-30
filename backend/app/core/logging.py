"""Structured-ish logging setup.

Kept intentionally simple (acceptable for a research prototype -- see
docs/backend/error-handling.md), but every log line at least gets a
consistent format including logger name, so it is easy to grep by module.
Revisit if/when request/session IDs and latency categories are needed (see
the architecture audit's "logging is minimal" finding).
"""
from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
