"""Structured-ish logging setup.

Structured output suitable for production observability pipelines, but every
log line at least gets a consistent format including logger name, so it is
easy to grep by module. Revisit if/when request/session IDs and latency
categories are needed."""
from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
