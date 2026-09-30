"""CLI: hit a running server's /health endpoint and print the result.

Usage:
    python -m scripts.health_check [url]
"""
from __future__ import annotations

import sys

import httpx


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/api/v1/health"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Health check failed: {exc}")
        return 1
    print(resp.json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
