"""Replay saved labeled captures through the live /api/v1/translation/word endpoint.

Posts each capture in the exact frontend request format ({frames: [...]}) and
prints, per file: the expected label (if stored), the response top-3 with
probabilities, and whether the prediction was accepted.

Usage:
    python scripts/replay_capture.py backend/captures_labeled/water http://localhost:8000
    python scripts/replay_capture.py backend/captures_labeled/water/20260101_000000_000000.npz http://localhost:8000

The first call uses a 90 s timeout (cold backend / model load); later calls use
--timeout (default 60 s).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/scripts/ -> repo root

TIMEOUT_FIRST_S = 90.0


def _load_capture(path: Path) -> tuple[np.ndarray, str | None]:
    """Return (landmarks, expected_label) from a saved capture .npz."""
    with np.load(path) as z:
        landmarks = np.asarray(z["landmarks"])
        expected: str | None = None
        for key in ("true_label", "label"):
            if key in z:
                value = z[key]
                try:
                    expected = str(value.tolist()[0])
                except (AttributeError, IndexError):
                    expected = str(value)
                break
    return landmarks, expected


def _frames_payload(landmarks: np.ndarray) -> list[dict]:
    """Exact frontend request format: [{landmarks: [x,y,z]x543}, ...]."""
    return [{"landmarks": frame.tolist()} for frame in landmarks]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay labeled captures through the word endpoint.",
    )
    parser.add_argument(
        "path",
        help="Saved .npz capture, or a folder of .npz captures to replay",
    )
    parser.add_argument(
        "base_url",
        help="Base URL of the backend, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for calls after the first (default: 60)",
    )
    args = parser.parse_args(argv)

    target = Path(args.path)
    if target.is_dir():
        files = sorted(target.rglob("*.npz"))
    elif target.is_file():
        files = [target]
    else:
        print(f"Not found: {target}")
        return 1

    if not files:
        print(f"No .npz captures found at {target}")
        return 0

    base = args.base_url.rstrip("/")
    url = f"{base}/api/v1/translation/word"

    print(f"Replaying {len(files)} capture(s) -> {url}")
    for i, path in enumerate(files, 1):
        landmarks, expected = _load_capture(path)
        payload = {"frames": _frames_payload(landmarks)}
        timeout = TIMEOUT_FIRST_S if i == 1 else args.timeout
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload)
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - report and keep replaying
            print(f"[{i}/{len(files)}] {path.name}: REQUEST FAILED: {exc}")
            continue

        top3 = body.get("top3", []) or []
        top3_str = ", ".join(
            f"{t.get('label')} {float(t.get('probability', 0)):.3f}" for t in top3
        ) or "—"
        print(
            f"[{i}/{len(files)}] {path.name}  "
            f"expected={expected or '—'}  "
            f"accepted={body.get('accepted')}  "
            f"word={body.get('word')}  "
            f"top3: {top3_str}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())