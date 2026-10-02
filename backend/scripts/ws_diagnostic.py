"""SignSage WebSocket TRANSPORT / CONTRACT diagnostic (developer tool).

Verifies the live pipeline's plumbing against a RUNNING backend:

    WS connect -> hostile input rejected -> synthetic landmark frames accepted
    -> sentence.boundary -> structured translation.text (or honest MODEL_UNAVAILABLE)

The landmark payload is a clearly synthetic ramp pattern used ONLY to exercise
transport, validation, buffering, and the response contract. It is NOT sign
recognition and its output says NOTHING about model accuracy.

Exit 0 when every applicable check passes, 1 otherwise.

Usage:
    python scripts/ws_diagnostic.py [ws://localhost:8000/ws/translate] [--frames 25]
"""
from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

try:
    import websockets
except ImportError:  # pragma: no cover
    raise SystemExit("the 'websockets' package is required (pip install 'uvicorn[standard]')")


CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def http_get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=10) as res:
        return json.loads(res.read().decode())


def synthetic_frame(i: int, n: int) -> list[list[float]]:
    """Deterministic ramp: TRANSPORT TEST DATA, not a sign."""
    t = i / max(n - 1, 1)
    return [[round(t, 4), round(1 - t, 4), 0.0] for _ in range(543)]


async def main(url: str, n_frames: int) -> int:
    http_base = url.replace("ws://", "http://").replace("wss://", "https://").removesuffix("/ws/translate")

    try:
        health = http_get(http_base, "/api/v1/health")
        caps = http_get(http_base, "/api/v1/capabilities")
    except Exception as exc:  # noqa: BLE001
        check("backend reachable (health + capabilities)", False, str(exc))
        return 1
    check("backend reachable (health + capabilities)", True)
    print(f"  health={health}")
    print(f"  sign_to_text_available={caps.get('sign_to_text_available')} "
          f"reason={caps.get('sign_to_text_unavailable_reason')}")
    gated = not caps.get("sign_to_text_available", False)

    try:
        ws = await websockets.connect(url, max_size=2 ** 22)
    except Exception as exc:  # noqa: BLE001
        check("websocket connect", False, str(exc))
        return 1
    check("websocket connect", True, url)

    async def send_raw(text: str) -> dict:
        await ws.send(text)
        return json.loads(await ws.recv())

    # Gated servers send MODEL_UNAVAILABLE then close immediately on connect;
    # serving servers stay silent until the client speaks. Peek first.
    try:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
    except asyncio.TimeoutError:
        first = None
    if first is not None:
        check("gated server refuses honestly on connect (MODEL_UNAVAILABLE)",
              first.get("code") == "MODEL_UNAVAILABLE", str(first))
        print("  SKIP: hostile-input and sequence checks need a serving backend.")
        print("  NOTE: set SIGNSAGE_ALLOW_UNPROMOTED_DEV=true locally with a real bundle")
        print("  to exercise the full inference path (dev only, never production).")
        return 0 if all(ok for _, ok, _ in CHECKS) else 1

    # 1. Hostile input must get coded errors, never kill the socket.
    msg = await send_raw("not json at all")
    check("invalid JSON -> INVALID_JSON", msg.get("code") == "INVALID_JSON", str(msg))

    msg = await send_raw(json.dumps({"type": "landmarks.frame", "landmarks": [[0, 0]] * 543}))
    check("ragged frame -> INVALID_MESSAGE", msg.get("code") == "INVALID_MESSAGE", str(msg))

    nan_frame = [[float("nan"), 0, 0]] + [[0, 0, 0]] * 542
    msg = await send_raw(json.dumps({"type": "landmarks.frame", "landmarks": nan_frame}))
    check("NaN frame -> INVALID_MESSAGE", msg.get("code") == "INVALID_MESSAGE", str(msg))

    msg = await send_raw(json.dumps({"type": "nope"}))
    check("unknown type -> UNKNOWN_MESSAGE_TYPE", msg.get("code") == "UNKNOWN_MESSAGE_TYPE", str(msg))

    # 2. Synthetic frames -> boundary -> structured result (or honest gating).
    # Send enough to cross the rolling window (80) so a serving backend may
    # also emit a stabilizing interim. The ramp has near-zero motion, so the
    # gate usually rejects it: absence of an interim is INFO, not failure.
    total = max(n_frames, 85)
    for i in range(total):
        await ws.send(json.dumps({"type": "landmarks.frame", "landmarks": synthetic_frame(i, total)}))
    msg = None
    try:
        while True:
            early = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            if early.get("type") == "translation.text" and early.get("final") is False:
                check("rolling interim received (final=false)", True,
                      f"accepted={early.get('accepted')} reason={early.get('reason')}")
            else:
                msg = early
                break
    except asyncio.TimeoutError:
        check("rolling interim received (final=false)", True, "none emitted (gate rejected ramp); skipped")
    if msg is None:
        await ws.send(json.dumps({"type": "sentence.boundary", "save_history": False}))
        msg = json.loads(await ws.recv())

    if msg.get("type") == "translation.text":
        ok = all(k in msg for k in ("text", "accepted", "reason", "confidence", "result_type", "final"))
        check("synthetic sequence -> translation.text", ok, f"accepted={msg.get('accepted')} "
              f"reason={msg.get('reason')} result_type={msg.get('result_type')} final={msg.get('final')}")
        print("  NOTE: output is from SYNTHETIC transport-test landmarks, not a real sign.")
    elif msg.get("code") == "MODEL_UNAVAILABLE":
        check("gated server refuses honestly (MODEL_UNAVAILABLE)", True, msg.get("message", ""))
        print("  NOTE: set SIGNSAGE_ALLOW_UNPROMOTED_DEV=true locally with a real bundle")
        print("  to exercise the full inference path (dev only, never production).")
    else:
        check("synthetic sequence -> structured reply", False, str(msg))

    try:
        await ws.close()
    except Exception:  # noqa: BLE001
        pass
    if gated:
        print("  NOTE: capabilities report sign_to_text_available=false; full-path")
        print("  translation.text above (if any) came from dev-unpromoted mode or a")
        print("  promoted bundle -- see backend/models/README.md.")
    return 0 if all(ok for _, ok, _ in CHECKS) else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = 25
    for a in sys.argv[1:]:
        if a.startswith("--frames="):
            n = int(a.split("=", 1)[1])
    sys.exit(asyncio.run(main(args[0] if args else "ws://localhost:8000/ws/translate", n)))
