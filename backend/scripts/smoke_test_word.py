"""Smoke test for the word-recognition path.

Standard library + numpy only. Usage:
    python scripts/smoke_test_word.py <BASE_URL>

Checks:
  1. GET  /api/v1/health          -> 200
  2. GET  /api/v1/capabilities    -> 200, word_recognition_enabled true
  3. POST /api/v1/translation/word  with the golden test vector
      (frontend frame format) -> word="hello", accepted=true, prints model_version
  4. POST /api/v1/translation/compose with ["where","water"]
      -> sentence="Where is the water?"

Prints PASS/FAIL per check; exits non-zero on any failure.
The first call allows a 90 s timeout (Render cold start).
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"
BASE_URL = BASE_URL.rstrip("/")

HEALTH = BASE_URL + "/api/v1/health"
CAPABILITIES = BASE_URL + "/api/v1/capabilities"
WORD = BASE_URL + "/api/v1/translation/word"
COMPOSE = BASE_URL + "/api/v1/translation/compose"

GOLDEN = Path(__file__).resolve().parent.parent / "word_model" / "golden_test_vector.npy"

failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def request(method, path, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body) if body else {}
        except Exception:
            return exc.code, {"_raw": body.decode("utf-8", "replace")}
    except Exception as exc:  # urllib.error.URLError, timeout, etc.
        return None, {"_error": repr(exc)}


def main():
    # 1. health (90 s timeout for cold start)
    status, body = request("GET", HEALTH, timeout=90)
    check("GET /api/v1/health", status == 200, f"status={status} body={body}")

    # 2. capabilities
    status, body = request("GET", CAPABILITIES)
    wr_enabled = body.get("word_recognition_enabled") is True
    check(
        "GET /api/v1/capabilities",
        status == 200 and wr_enabled,
        f"status={status} word_recognition_enabled={body.get('word_recognition_enabled')}",
    )

    # 3. word recognition on the golden test vector (frontend frame format)
    arr = np.load(GOLDEN).astype(np.float32)
    payload = {"frames": [{"landmarks": frame.tolist()} for frame in arr]}
    status, body = request("POST", WORD, payload, timeout=30)
    word_ok = (
        status == 200
        and body.get("word") == "hello"
        and body.get("accepted") is True
        and "model_version" in body
    )
    check(
        "POST /api/v1/translation/word (golden vector)",
        word_ok,
        f"status={status} word={body.get('word')} accepted={body.get('accepted')} "
        f"model_version={body.get('model_version')} status_field={body.get('status')}",
    )

    # 4. compose ["where","water"]
    status, body = request("POST", COMPOSE, {"words": ["where", "water"]}, timeout=30)
    check(
        'POST /api/v1/translation/compose ["where","water"]',
        status == 200 and body.get("sentence") == "Where is the water?",
        f"status={status} sentence={body.get('sentence')!r} method={body.get('method')}",
    )

    print()
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("SMOKE TEST PASSED: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
