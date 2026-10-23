"""Regression tests for the failures seen in the signsage-21-oct Kaggle run.

Run from the repository root:   python -m pytest research/experiments/test_pipeline_hardening.py -q

Covers (fakes/fixtures only -- no network, no GPU, no Kaggle, no real data, no training):
  * bounded retry of the remote size probe (the call that was reset in the failed run)
  * fail-closed behaviour: exhausted retries raise a typed transport error, 403 is not retried
  * atomic checkpoint writes (a killed write must not corrupt the previous checkpoint)
  * cache mode refuses to silently stream remotely
  * seed -> subset/split determinism (the seed-1 vs seed-42 bug that caused 471/7,500 cache overlap)
  * an interrupted overfit gate is reported as `interrupted`, never as a quality failure
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, "research/experiments")
sys.path.insert(0, "backend")
import train_sign_to_text as T  # noqa: E402

TOKEN = "hf_SECRETTOKEN0123456789"


class _Resp:
    def __init__(self, status, headers=None):
        self.status_code, self.headers = status, headers or {}


class _FakeRequests:
    """Stands in for the `requests` module inside RemoteConcatFile (get + real exception classes)."""

    exceptions = requests.exceptions

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def get(self, url, **kw):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _probe_obj(script, monkeypatch):
    sleeps = []
    monkeypatch.setattr(T.time, "sleep", lambda s: sleeps.append(s))
    obj = T.RemoteConcatFile.__new__(T.RemoteConcatFile)
    obj._requests = _FakeRequests(script)
    return obj, sleeps


OK_206 = lambda total=12345: _Resp(206, {"Content-Range": f"bytes 0-0/{total}"})  # noqa: E731
RESET = lambda: requests.exceptions.ConnectionError(  # noqa: E731
    "('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))")


def test_probe_recovers_from_transient_resets(monkeypatch):
    obj, sleeps = _probe_obj([RESET(), RESET(), OK_206()], monkeypatch)
    assert obj._probe("https://x/part_aa", TOKEN) == 12345
    assert obj._requests.calls == 3 and len(sleeps) == 2


def test_probe_exhaustion_is_bounded_typed_and_leaks_no_token(monkeypatch):
    obj, sleeps = _probe_obj([RESET() for _ in range(T.HTTP_RETRY_ATTEMPTS)], monkeypatch)
    with pytest.raises(T.RemoteTransportError) as ei:
        obj._probe("https://x/part_aa", TOKEN)
    assert isinstance(ei.value, ConnectionError)               # old handlers still catch it
    assert obj._requests.calls == T.HTTP_RETRY_ATTEMPTS        # bounded
    assert len(sleeps) == T.HTTP_RETRY_ATTEMPTS - 1            # no pointless sleep after the last try
    assert TOKEN not in str(ei.value) and "Bearer" not in str(ei.value)


def test_probe_retries_5xx_and_caps_retry_after(monkeypatch):
    obj, sleeps = _probe_obj([_Resp(503, {"Retry-After": "7"}), _Resp(503, {"Retry-After": "99999"}), OK_206()],
                             monkeypatch)
    assert obj._probe("u", TOKEN) == 12345
    assert sleeps == [7.0, T.RETRY_AFTER_CAP_S]


def test_probe_5xx_exhaustion_fails_closed(monkeypatch):
    obj, _ = _probe_obj([_Resp(503) for _ in range(T.HTTP_RETRY_ATTEMPTS)], monkeypatch)
    with pytest.raises(T.RemoteTransportError):
        obj._probe("u", TOKEN)
    assert obj._requests.calls == T.HTTP_RETRY_ATTEMPTS


@pytest.mark.parametrize("status,exc", [(403, PermissionError), (401, PermissionError), (404, RuntimeError)])
def test_probe_does_not_retry_configuration_errors(status, exc, monkeypatch):
    obj, sleeps = _probe_obj([_Resp(status), OK_206()], monkeypatch)
    with pytest.raises(exc):
        obj._probe("u", TOKEN)
    assert obj._requests.calls == 1 and sleeps == []


def test_retry_after_cap_and_invalid_header():
    assert T._retry_after_delay("100000", 0) == T.RETRY_AFTER_CAP_S
    assert T._retry_after_delay("-5", 0) == 0.0
    assert 1.5 <= T._retry_after_delay("garbage", 0) < 2.6      # falls back to jittered backoff


# ---------------------------------------------------------------- atomic checkpoints
def test_atomic_save_roundtrip_and_no_tmp_left(tmp_path):
    import torch
    p = tmp_path / "ck.pt"
    T._atomic_torch_save({"a": torch.arange(3)}, p)
    assert torch.equal(torch.load(p)["a"], torch.arange(3))
    assert not list(tmp_path.glob("*.tmp"))


def test_killed_write_leaves_previous_checkpoint_intact(tmp_path, monkeypatch):
    import torch
    p = tmp_path / "ck.pt"
    T._atomic_torch_save({"v": 1}, p)
    good = p.read_bytes()

    def dying_save(obj, fh):
        fh.write(b"PARTIAL")                       # session dies mid-write
        raise OSError("killed")

    monkeypatch.setattr(T.torch, "save", dying_save)
    with pytest.raises(OSError):
        T._atomic_torch_save({"v": 2}, p)
    assert p.read_bytes() == good                  # previous good checkpoint untouched
    monkeypatch.undo()
    assert torch.load(p)["v"] == 1


# ---------------------------------------------------------------- cache mode never silently streams
@pytest.mark.parametrize("stage", ["overfit", "train", "evaluate", "export"])
def test_cache_mode_refuses_incomplete_cache(stage):
    with pytest.raises(SystemExit) as ei:
        T.require_local_cache("cache", {"tokenizer", stage}, present=471, missing=7029, total=7500)
    msg = str(ei.value)
    assert "471/7,500" in msg and "7,029" in msg and "--seed/--subset" in msg


def test_cache_gate_allows_the_valid_cases():
    T.require_local_cache("cache", {"overfit", "train"}, present=7500, missing=0, total=7500)   # complete
    T.require_local_cache("cache", {"tokenizer", "prepare"}, present=0, missing=7500, total=7500)  # prepare builds it
    T.require_local_cache("stream", {"overfit", "train"}, present=0, missing=7500, total=7500)     # explicit stream


# ---------------------------------------------------------------- seed/subset determinism (21-Oct bug)
def _frame():
    rows = [{"uid": f"vid{v:03d}-{s}", "text": f"sentence {v} {s}"} for v in range(400) for s in range(3)]
    cleaned, _ = T.clean_dataframe(pd.DataFrame(rows))
    return cleaned


def _pilot(seed, subset=50):
    import contextlib, io
    a, b, c, _ = T.video_split(_frame(), seed)
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te = T.apply_subset(a, b, c, subset, seed)
    return [list(map(str, f["uid"])) for f in (tr, va, te)]


def test_same_seed_gives_identical_pilot_identities():
    assert _pilot(42) == _pilot(42)


def test_different_seed_changes_split_and_subset_so_mixing_seeds_is_detectable():
    a, b = _pilot(42), _pilot(1)
    assert a != b
    ids42, ids1 = {u for f in a for u in f}, {u for f in b for u in f}
    assert len(ids42 & ids1) < 0.5 * len(ids42)        # a cache built with seed 42 would NOT cover a seed-1 run
    # ...and the strict gate turns that mismatch into a hard stop instead of remote streaming:
    present = len(ids42 & ids1)
    with pytest.raises(SystemExit):
        T.require_local_cache("cache", {"train"}, present, len(ids1) - present, len(ids1))


def test_pilot_split_has_no_video_leakage():
    a, b, c, _ = T.video_split(_frame(), 42)
    v = [set(f["video_id"]) for f in (a, b, c)]
    assert v[0].isdisjoint(v[1]) and v[0].isdisjoint(v[2]) and v[1].isdisjoint(v[2])


# ---------------------------------------------------------------- interrupted gate != quality failure
def test_interrupted_gate_is_reported_as_interrupted(tmp_path, monkeypatch):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("('Connection aborted.', ConnectionResetError(104,'x'))")

    monkeypatch.setattr(T, "run_epoch", boom)
    with pytest.raises(requests.exceptions.ConnectionError):
        T.main(["--smoke", "--out", str(tmp_path), "--stages", "tokenizer,overfit"])
    rep = json.loads((tmp_path / "gate_report.json").read_text())
    assert calls["n"] >= 1
    assert rep["passed"] is False and rep["status"] == "interrupted"
    assert rep["quality_verdict"] is None and rep["final"] is None
    assert "ConnectionError" in rep["error"] and rep["pose_source"] == "synthetic"
