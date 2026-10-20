"""Focused regression tests for cache-coverage and HTTP 503 handling.

Covers the fixes made after the Kaggle run that failed mid-overfit-gate with
`RuntimeError: range request failed: HTTP 503` after preparation had already
printed `PREPARE COMPLETE` while only 471/7,500 pilot samples were cached.

Tests use mocks/fixtures only -- no network, no GPU, no Kaggle, no training.
"""

import io
import json
import os
import sys
import tempfile
import zipfile
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, "research/experiments")
sys.path.insert(0, "backend")

import requests  # noqa: E402
import train_sign_to_text as T  # noqa: E402

NOTEBOOK = Path("research/notebooks/signsage_kaggle.ipynb")


class FakeResponse:
    """Minimal requests.Response substitute for range-request tests."""

    def __init__(self, status_code, headers=None, content=b"", url="http://x"):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self.url = url


class FakeSession:
    """Session whose get() returns responses from a caller-supplied queue."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, timeout=None, allow_redirects=True):
        self.calls.append((url, headers, timeout, allow_redirects))
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# 1 & 2. cache_coverage: partial coverage is not complete; complete coverage is.
# ---------------------------------------------------------------------------

def _write_shard(cache_dir, name, uids):
    with zipfile.ZipFile(cache_dir / name, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for uid in uids:
            zf.writestr(f"{uid}.pose", b"x" * 100)


def test_cache_coverage_partial_is_not_complete(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Manifest lists only the uids physically present; u3 is simply not indexed.
    manifest = {
        "version": T.POSE_CACHE_VERSION,
        "uid_to_shard": {"u1": "shard-00000.zip", "u2": "shard-00000.zip"},
        "complete": False,
        "n": 2,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    _write_shard(cache_dir, "shard-00000.zip", ["u1", "u2"])  # u3 not present
    present, missing = T.cache_coverage(cache_dir, ["u1", "u2", "u3"])
    assert present == 2 and missing == 1, (present, missing)


def test_cache_coverage_complete_can_complete(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = {
        "version": T.POSE_CACHE_VERSION,
        "uid_to_shard": {"u1": "shard-00000.zip", "u2": "shard-00000.zip", "u3": "shard-00001.zip"},
        "complete": True,
        "n": 3,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    _write_shard(cache_dir, "shard-00000.zip", ["u1", "u2"])
    _write_shard(cache_dir, "shard-00001.zip", ["u3"])
    present, missing = T.cache_coverage(cache_dir, ["u1", "u2", "u3"])
    assert present == 3 and missing == 0, (present, missing)


# ---------------------------------------------------------------------------
# 3. materialize_pose_cache fails closed when fetches fail (no false COMPLETE).
# ---------------------------------------------------------------------------

class FailingStore:
    def read_member_bytes(self, uid):
        raise RuntimeError("remote down")


def test_materialize_pose_cache_fails_closed_on_fetch_errors(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    uids = ["u1", "u2", "u3"]
    try:
        T.materialize_pose_cache(
            uids, FailingStore(), cache_dir, jobs=1,
            source_factory=lambda: FailingStore(),
        )
        raise AssertionError("expected SystemExit for incomplete cache")
    except SystemExit as exc:
        assert "incomplete" in str(exc).lower(), str(exc)
    # A partial cache is kept on disk (not deleted) for resume.
    assert cache_dir.exists()


# ---------------------------------------------------------------------------
# 4 & 5. _fetch retry semantics on transient 503 vs hard failure / corrupt data.
# ---------------------------------------------------------------------------

def _make_rcf(sizes=(1000,), chunk_size=1024):
    rcf = object.__new__(T.RemoteConcatFile)
    rcf.sizes = list(sizes)
    rcf.starts = [0]
    rcf.total_size = sum(sizes)
    rcf.resolved = {}
    rcf.cache = OrderedDict()
    rcf.chunk_size = chunk_size
    rcf.token = "test-token"
    rcf.urls = [f"http://example.test/part_{i}" for i in range(len(sizes))]
    # __init__ is bypassed, so set the requests module reference used by the
    # except clause in _fetch.
    rcf._requests = requests
    return rcf


def _ok_response(chunk_start, chunk_end, total):
    content = b"Y" * (chunk_end - chunk_start + 1)
    headers = {"Content-Range": f"bytes {chunk_start}-{chunk_end}/{total}"}
    return FakeResponse(206, headers, content)


def _five_oh_three():
    return FakeResponse(503, {}, b"")


def test_transient_503_retried_then_success():
    rcf = _make_rcf()
    rcf.session = FakeSession([_five_oh_three(), _ok_response(0, 999, 1000)])
    piece = rcf._fetch(0, 0, 10)
    assert piece == b"Y" * 10
    assert len(rcf.session.calls) == 2  # one 503 retry, then success


def test_repeated_503_fails_after_bounded_attempts():
    rcf = _make_rcf()
    rcf.session = FakeSession([_five_oh_three()] * 5)
    try:
        rcf._fetch(0, 0, 10)
        raise AssertionError("expected error after bounded 503 retries")
    except (ConnectionError, RuntimeError) as exc:
        assert "503" in str(exc) or "fetch failed" in str(exc)
    assert len(rcf.session.calls) == 5  # exactly 5 attempts, no indefinite retry


def test_short_range_body_is_rejected_not_accepted():
    rcf = _make_rcf()
    # Content-Range claims the full chunk (1000 bytes) but only 100 are sent.
    headers = {"Content-Range": "bytes 0-999/1000"}
    rcf.session = FakeSession([FakeResponse(206, headers, b"Y" * 100)])
    try:
        rcf._fetch(0, 0, 10)
        raise AssertionError("expected RuntimeError for incomplete range body")
    except RuntimeError as exc:
        assert "incomplete range body" in str(exc)


# ---------------------------------------------------------------------------
# 6. Cell 11 failure classification: infrastructure vs model-quality verdict.
# ---------------------------------------------------------------------------

def _cell_11_source():
    import json as _json
    with NOTEBOOK.open() as f:
        nb = _json.load(f)
    return "".join(nb["cells"][11]["source"])


def test_cell_11_has_infrastructure_classification():
    src = _cell_11_source()
    assert "HTTP 503" in src, "cell 11 must detect HTTP 503"
    assert "data-access / infrastructure failure" in src, (
        "cell 11 must classify infra failures separately"
    )
    assert "not a model-quality verdict" in src


# Mirror of cell 11's classification logic (kept in sync with the notebook).
def classify_training_outcome(train_out, collapsed, gate_passed, gate_absent):
    infra_keywords = ("HTTP 503", "range request failed", "pose cache incomplete",
                      "ConnectionError", "fetch failed", "Fetch failed")
    infra_failure = any(k.lower() in train_out.lower() for k in infra_keywords)
    if collapsed:
        return "diversity-collapse failure"
    if infra_failure:
        return "data-access / infrastructure failure"
    if gate_absent:
        return "other training failure"
    if gate_passed is False:
        return "overfit-gate failure"
    if gate_passed is True:
        return "training error after passing gate"
    return "other training failure"


def test_interrupted_overfit_gate_classified_as_infrastructure():
    # Gate interrupted by HTTP 503: infrastructure, not overfit-gate.
    assert classify_training_outcome(
        "RuntimeError: range request failed: HTTP 503",
        collapsed=False, gate_passed=False, gate_absent=False,
    ) == "data-access / infrastructure failure"


def test_collapse_still_classified_as_collapse_when_no_infra():
    assert classify_training_outcome(
        "some normal training output",
        collapsed=True, gate_passed=False, gate_absent=True,
    ) == "diversity-collapse failure"


def test_other_failure_when_no_infra_and_no_collapse():
    assert classify_training_outcome(
        "some normal training output",
        collapsed=False, gate_passed=None, gate_absent=True,
    ) == "other training failure"


def test_overfit_gate_when_infra_absent():
    assert classify_training_outcome(
        "some normal training output",
        collapsed=False, gate_passed=False, gate_absent=False,
    ) == "overfit-gate failure"


if __name__ == "__main__":
    import traceback

    tests = [
        test_cache_coverage_partial_is_not_complete,
        test_cache_coverage_complete_can_complete,
        test_materialize_pose_cache_fails_closed_on_fetch_errors,
        test_transient_503_retried_then_success,
        test_repeated_503_fails_after_bounded_attempts,
        test_short_range_body_is_rejected_not_accepted,
        test_cell_11_has_infrastructure_classification,
        test_interrupted_overfit_gate_classified_as_infrastructure,
        test_collapse_still_classified_as_collapse_when_no_infra,
        test_other_failure_when_no_infra_and_no_collapse,
        test_overfit_gate_when_infra_absent,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
