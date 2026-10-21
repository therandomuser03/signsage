#!/usr/bin/env python3
"""Regression tests for the strict identity-based cache gate in the canonical
notebook (signsage_kaggle.ipynb, cell 12 training gate).

The gate replaced the old near-complete count check
(`cached_n < 0.9 * 7500`) with exact set equality between:

  * the expected pilot identities, re-derived from auto_subset.json using the
    SAME selection logic as cache preparation + training
    (clean_dataframe -> video_split(seed=42) -> apply_subset(seed=42)), and
  * the manifest's cached identities (uid_to_shard.keys()), after validating
    the manifest schema (version, shard references, completion state, n).

These tests derive the expected identities from the real staged iSign CSV
(so they exercise the actual derivation path) and run the gate's validation +
exact-equality comparison against synthetic manifests. They also assert the
notebook cell source no longer contains the old count-based gate.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as _pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "research" / "training"))
sys.path.insert(0, str(REPO))
import train_sign_to_text as _T  # real deployed code under test

NOTEBOOK = REPO / "research" / "notebooks" / "signsage_kaggle.ipynb"
STAGED_CSV = REPO / "staging_kaggle_results" / "isign" / "iSign_v1.1.csv"
AUTO_SUBSET = REPO / "staging_kaggle_results" / "s2t_v2" / "auto_subset.json"

POSE_CACHE_VERSION = 1
POSE_CACHE_SHARD_PREFIX = "shard-"
POSE_CACHE_SHARD_SUFFIX = ".zip"
SEED = 42


def _load_notebook():
    with NOTEBOOK.open() as f:
        return json.load(f)


def _cell_source(nb, idx):
    cell = nb["cells"][idx]
    if cell["cell_type"] != "code":
        raise AssertionError(f"Cell {idx} is not a code cell")
    return "".join(cell["source"])


# --------------------------------------------------------------------------
# Expected-identity derivation (mirrors the notebook gate exactly).
# --------------------------------------------------------------------------

def _derive_expected_uids(csv_path, subset, seed=SEED):
    """Re-derive the pilot identities the gate expects.

    Mirrors the notebook gate: clean_dataframe -> video_split(seed) ->
    apply_subset(seed), then union uids across train/val/test.
    """
    from audit_dataset import clean_dataframe, video_split
    from train_sign_to_text import apply_subset

    df = _pd.read_csv(csv_path)
    cleaned, info = clean_dataframe(df)
    if info["duplicate_uid"]:
        raise AssertionError(f"staged CSV has {info['duplicate_uid']} duplicate UIDs")
    tr, va, te, _split = video_split(cleaned, seed)
    tr, va, te = apply_subset(tr, va, te, subset, seed)
    return {str(u) for u in list(tr["uid"]) + list(va["uid"]) + list(te["uid"])}


# --------------------------------------------------------------------------
# Gate under test: validation + exact set equality (mirrors notebook cell).
# --------------------------------------------------------------------------

def _gate_errors(manifest, expected_uids, cache_dir):
    """Run the notebook gate's manifest validation + exact-equality comparison.

    Returns the list of failure messages (empty == pass). Mirrors cell 12's
    gate block #2, including POSE_CACHE_VERSION, shard-prefix/suffix and SEED.
    """
    errors = []

    def fail(msg):
        errors.append(msg)

    if manifest.get("version") != POSE_CACHE_VERSION:
        fail(f"pose cache version {manifest.get('version')} != {POSE_CACHE_VERSION}: incompatible.")
    if not isinstance(manifest.get("uid_to_shard"), dict) or not manifest.get("uid_to_shard"):
        fail("manifest uid_to_shard is empty or not a dict: no cached samples.")
        return errors
    uid_to_shard = manifest["uid_to_shard"]

    # duplicate/invalid identities must not be silently ignored.
    _bad_keys = [k for k in uid_to_shard if not isinstance(k, str) or not k.strip()]
    if _bad_keys:
        fail(f"manifest has {len(_bad_keys)} invalid/empty identity key(s), e.g. {_bad_keys[:3]!r}: refusing.")

    # valid shard references: non-empty string naming an existing shard file.
    _shard_files = {p.name for p in cache_dir.glob(f"{POSE_CACHE_SHARD_PREFIX}*{POSE_CACHE_SHARD_SUFFIX}")}
    _bad_shards = [
        s for s in uid_to_shard.values()
        if not isinstance(s, str) or not s.strip() or s not in _shard_files
    ]
    if _bad_shards:
        fail(f"manifest has {len(_bad_shards)} invalid/missing shard reference(s), e.g. {_bad_shards[:3]!r}: refusing.")

    if manifest.get("complete") is not True:
        fail(f"manifest completion state is not True (complete={manifest.get('complete')!r}): cache incomplete.")
    if int(manifest.get("n", -1)) != len(uid_to_shard):
        fail(f"manifest n={manifest.get('n')!r} != len(uid_to_shard)={len(uid_to_shard)}: inconsistent manifest.")

    # identity comparison delegates to the REAL helper cell 12 calls
    # (_T.check_cache_identity), so the test exercises deployed code.
    missing, extra = _T.check_cache_identity(uid_to_shard, expected_uids)
    if missing or extra:
        fail(
            f"pose cache identity mismatch: missing={len(missing)} (e.g. {missing[:5]}), "
            f"extra={len(extra)} (e.g. {extra[:5]}), expected={len(expected_uids)}, "
            f"cached={len(uid_to_shard)}."
        )
    return errors


def _make_manifest(uids, shard_name="shard-00000.zip", version=POSE_CACHE_VERSION, complete=True, n=None):
    return {
        "version": version,
        "uid_to_shard": {u: shard_name for u in uids},
        "complete": complete,
        "n": len(uids) if n is None else n,
    }


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_derives_expected_uids_matches_subset_config():
    """Expected identities are derived from auto_subset.json's subset, not a
    hard-coded 7,500; count = 3 * subset per the pilot configuration."""
    subset = int(json.loads(AUTO_SUBSET.read_text())["subset"])
    expected = _derive_expected_uids(STAGED_CSV, subset, SEED)
    assert len(expected) == 3 * subset, (len(expected), subset)
    # Staged auto_subset.json selects 2500 per split -> 7,500 expected identities.
    assert subset == 2500
    assert len(expected) == 7500


def test_exact_match_passes():
    """A complete, exact-match manifest passes (0 errors)."""
    expected = _derive_expected_uids(STAGED_CSV, 2500, SEED)
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        (cache_dir / "shard-00000.zip").write_bytes(b"")
        manifest = _make_manifest(expected)
        errors = _gate_errors(manifest, expected, cache_dir)
    assert errors == [], errors


def test_missing_one_identity_fails():
    """A manifest missing one expected identity fails."""
    expected = _derive_expected_uids(STAGED_CSV, 2500, SEED)
    missing_uid = sorted(expected)[0]
    partial = expected - {missing_uid}
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        (cache_dir / "shard-00000.zip").write_bytes(b"")
        manifest = _make_manifest(partial)
        errors = _gate_errors(manifest, expected, cache_dir)
    assert errors, "expected a failure for a missing identity"
    assert any("missing=1" in e for e in errors), errors
    assert any(missing_uid in e for e in errors), errors


def test_extra_unexpected_identity_fails():
    """A manifest containing an unexpected identity fails."""
    expected = _derive_expected_uids(STAGED_CSV, 2500, SEED)
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        (cache_dir / "shard-00000.zip").write_bytes(b"")
        manifest = _make_manifest(set(expected) | {"unexpected-uid-99999"})
        errors = _gate_errors(manifest, expected, cache_dir)
    assert errors, "expected a failure for an extra identity"
    assert any("extra=1" in e for e in errors), errors
    assert any("unexpected-uid-99999" in e for e in errors), errors


def test_near_complete_7400_of_7500_fails():
    """A near-complete cache (7,400/7,500) can no longer pass: exact equality
    requires every expected identity, so a 98.67% count still fails."""
    expected = _derive_expected_uids(STAGED_CSV, 2500, SEED)
    assert len(expected) == 7500
    near_complete = set(sorted(expected)[:7400])  # 7,400/7,500
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        (cache_dir / "shard-00000.zip").write_bytes(b"")
        manifest = _make_manifest(near_complete)
        errors = _gate_errors(manifest, expected, cache_dir)
    assert errors, "expected a failure for a 7,400/7,500 near-complete cache"
    assert any("missing=100" in e for e in errors), errors


def test_stale_incompatible_manifest_fails():
    """A stale/incompatible manifest (wrong version, or scoped to a different
    subset/configuration) fails."""
    expected = _derive_expected_uids(STAGED_CSV, 2500, SEED)
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        (cache_dir / "shard-00000.zip").write_bytes(b"")
        # Wrong cache version -> fail independently of identity.
        manifest = _make_manifest(expected, version=999)
        errors = _gate_errors(manifest, expected, cache_dir)
        assert errors and any("version" in e for e in errors), errors
        # Stale manifest scoped to a different subset (500 != 2500) -> identity mismatch.
        subset_500 = _derive_expected_uids(STAGED_CSV, 500, SEED)
        manifest = _make_manifest(subset_500)
        errors = _gate_errors(manifest, expected, cache_dir)
        assert errors and any("missing=" in e for e in errors), errors


def test_invalid_or_duplicate_identity_metadata_fails():
    """Invalid (empty / non-string) identities and an inconsistent 'n' or
    completion state must not be silently ignored."""
    expected = _derive_expected_uids(STAGED_CSV, 2500, SEED)
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        (cache_dir / "shard-00000.zip").write_bytes(b"")
        # Empty identity key (invalid metadata).
        manifest = _make_manifest(set(expected) | {""})
        errors = _gate_errors(manifest, expected, cache_dir)
        assert errors and any("invalid/empty identity key" in e for e in errors), errors
        # Non-string identity key (invalid metadata).
        manifest = _make_manifest(set(expected) | {12345})
        errors = _gate_errors(manifest, expected, cache_dir)
        assert errors and any("invalid/empty identity key" in e for e in errors), errors
        # Inconsistent n (n != len(uid_to_shard)).
        manifest = _make_manifest(expected, n=0)
        errors = _gate_errors(manifest, expected, cache_dir)
        assert errors and any("inconsistent manifest" in e for e in errors), errors
        # completion state not True.
        manifest = _make_manifest(expected, complete=False)
        errors = _gate_errors(manifest, expected, cache_dir)
        assert errors and any("completion state is not True" in e for e in errors), errors


def test_load_strict_manifest_rejects_duplicate_keys():
    """A manifest file containing duplicate UID keys is rejected by the real
    loader (_T.load_strict_manifest, used by cell 12) before keys can be
    silently overwritten by stdlib json."""
    expected = _derive_expected_uids(STAGED_CSV, 2500, SEED)
    uid = sorted(expected)[0]
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        (cache_dir / "shard-00000.zip").write_bytes(b"")
        # Same UID key appears twice -> stdlib json would silently keep the
        # last value; the object_pairs_hook must raise instead.
        raw = (
            '{"version":1,"uid_to_shard":{"%s":"shard-00000.zip","%s":"shard-00000.zip"},'
            '"complete":true,"n":1}'
        ) % (uid, uid)
        manifest_path = cache_dir / "manifest.json"
        manifest_path.write_text(raw)
        try:
            _T.load_strict_manifest(manifest_path)
        except ValueError as exc:
            assert "duplicate JSON key" in str(exc), str(exc)
            return
    raise AssertionError("load_strict_manifest did not reject duplicate keys")


# --------------------------------------------------------------------------
# Structural source checks on the notebook cell (guard the fix itself).
# --------------------------------------------------------------------------

def _gate_cell_source():
    nb = _load_notebook()
    # cell 12 holds the STRICT PRE-TRAINING CACHE GATES block.
    return _cell_source(nb, 12)


def test_notebook_cell_has_no_count_based_gate():
    """The old near-complete count gate (0.9 * 7500 / _expected_n = 7500) is
    gone from the notebook cell."""
    src = _gate_cell_source()
    assert "0.9 * 7500" not in src, "old 90%-of-7500 count gate still present"
    assert "_expected_n = 7500" not in src, "hard-coded 7,500 expected count still present"
    assert "differs from full pilot 7500" not in src, "hard-coded 7,500 message still present"


def test_notebook_cell_derives_from_auto_subset():
    """The notebook cell derives the pilot scope from auto_subset.json's
    'subset' key (validated pilot configuration), not a hard-coded value."""
    src = _gate_cell_source()
    assert 'auto_subset.json' in src, "cell does not read auto_subset.json"
    assert '.get("subset", 0)' in src, "cell does not read the 'subset' key"
    assert 'expected_n != 3 * _subset_size' in src, "cell does not derive expected count from subset"


def test_notebook_cell_uses_exact_set_equality():
    """The notebook cell compares expected vs manifest identities by exact
    set equality (missing + extra), not a count threshold, delegating manifest
    parsing and identity comparison to the real train_sign_to_text helpers."""
    src = _gate_cell_source()
    assert "_T.load_strict_manifest(_man_path)" in src, "cell must parse manifest with the strict loader"
    assert "_T.check_cache_identity(uid_to_shard, expected_ids)" in src, "cell must compare identities via the real helper"
    assert "if _missing or _extra:" in src, "cell must fail on missing/extra identities"
    assert "0.9 * 7500" not in src, "old 90%-of-7500 count gate still present"
    assert "_expected_n = 7500" not in src, "hard-coded 7,500 expected count still present"


if __name__ == "__main__":
    tests = [
        test_derives_expected_uids_matches_subset_config,
        test_exact_match_passes,
        test_missing_one_identity_fails,
        test_extra_unexpected_identity_fails,
        test_near_complete_7400_of_7500_fails,
        test_stale_incompatible_manifest_fails,
        test_invalid_or_duplicate_identity_metadata_fails,
        test_notebook_cell_has_no_count_based_gate,
        test_notebook_cell_derives_from_auto_subset,
        test_notebook_cell_uses_exact_set_equality,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
