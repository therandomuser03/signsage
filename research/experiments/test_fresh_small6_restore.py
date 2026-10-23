"""Focused tests for fresh small6 restore initialization vs restore verification.

Tests that:
1. A fresh session with no prior s2t_small6 artifacts skips verification and
   initializes state for a clean start (no mandatory manifest check).
2. An existing partial restore with missing manifest fails closed.
3. Downstream cells use /kaggle/working/s2t_small6 consistently.
4. No --resume or --force in training cell; no full-v2 artifact reuse.
"""

import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import tempfile
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "signsage_kaggle.ipynb"


def _load_notebook():
    with NOTEBOOK.open() as f:
        return json.load(f)


def _cell_source(nb, idx):
    from _notebook_cells import legacy_cell_source
    return legacy_cell_source(nb, idx)


def _restore_cell_source(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "_T.POSE_CACHE_VERSION" in src:
            return src
    raise AssertionError("restore cell not found")


# ---------------------------------------------------------------------------
# Test 1: Fresh run skips verification entirely
# ---------------------------------------------------------------------------

def test_fresh_run_skips_verification():
    """When no prior artifacts are found (restored is None), the restore cell
    must skip the pose-cache manifest check and proceed cleanly."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    # Fresh-run branch must exist
    assert "if not restored:" in src, (
        "Restore cell must have `if not restored:` fresh-run branch"
    )
    assert "FRESH RUN" in src, (
        "Fresh run must print FRESH RUN message"
    )
    assert "_reuse_lock = False" in src, (
        "Fresh run must set _reuse_lock = False"
    )

    # The manifest check must be INSIDE `if restored:` block, not at top level
    # Find the manifest check and verify it's indented under `if restored:`
    lines = src.split("\n")
    in_if_restored = False
    manifest_check_indented = False
    for line in lines:
        stripped = line.strip()
        if stripped == "if restored:":
            in_if_restored = True
            continue
        if in_if_restored and stripped.startswith("_fail(\"pose cache manifest MISSING\")"):
            manifest_check_indented = True
            break
        # If we hit a dedent (e.g., `else:` at same level as `if restored:`)
        if in_if_restored and stripped and not line.startswith(" "):
            break

    assert manifest_check_indented, (
        "Pose-cache manifest MISSING check must be inside `if restored:` block"
    )


# ---------------------------------------------------------------------------
# Test 2: Fresh run initializes state consistently
# ---------------------------------------------------------------------------

def test_fresh_run_initializes_state():
    """On a fresh run, _reuse_lock must be False and _locked_subset must be 0,
    so measure/certify cells run fresh."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    # Check the else branch (fresh run) sets both vars
    lines = src.split("\n")
    in_else = False
    found_reuse_false = False
    found_subset_zero = False
    for line in lines:
        stripped = line.strip()
        if stripped == "else:":
            in_else = True
            continue
        if in_else:
            if "_reuse_lock = False" in stripped:
                found_reuse_false = True
            if "_locked_subset = 0" in stripped:
                found_subset_zero = True
            # Stop at next dedent
            if stripped and not line.startswith("    "):
                break

    assert found_reuse_false, "Fresh run else branch must set _reuse_lock = False"
    assert found_subset_zero, "Fresh run else branch must set _locked_subset = 0"


# ---------------------------------------------------------------------------
# Test 3: Restore branch keeps strict fail-closed validation
# ---------------------------------------------------------------------------

def test_restore_branch_strict_validation():
    """When prior artifacts are restored, the cell must still require the
    manifest, tokenizer files, checkpoint integrity, architecture config,
    and provenance fingerprints."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    # Manifest check inside if restored:
    assert '_fail("pose cache manifest MISSING")' in src, (
        "Restore branch must fail on missing manifest"
    )
    assert '_fail("tokenizer model MISSING")' in src, (
        "Restore branch must fail on missing tokenizer model"
    )
    assert '_fail("recovery checkpoint MISSING")' in src, (
        "Restore branch must fail on missing recovery checkpoint"
    )
    assert '_fail("best checkpoint MISSING")' in src, (
        "Restore branch must fail on missing best checkpoint"
    )
    assert '_fail(f"checkpoint model_config mismatch' in src, (
        "Restore branch must fail on model_config mismatch"
    )
    assert '_fail(f"tokenizer_fingerprint mismatch' in src, (
        "Restore branch must fail on tokenizer_fingerprint mismatch"
    )
    assert '_T.POSE_CACHE_VERSION' in src, (
        "Restore branch must validate pose cache version"
    )


# ---------------------------------------------------------------------------
# Test 4: Incomplete existing restore fails closed
# ---------------------------------------------------------------------------

def test_incomplete_restore_fails_closed():
    """Simulate an existing restore with missing manifest: the cell must
    raise RuntimeError via _fail when REQUIRE_RESTORE=True."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    # REQUIRE_RESTORE must be True (default)
    assert "REQUIRE_RESTORE = True" in src, (
        "REQUIRE_RESTORE must default to True for fail-closed behavior"
    )

    # _fail must raise RuntimeError when REQUIRE_RESTORE is True
    assert 'raise RuntimeError("RESTORE FAILED: " + msg)' in src, (
        "_fail must raise RuntimeError when REQUIRE_RESTORE=True"
    )


# ---------------------------------------------------------------------------
# Test 5: Downstream cells use s2t_small6 consistently
# ---------------------------------------------------------------------------

def test_downstream_cells_use_s2t_small6():
    """All downstream cells (measure, certify, prepare, train) must use
    /kaggle/working/s2t_small6 consistently."""
    nb = _load_notebook()
    for idx in [7, 8, 9, 10]:  # measure, certify, prepare, train cells
        src = _cell_source(nb, idx)
        assert "/kaggle/working/s2t_small6" in src, (
            f"Cell {idx} must use /kaggle/working/s2t_small6 output path"
        )


# ---------------------------------------------------------------------------
# Test 6: Training cell has no --resume or --force
# ---------------------------------------------------------------------------

def test_training_cell_no_resume_or_force():
    """The training cell must not use --resume or --force for a fresh
    small6 run."""
    nb = _load_notebook()
    src = _cell_source(nb, 11)  # training cell

    assert '("--resume" in cmd) == RESUME_TRAINING' in src and 'RESUME_TRAINING = False' in src, (
        "Training cell must assert --resume not in cmd"
    )
    assert '--force" not in cmd' in src, (
        "Training cell must assert --force not in cmd"
    )


# ---------------------------------------------------------------------------
# Test 7: No full-v2 artifact reuse
# ---------------------------------------------------------------------------

def test_no_full_v2_reuse():
    """No cell may reference /kaggle/working/s2t_v2 in code -- full-v2
    artifacts must never be the target of small6 training."""
    nb = _load_notebook()
    found = False
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for line in src.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if "s2t_v2" in stripped:
                print(f"  Cell {i}: {stripped[:100]}")
                found = True
    assert not found, (
        "No s2t_v2 code references allowed -- full-v2 output dir must not be used"
    )


# ---------------------------------------------------------------------------
# Test 8: Restore cell targets s2t_small6 directory by name
# ---------------------------------------------------------------------------

def test_restore_targets_s2t_small6():
    """The restore cell must search for s2t_small6 directories, not s2t_v2."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    assert f's2t_{{EXPERIMENT_ID}}' in src, (
        "Restore cell must use experiment-scoped directory name"
    )
    assert '"s2t_v2" in dirnames' not in src, (
        "Restore cell must not search for s2t_v2 directories"
    )


# ---------------------------------------------------------------------------
# Test 9: Fresh run cannot reuse full-v2 artifacts
# ---------------------------------------------------------------------------

def test_fresh_run_cannot_reuse_full_v2():
    """On a fresh run, the restore cell does not look for or copy any
    s2t_v2 artifacts. The OUT directory is s2t_small6 only."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    assert 'OUT = f"/kaggle/working/s2t_{EXPERIMENT_ID}"' in src, (
        "OUT must be experiment-scoped to s2t_small6"
    )
    assert 'OUT = "/kaggle/working/s2t_v2"' not in src, (
        "OUT must not be hardcoded to s2t_v2"
    )


if __name__ == "__main__":
    import sys
    tests = [
        test_fresh_run_skips_verification,
        test_fresh_run_initializes_state,
        test_restore_branch_strict_validation,
        test_incomplete_restore_fails_closed,
        test_downstream_cells_use_s2t_small6,
        test_training_cell_no_resume_or_force,
        test_no_full_v2_reuse,
        test_restore_targets_s2t_small6,
        test_fresh_run_cannot_reuse_full_v2,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
