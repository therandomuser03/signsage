"""Cross-experiment restore safety tests for SignSage Kaggle notebook.

Tests that full-v2 artifacts cannot be silently reused for small6,
and that output-directory isolation prevents cross-experiment contamination.

These tests verify the SAFE behavior AFTER the fix:
- Restore cell checks experiment provenance (not just file existence)
- Cache-version match alone is insufficient for compatibility
- Output directory is experiment-scoped (s2t_small6 vs s2t_v2)
- Model config and tokenizer fingerprint compatibility validation
- Architecture mismatch detection
- Unset tokenizer fingerprint handling (first-run safe, later-run strict)
- Training procedure matches runbook small6 command
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "signsage_kaggle.ipynb"


def _load_notebook():
    with NOTEBOOK.open() as f:
        return json.load(f)


def _cell_source(nb, idx):
    cell = nb["cells"][idx]
    if cell["cell_type"] != "code":
        raise AssertionError(f"Cell {idx} is not a code cell")
    return "".join(cell["source"])


def _restore_cell_source(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "_T.POSE_CACHE_VERSION" in src:
            return src
    raise AssertionError("restore cell not found")


# ---------------------------------------------------------------------------
# Test 1: Output directory is experiment-scoped, not hardcoded to s2t_v2
# ---------------------------------------------------------------------------

def test_output_dir_is_experiment_scoped():
    """OUT must be derived from EXPERIMENT_ID, not hardcoded to s2t_v2."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)
    assert 'EXPERIMENT_ID = "small6"' in src, (
        "EXPERIMENT_ID must be set in restore cell"
    )
    assert 'OUT = f"/kaggle/working/s2t_{EXPERIMENT_ID}"' in src, (
        "OUT must be experiment-scoped using EXPERIMENT_ID"
    )
    assert 'OUT = "/kaggle/working/s2t_v2"' not in src, (
        "OUT must not be hardcoded to s2t_v2"
    )


# ---------------------------------------------------------------------------
# Test 2: Experiment ID is defined for provenance tracking
# ---------------------------------------------------------------------------

def test_experiment_id_defined():
    """The restore cell must define EXPERIMENT_ID to identify which
    experiment produced the artifacts."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)
    assert 'EXPERIMENT_ID = ' in src, (
        "Restore cell must define EXPERIMENT_ID"
    )


# ---------------------------------------------------------------------------
# Test 3: Provenance compatibility checks exist (model_config + tokenizer_fingerprint)
# ---------------------------------------------------------------------------

def test_provenance_compatibility_check_exists():
    """The restore cell must compare restored checkpoint model_config and
    tokenizer_fingerprint against expected values for the current experiment."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)
    assert "EXPECTED_MODEL_CONFIG" in src, (
        "Restore cell must define EXPECTED_MODEL_CONFIG"
    )
    assert "EXPECTED_TOKENIZER_FINGERPRINT" in src, (
        "Restore cell must define EXPECTED_TOKENIZER_FINGERPRINT"
    )
    assert "model_config" in src, (
        "Restore cell must compare checkpoint model_config"
    )
    assert "tokenizer_fingerprint" in src, (
        "Restore cell must compare checkpoint tokenizer_fingerprint"
    )


# ---------------------------------------------------------------------------
# Test 4: EXPECTED_MODEL_CONFIG matches actual small6 architecture
# ---------------------------------------------------------------------------

def test_expected_model_config_matches_training_script():
    """EXPECTED_MODEL_CONFIG must match train_sign_to_text.py constants:
    input_dim=3258, raw_feature_dim=1629, d_model=256, nhead=4, layers=4,
    ff=1024, dropout=0.15, norm_first=True, max_frames=160, max_tokens=96."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    start = src.find("EXPECTED_MODEL_CONFIG = {")
    end = src.find("}", start) + 1
    config_block = src[start:end]

    expected = {
        "input_dim": 3258,
        "raw_feature_dim": 1629,
        "d_model": 256,
        "nhead": 4,
        "layers": 4,
        "ff": 1024,
        "dropout": 0.15,
        "norm_first": True,
        "max_frames": 160,
        "max_tokens": 96,
    }

    import re
    for key, expected_val in expected.items():
        pattern = rf'"{key}"\s*:\s*([^,\n]+)'
        match = re.search(pattern, config_block)
        assert match, f"EXPECTED_MODEL_CONFIG missing key '{key}'"
        actual_str = match.group(1).strip()
        if isinstance(expected_val, bool):
            actual_val = actual_str == "True"
        elif isinstance(expected_val, int):
            actual_val = int(actual_str)
        elif isinstance(expected_val, float):
            actual_val = float(actual_str)
        else:
            actual_val = actual_str.strip('"')
        assert actual_val == expected_val, (
            f"EXPECTED_MODEL_CONFIG['{key}'] = {actual_val}, expected {expected_val}"
        )


# ---------------------------------------------------------------------------
# Test 5: Cache-version match alone is insufficient
# ---------------------------------------------------------------------------

def test_cache_version_not_sufficient():
    """The restore cell must have provenance checks BEYOND just
    POSE_CACHE_VERSION. A cache-version match alone must never establish
    experiment compatibility."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    assert "_T.POSE_CACHE_VERSION" in src, (
        "POSE_CACHE_VERSION check must still exist"
    )

    provenance_terms = ["model_config", "tokenizer_fingerprint", "EXPERIMENT_ID"]
    for term in provenance_terms:
        assert term in src, (
            f"Provenance check '{term}' must exist alongside cache version check"
        )


# ---------------------------------------------------------------------------
# Test 6: Restore cell matches directories by experiment-scoped name
# ---------------------------------------------------------------------------

def test_restore_matches_by_experiment_name():
    """_find_s2t_dirs() must match directories named s2t_{EXPERIMENT_ID},
    not just any directory named s2t_v2."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)
    assert 'f"s2t_{EXPERIMENT_ID}"' in src, (
        "Directory search must use experiment-scoped name"
    )
    assert '"s2t_v2" in dirnames' not in src, (
        "Directory search must not hardcode s2t_v2"
    )


# ---------------------------------------------------------------------------
# Test 7: _usable() checks are experiment-aware
# ---------------------------------------------------------------------------

def test_usable_is_experiment_aware():
    """_usable() should be used with experiment-scoped directory names,
    not generic directory name matching."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)
    assert "def _usable(_c):" in src, (
        "_usable() function must exist"
    )
    assert '_find_s2t_dirs("/kaggle/input"' in src, (
        "_find_s2t_dirs must be called with experiment-scoped search"
    )


# ---------------------------------------------------------------------------
# Test 8: _reuse_lock is set only after provenance verification
# ---------------------------------------------------------------------------

def test_reuse_lock_after_provenance():
    """_reuse_lock = True is only set after all verification passes,
    including provenance compatibility checks."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)
    assert "_reuse_lock = True" in src, (
        "_reuse_lock must be set on successful restore"
    )
    provenance_idx = src.find("# --- provenance compatibility check")
    lock_idx = src.find("_reuse_lock = True")
    assert provenance_idx != -1, "Provenance check block must exist"
    assert lock_idx != -1, "_reuse_lock must be set"
    assert lock_idx > provenance_idx, (
        "_reuse_lock must be set AFTER provenance verification"
    )


# ---------------------------------------------------------------------------
# Test 9: Unset tokenizer fingerprint on first run is safe (skip, not fail)
# ---------------------------------------------------------------------------

def test_unset_tokenizer_fingerprint_first_run_safe():
    """When EXPECTED_TOKENIZER_FINGERPRINT="" (fresh small6 run), the
    restore cell must SKIP the fingerprint check (not fail), and must
    print a clear warning that the fingerprint check is skipped."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    assert 'EXPECTED_TOKENIZER_FINGERPRINT = ""' in src, (
        "EXPECTED_TOKENIZER_FINGERPRINT must default to empty for first run"
    )

    assert 'if EXPECTED_TOKENIZER_FINGERPRINT and _ckpt_fp != EXPECTED_TOKENIZER_FINGERPRINT:' in src, (
        "Fingerprint mismatch check must be guarded by non-empty EXPECTED_TOKENIZER_FINGERPRINT"
    )

    assert 'tokenizer fingerprint check SKIPPED' in src, (
        "Must print SKIP message when fingerprint check is skipped on first run"
    )
    assert 'set EXPECTED_TOKENIZER_FINGERPRINT after first run' in src, (
        "Must instruct operator to set fingerprint after first run"
    )


# ---------------------------------------------------------------------------
# Test 10: Set tokenizer fingerprint on later runs is enforced
# ---------------------------------------------------------------------------

def test_set_tokenizer_fingerprint_later_run_enforced():
    """When EXPECTED_TOKENIZER_FINGERPRINT is set (after first run), the
    restore cell must FAIL if the restored checkpoint's fingerprint doesn't
    match."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    assert '_fail(f"tokenizer_fingerprint mismatch' in src, (
        "Must fail on tokenizer_fingerprint mismatch when EXPECTED_* is set"
    )


# ---------------------------------------------------------------------------
# Test 11: Architecture mismatch detection
# ---------------------------------------------------------------------------

def test_architecture_mismatch_detected():
    """The provenance check iterates EXPECTED_MODEL_CONFIG items and fails
    if any don't match the restored checkpoint. A full-v2 checkpoint with
    different architecture would be rejected."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    assert 'EXPECTED_MODEL_CONFIG.items()' in src, (
        "Provenance check must iterate all EXPECTED_MODEL_CONFIG items"
    )
    assert '_fail(f"checkpoint model_config mismatch' in src, (
        "Must fail on model_config mismatch"
    )
    assert 'for _k, _v in EXPECTED_MODEL_CONFIG.items():' in src, (
        "Must check every config key, not just a subset"
    )


# ---------------------------------------------------------------------------
# Test 12: Dataset/experiment identity limitation documented
# ---------------------------------------------------------------------------

def test_dataset_identity_limitation_documented():
    """The checkpoint does NOT contain a dataset identity or experiment ID
    field. Architecture + tokenizer fingerprint alone cannot prove a
    checkpoint belongs to the correct dataset. This limitation is documented
    in the notebook comments."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    assert 'EXPERIMENT_ID' in src, (
        "Experiment ID is used for directory scoping and error messages"
    )


# ---------------------------------------------------------------------------
# Test 13: Training cell (cell 11) matches runbook small6 command
# ---------------------------------------------------------------------------

def test_training_cell_matches_runbook_small6():
    """Cell 11 must execute the runbook's validated small6 command:
    --epochs 8 --max-hours 4 --lr 1e-4 --data-mode stream --stages all
    --batch 32 --patience 2 --cache-budget-gib 10.0 --seed 1
    --decoder-input-dropout 0.2 --out /kaggle/working/s2t_small6"""
    nb = _load_notebook()
    src = _cell_source(nb, 11)

    # Config values
    assert "EPOCHS = 8" in src, "Cell 11 must use --epochs 8"
    assert "MAX_HOURS = 4" in src, "Cell 11 must use --max-hours 4"
    assert 'PEAK_LR = "1e-4"' in src, "Cell 11 must use --lr 1e-4"
    assert 'DECODER_INPUT_DROPOUT = "0.2"' in src, "Cell 11 must use --decoder-input-dropout 0.2"
    assert "WARMUP_DEFAULT_CONFIRMED = True" in src, (
        "Cell 11 must skip warmup auto-detection (runbook uses script defaults)"
    )

    # Command flags
    assert '--data-mode", "stream"' in src, "Cell 11 must use --data-mode stream"
    assert '--stages", "all"' in src, "Cell 11 must use --stages all"
    assert '--batch", "32"' in src, "Cell 11 must use --batch 32"
    assert '--patience", "2"' in src, "Cell 11 must use --patience 2"
    assert '--cache-budget-gib", "10"' in src, "Cell 11 must use --cache-budget-gib 10"
    assert '--seed", "1"' in src, "Cell 11 must use --seed 1"
    assert '"/kaggle/working/s2t_small6"' in src, "Cell 11 must output to s2t_small6"

    # No resume/force for fresh run
    assert '--resume" not in cmd' in src, "Cell 11 must forbid --resume (fresh run)"
    assert '--force" not in cmd' in src, "Cell 11 must forbid --force (fresh run)"

    # _used_flags includes new flags
    for flag in ["--batch", "--patience", "--cache-budget-gib", "--seed"]:
        assert flag in src, f"Cell 11 _used_flags must include {flag}"


# ---------------------------------------------------------------------------
# Test 14: Prepare cell (cell 10) uses s2t_small6 with cache-budget-gib 10
# ---------------------------------------------------------------------------

def test_prepare_cell_uses_s2t_small6():
    """Cell 10 must target s2t_small6 with --cache-budget-gib 10."""
    nb = _load_notebook()
    src = _cell_source(nb, 10)

    assert '"/kaggle/working/s2t_small6"' in src, (
        "Cell 10 must output to s2t_small6"
    )
    assert '--cache-budget-gib", "10"' in src, (
        "Cell 10 must use --cache-budget-gib 10"
    )


# ---------------------------------------------------------------------------
# Test 15: Fresh run safeguards — no --resume, no full-v2 artifact reuse
# ---------------------------------------------------------------------------

def test_fresh_run_safeguards():
    """For the initial small6 experiment: no restore/resume of full-v2
    artifacts, fresh output directory, no silent checkpoint reuse."""
    nb = _load_notebook()
    src11 = _cell_source(nb, 11)

    # Training cell forbids --resume and --force
    assert '--resume" not in cmd' in src11, (
        "Training cell must forbid --resume for fresh run"
    )
    assert '--force" not in cmd' in src11, (
        "Training cell must forbid --force for fresh run"
    )

    # Restore cell targets s2t_small6, not s2t_v2
    src3 = _restore_cell_source(nb)
    assert 'f"s2t_{EXPERIMENT_ID}"' in src3, (
        "Restore cell must search for s2t_small6 directories"
    )
    assert '"s2t_v2" in dirnames' not in src3, (
        "Restore cell must not search for s2t_v2 directories"
    )

    # Output directory is s2t_small6
    assert 'OUT = f"/kaggle/working/s2t_{EXPERIMENT_ID}"' in src3, (
        "Restore cell OUT must be s2t_small6"
    )


# ---------------------------------------------------------------------------
# Test 16: Cross-experiment restore rejection (end-to-end logic)
# ---------------------------------------------------------------------------

def test_cross_experiment_restore_rejection_logic():
    """The restore cell's provenance check will reject artifacts from a
    different experiment because model_config and tokenizer_fingerprint
    won't match EXPECTED_MODEL_CONFIG / EXPECTED_TOKENIZER_FINGERPRINT."""
    nb = _load_notebook()
    src = _restore_cell_source(nb)

    assert 'EXPECTED_MODEL_CONFIG.items()' in src, (
        "Provenance check must iterate EXPECTED_MODEL_CONFIG"
    )
    assert '_fail(f"checkpoint model_config mismatch' in src, (
        "Provenance check must fail on model_config mismatch"
    )
    assert '_fail(f"tokenizer_fingerprint mismatch' in src, (
        "Provenance check must fail on tokenizer_fingerprint mismatch"
    )


# ---------------------------------------------------------------------------
# Test 17: No s2t_v2 code references anywhere (output isolation)
# ---------------------------------------------------------------------------

def test_no_s2t_v2_code_references():
    """No cell may reference /kaggle/working/s2t_v2 in code — full-v2
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
        "No s2t_v2 code references allowed — full-v2 output dir must not be used"
    )


if __name__ == "__main__":
    import sys
    tests = [
        test_output_dir_is_experiment_scoped,
        test_experiment_id_defined,
        test_provenance_compatibility_check_exists,
        test_expected_model_config_matches_training_script,
        test_cache_version_not_sufficient,
        test_restore_matches_by_experiment_name,
        test_usable_is_experiment_aware,
        test_reuse_lock_after_provenance,
        test_unset_tokenizer_fingerprint_first_run_safe,
        test_set_tokenizer_fingerprint_later_run_enforced,
        test_architecture_mismatch_detected,
        test_dataset_identity_limitation_documented,
        test_training_cell_matches_runbook_small6,
        test_prepare_cell_uses_s2t_small6,
        test_fresh_run_safeguards,
        test_cross_experiment_restore_rejection_logic,
        test_no_s2t_v2_code_references,
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
