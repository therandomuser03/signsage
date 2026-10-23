"""Focused regression tests for the tokenizer fingerprint guard."""
import json
import sys
from pathlib import Path

NOTEBOOK = Path("research/notebooks/signsage_kaggle.ipynb")


def _restore_cell_source(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "EXPECTED_TOKENIZER_FINGERPRINT" in src and "_fail" in src:
            return src
    raise AssertionError("restore cell not found")


def load_nb():
    with NOTEBOOK.open() as f:
        return json.load(f)


def test_missing_expected_fingerprint_fails_for_restored():
    """When EXPECTED_TOKENIZER_FINGERPRINT='' and restored=True, restore must fail."""
    nb = load_nb()
    src = _restore_cell_source(nb)
    # The restore cell must contain the fail-closed guard
    assert 'restored and not EXPECTED_TOKENIZER_FINGERPRINT' in src, (
        "Restored checkpoint with missing fingerprint must be rejected")
    assert 'tokenizer fingerprint check SKIPPED' in src or 'tokenizer fingerprint check skipped' in src, (
        "Message format must reference skipped fingerprint check")


def test_fresh_training_does_not_fail_on_missing_fingerprint():
    """Fresh training does not fail merely because no prior checkpoint fingerprint exists."""
    nb = load_nb()
    src = _restore_cell_source(nb)
    # Fresh run skips fingerprint check; it must not raise
    assert 'fresh run' in src.lower() or 'FRESH RUN' in src, (
        "Fresh run must be handled explicitly")
    # The fingerprint mismatch check is guarded by non-empty expected value
    assert 'if EXPECTED_TOKENIZER_FINGERPRINT and _ckpt_fp != EXPECTED_TOKENIZER_FINGERPRINT:' in src, (
        "Mismatch check must be guarded by non-empty expected fingerprint")


def test_set_fingerprint_is_enforced():
    """When EXPECTED_TOKENIZER_FINGERPRINT is set, a mismatch must fail."""
    nb = load_nb()
    src = _restore_cell_source(nb)
    assert '_fail(f"tokenizer_fingerprint mismatch' in src, (
        "Mismatched fingerprint with set expected value must fail")


def test_message_contains_skip_and_set_instruction():
    """The skip message must instruct the operator to set the fingerprint after first run."""
    nb = load_nb()
    src = _restore_cell_source(nb)
    assert 'set EXPECTED_TOKENIZER_FINGERPRINT after first run' in src, (
        "Must instruct operator to set fingerprint after first run")


if __name__ == "__main__":
    tests = [
        test_missing_expected_fingerprint_fails_for_restored,
        test_fresh_training_does_not_fail_on_missing_fingerprint,
        test_set_fingerprint_is_enforced,
        test_message_contains_skip_and_set_instruction,
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
