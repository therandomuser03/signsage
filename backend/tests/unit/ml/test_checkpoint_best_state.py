"""Regression tests for checkpoint best-loss ordering.

Covers update_best_state() in research/experiments/train_sign_to_text.py:
- First finite validation loss replaces initial inf.
- Improving vs non-improving epochs (incl. 1e-4 tolerance boundary).
- Resume after an improvement and after a non-improving epoch.
- Non-finite (nan/inf) validation loss never becomes best.
- Patience counting and early-stopping signal.
"""
from __future__ import annotations
import math
import sys

sys.path.insert(0, "/mnt/D/Codes/signsage/backend")
sys.path.insert(0, "/mnt/D/Codes/signsage/research/experiments")

from train_sign_to_text import update_best_state


def test_first_finite_replaces_inf():
    best, bad, improved, stop = update_best_state(float("inf"), 0, 13.5, patience=2)
    assert best == 13.5 and bad == 0 and improved is True and stop is False


def test_improving_resets_patience():
    best, bad, improved, stop = update_best_state(5.0, 1, 4.0, patience=2)
    assert best == 4.0 and bad == 0 and improved is True and stop is False


def test_tolerance_boundary_not_improved():
    # 5.0 - 4.99999 = 1e-5 < tol 1e-4 -> not an improvement
    best, bad, improved, stop = update_best_state(5.0, 0, 4.99999, patience=3)
    assert best == 5.0 and bad == 1 and improved is False and stop is False


def test_non_improving_preserves_best():
    best, bad, improved, stop = update_best_state(4.0, 0, 4.5, patience=3)
    assert best == 4.0 and bad == 1 and improved is False and stop is False


def test_patience_triggers_stop():
    best, bad, improved, stop = update_best_state(4.0, 1, 4.5, patience=2)
    assert best == 4.0 and bad == 2 and improved is False and stop is True


def test_resume_after_improvement():
    b, d, _, _ = update_best_state(float("inf"), 0, 5.0, patience=3)
    b, d, _, _ = update_best_state(b, d, 4.0, patience=3)
    assert b == 4.0 and d == 0


def test_resume_after_non_improving():
    b, d, _, _ = update_best_state(float("inf"), 0, 5.0, patience=3)
    b, d, _, _ = update_best_state(b, d, 6.0, patience=3)
    assert (b, d) == (5.0, 1)
    b2, d2, improved, _ = update_best_state(b, d, 4.5, patience=3)
    assert (b2, d2, improved) == (4.5, 0, True)


def test_non_finite_never_best():
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        best, bad, improved, _ = update_best_state(4.0, 0, bad_val, patience=3)
        assert best == 4.0 and bad == 1 and improved is False
    # first epoch nan: best stays inf, patience increments
    best, bad, improved, _ = update_best_state(float("inf"), 0, float("nan"), patience=3)
    assert math.isinf(best) and bad == 1 and improved is False
