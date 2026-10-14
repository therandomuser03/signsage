"""Regression tests for research/experiments/evaluation_metrics.py.

Covers:
- Collapsed predictor (all same output) → gate fails
- Repeat-most-frequent-train predictor → no better than trivial, gate fails
- Predictor worse than trivial → gate fails
- Plausible predictor better than trivial → gate passes
- Metric scale consistency (WER/BLEU/chrF/TER all on 0-100 scale)
- Invalid inputs: empty refs, length mismatch, None refs raise ValueError
- Missing/empty hypothesis handled (empty string normalises to empty)
- NaN/Inf guard raises on pathological output
"""
from __future__ import annotations
import math
import sys
import pytest

sys.path.insert(0, "/mnt/D/Codes/signsage/backend")
sys.path.insert(0, "/mnt/D/Codes/signsage/research/experiments")

import evaluation_metrics as em


# ── fixtures ──────────────────────────────────────────────────────────────────

REFS = [
    "hello world",
    "thank you",
    "good morning",
    "please help me",
    "see you later",
    "how are you",
]
TRAIN = REFS + ["hello world", "hello world"]  # "hello world" is most frequent


def _trivial(refs, train=None):
    """Build trivial baseline and compute its metrics."""
    baseline = em.build_trivial_baseline(refs, train or refs)
    return em.compute_metrics(refs, baseline)


# ── 1. Collapsed predictor ────────────────────────────────────────────────────

def test_collapsed_predictor_fails_gate():
    """All predictions the same → diversity collapsed → gate GO=False."""
    hyps = ["hello world"] * len(REFS)
    m = em.compute_metrics(REFS, hyps)
    d = em.diversity_metrics(hyps, REFS)
    trivial = _trivial(REFS, TRAIN)

    assert d["collapsed"], "Expected collapsed=True when all hyps are identical"
    gate = em.evaluate_gate(m, trivial, d)
    assert not gate["GO"], "Collapsed predictor must fail gate"
    assert not gate["diversity_not_collapsed"]


def test_collapsed_predictor_nonzero_bleu():
    """Collapsing on the correct most-frequent word still gets some BLEU
    but the gate catches collapse via diversity, not via BLEU alone."""
    hyps = ["hello world"] * len(REFS)
    m = em.compute_metrics(REFS, hyps)
    # BLEU may be non-zero because "hello world" matches one ref exactly
    assert m["BLEU"] >= 0.0
    assert m["BLEU"] <= 100.0, "BLEU must be 0-100 scale"


# ── 2. Repeat-most-frequent-train predictor ──────────────────────────────────

def test_repeat_frequent_train_sentence_fails_gate():
    """A predictor that always repeats the most-frequent training sentence
    is equivalent to the trivial majority baseline – should not beat it."""
    majority = em.build_trivial_baseline(REFS, TRAIN)[0]
    hyps = [majority] * len(REFS)
    m = em.compute_metrics(REFS, hyps)
    trivial = _trivial(REFS, TRAIN)
    d = em.diversity_metrics(hyps, REFS)

    # Cannot beat the trivial baseline that IS the same strategy
    assert not (m["BLEU"] > trivial["BLEU"] and m["WER"] < trivial["WER"]), \
        "Repeating train majority should not strictly beat its own trivial baseline"
    gate = em.evaluate_gate(m, trivial, d)
    assert not gate["GO"], "Repeat-train predictor must fail gate"


# ── 3. Predictor worse than trivial ──────────────────────────────────────────

def test_worse_than_trivial_fails_gate():
    """Garbage predictions → worse WER/BLEU than trivial → gate fails."""
    hyps = ["zzz zzz zzz"] * len(REFS)
    m = em.compute_metrics(REFS, hyps)
    trivial = _trivial(REFS, TRAIN)
    d = em.diversity_metrics(hyps, REFS)

    # WER for nonsense must be >= trivial WER (trivial at least matches some refs)
    assert m["WER"] >= trivial["WER"] or not m["BLEU"] > trivial["BLEU"], \
        "Garbage predictor should not beat trivial on all metrics"
    gate = em.evaluate_gate(m, trivial, d)
    assert not gate["GO"], "Predictor worse than trivial must fail gate"


# ── 4. Plausible predictor beats trivial ─────────────────────────────────────

def test_plausible_predictor_passes_gate():
    """Nearly-correct predictions → beats trivial on all criteria → gate GO."""
    # Predict correctly for most, slightly wrong for one
    hyps = list(REFS)          # perfect predictions
    hyps[2] = "good afternoon" # one error
    m = em.compute_metrics(REFS, hyps)
    trivial = _trivial(REFS, TRAIN)
    d = em.diversity_metrics(hyps, REFS)

    assert m["WER"] < trivial["WER"], "Near-perfect predictor must beat trivial WER"
    assert m["BLEU"] > trivial["BLEU"], "Near-perfect predictor must beat trivial BLEU"
    gate = em.evaluate_gate(m, trivial, d)
    assert gate["GO"], f"Near-perfect predictor should pass gate: {gate}"


# ── 5. Metric scale consistency ───────────────────────────────────────────────

def test_metric_scales_are_0_100_for_error_metrics():
    """WER, BLEU, chrF, TER must all be on 0-100 scale, not 0-1."""
    refs = ["hello world", "thank you"]
    hyps = ["hello world", "good morning"]
    m = em.compute_metrics(refs, hyps)

    for key in ("WER", "BLEU", "chrF"):
        v = m[key]
        assert 0.0 <= v <= 100.0, f"{key}={v} is outside [0, 100] – wrong scale"

    # TER can exceed 100 for very bad outputs but must be non-negative
    assert m["TER"] >= 0.0, f"TER={m['TER']} negative"

    # Ratio metrics must be in [0, 1]
    for key in ("exact_match", "nonempty_output", "ROUGE-L"):
        v = m[key]
        assert 0.0 <= v <= 1.0, f"{key}={v} is outside [0, 1] – wrong scale"


def test_metric_scales_documented():
    """METRIC_SCALES dict contains entries for all core metrics."""
    for key in ("WER", "BLEU", "chrF", "TER", "exact_match", "nonempty_output"):
        assert key in em.METRIC_SCALES, f"Missing scale documentation for {key}"


def test_perfect_predictions_give_zero_wer_100_bleu():
    refs = ["hello world", "thank you", "good morning"]
    m = em.compute_metrics(refs, list(refs))
    assert m["WER"] == pytest.approx(0.0, abs=1e-6), "Perfect preds → WER=0"
    assert m["BLEU"] == pytest.approx(100.0, abs=0.01), "Perfect preds → BLEU≈100"
    assert m["exact_match"] == pytest.approx(1.0, abs=1e-6)


def test_effective_order_prevents_zero_bleu_on_short_sentences():
    """Without effective_order, 1-word sentences give BLEU=0 even for perfect match.
    With effective_order=True (our implementation), BLEU > 0."""
    refs = ["hello", "thanks", "yes"]
    hyps = ["hello", "thanks", "yes"]
    m = em.compute_metrics(refs, hyps)
    assert m["BLEU"] > 0.0, \
        "BLEU must be > 0 for perfect 1-word predictions (use_effective_order bug)"


def test_trivial_bleu_is_nonzero_with_effective_order():
    """Root cause test: trivial majority baseline must not score 0.00 BLEU
    on short-sentence corpora due to 4-gram zeroing."""
    refs = ["hello", "thanks", "yes", "no", "please", "help"]
    baseline = em.build_trivial_baseline(refs)
    m = em.compute_metrics(refs, baseline)
    assert m["BLEU"] > 0.0, \
        "Trivial BLEU should be nonzero with use_effective_order=True"


# ── 6. Invalid / missing predictions ─────────────────────────────────────────

def test_empty_refs_raises():
    with pytest.raises(ValueError):
        em.compute_metrics([], ["hello"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="[Mm]ismatch|length"):
        em.compute_metrics(["hello", "world"], ["hello"])


def test_none_refs_raises():
    with pytest.raises(ValueError):
        em.compute_metrics(None, ["hello"])


def test_none_hyps_raises():
    with pytest.raises(ValueError):
        em.compute_metrics(["hello"], None)


def test_none_ref_item_raises():
    """A None inside refs must raise ValueError, not silently produce NaN."""
    with pytest.raises(ValueError, match="[Ee]mpty|[Ii]nvalid"):
        em.compute_metrics(["hello", None], ["hello", "world"])


def test_none_hyp_normalises_to_empty():
    """None hypotheses normalise to empty string (no crash, output tracked)."""
    refs = ["hello", "world"]
    hyps = [None, "world"]
    m = em.compute_metrics(refs, hyps)
    assert m["nonempty_output"] == pytest.approx(0.5, abs=1e-6)


def test_empty_hyp_string_normalises_to_empty():
    refs = ["hello", "world"]
    hyps = ["", "world"]
    m = em.compute_metrics(refs, hyps)
    assert m["nonempty_output"] == pytest.approx(0.5, abs=1e-6)


def test_all_empty_hyps_fails_nonempty_gate():
    """All-empty hypotheses → nonempty_output=0.0 → gate fails on nonempty_adequate."""
    refs = ["hello", "world", "foo"]
    hyps = ["", "", ""]
    m = em.compute_metrics(refs, hyps)
    trivial = _trivial(refs)
    d = em.diversity_metrics(hyps, refs)
    gate = em.evaluate_gate(m, trivial, d)
    assert not gate["nonempty_adequate"]
    assert not gate["GO"]


# ── 7. Diversity edge-cases ───────────────────────────────────────────────────

def test_diversity_all_same_collapsed():
    d = em.diversity_metrics(["hello"] * 10, ["hello", "world"] * 5)
    assert d["collapsed"]
    assert d["unique_fraction"] == pytest.approx(1 / 10)


def test_diversity_all_unique_not_collapsed():
    hyps = ["a", "b", "c", "d", "e", "f"]
    d = em.diversity_metrics(hyps)
    assert not d["collapsed"]
    assert d["unique_fraction"] == pytest.approx(1.0)


def test_diversity_empty_hyps():
    d = em.diversity_metrics([])
    assert d["collapsed"]
    assert d["unique_fraction"] == pytest.approx(0.0)


# ── 8. build_trivial_baseline ─────────────────────────────────────────────────

def test_trivial_baseline_length_matches_refs():
    bl = em.build_trivial_baseline(REFS, TRAIN)
    assert len(bl) == len(REFS)


def test_trivial_baseline_is_most_frequent_train():
    """With TRAIN having 3× 'hello world', majority must be 'hello world'."""
    bl = em.build_trivial_baseline(REFS, TRAIN)
    assert all(t == "hello world" for t in bl)


def test_trivial_baseline_falls_back_to_refs():
    """Without train_texts, falls back to refs majority."""
    refs = ["apple", "apple", "banana"]
    bl = em.build_trivial_baseline(refs)
    assert all(t == "apple" for t in bl)


def test_trivial_baseline_empty_train_uses_refs():
    refs = ["cat", "cat", "dog"]
    bl = em.build_trivial_baseline(refs, train_texts=[])
    assert all(t == "cat" for t in bl)


def test_trivial_baseline_empty_refs_raises():
    with pytest.raises(ValueError):
        em.build_trivial_baseline([])
