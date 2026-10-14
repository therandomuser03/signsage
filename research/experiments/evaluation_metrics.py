"""Repaired evaluation metrics for SignSage sign-to-text models.

Standardizes metric scales:
- BLEU: percentage in [0.0, 100.0] via sacrebleu with effective order & smoothing
- chrF: percentage in [0.0, 100.0] via sacrebleu (character n-gram F-score)
- TER:  percentage in [0.0, 100.0+] via sacrebleu (Translation Edit Rate, lower is better)
- WER:  percentage in [0.0, 100.0+] via jiwer (Word Error Rate, lower is better)
- ROUGE-L: ratio in [0.0, 1.0] via rouge_score
- exact_match: ratio in [0.0, 1.0]
- nonempty_output: ratio in [0.0, 1.0]

Ensures:
- Identical text normalization for both hypotheses and references
- Identical evaluation pipeline for model predictions and trivial baseline
- Meaningful trivial baseline (no artificial zero BLEU from short sentences)
- Principled diversity measurement (detects mode collapse without weak thresholds)
- Defensible aggregate evaluation gate (replaces single exact match check)
- Strict validation: empty references, length mismatches, and NaNs fail safely
"""
from __future__ import annotations

import math
from collections import Counter
import numpy as np


METRIC_SCALES = {
    "WER": "percentage (0-100+)",
    "BLEU": "percentage (0-100)",
    "BLEU-1": "percentage (0-100)",
    "BLEU-2": "percentage (0-100)",
    "BLEU-3": "percentage (0-100)",
    "BLEU-4": "percentage (0-100)",
    "chrF": "percentage (0-100)",
    "TER": "percentage (0-100+)",
    "ROUGE-L": "ratio (0.0-1.0)",
    "exact_match": "ratio (0.0-1.0)",
    "nonempty_output": "ratio (0.0-1.0)",
    "diversity.unique_fraction": "ratio (0.0-1.0)",
}


def normalize_text(text: object) -> str:
    """Canonical text normalization: lowercase and collapsed whitespace."""
    if text is None:
        return ""
    return " ".join(str(text).lower().strip().split())


def validate_eval_pairs(refs: list[str], hyps: list[str]) -> tuple[list[str], list[str]]:
    """Validate and normalize evaluation pairs; fails safely on bad inputs."""
    if refs is None or hyps is None:
        raise ValueError("References and hypotheses must not be None.")
    if len(refs) != len(hyps):
        raise ValueError(
            f"Evaluation length mismatch: {len(refs)} references vs {len(hyps)} hypotheses."
        )
    if len(refs) == 0:
        raise ValueError("Empty evaluation dataset: references cannot be empty.")

    r_norm: list[str] = []
    for i, r in enumerate(refs):
        if r is None or not str(r).strip():
            raise ValueError(f"Empty or invalid reference at index {i}: {r!r}")
        r_norm.append(normalize_text(r))

    h_norm: list[str] = [normalize_text(h) for h in hyps]
    return r_norm, h_norm


def per_class_recall(r_norm: list[str], h_norm: list[str]) -> dict[str, float]:
    """Computes per-class recall over normalized references."""
    support: dict[str, list[int]] = {}
    for r, h in zip(r_norm, h_norm):
        cell = support.setdefault(r, [0, 0])
        cell[1] += 1
        cell[0] += int(r == h)
    return {k: float(v[0] / v[1]) for k, v in sorted(support.items())}


def compute_metrics(refs: list[str], hyps: list[str]) -> dict:
    """Compute comprehensive translation and recognition metrics.

    All error and translation metrics (WER, BLEU, chrF, TER) are on the 0-100 scale.
    Overlap and classification metrics (exact_match, nonempty_output, ROUGE-L)
    are ratios in [0.0, 1.0].
    """
    import jiwer
    import sacrebleu
    from rouge_score import rouge_scorer

    r_norm, h_norm = validate_eval_pairs(refs, hyps)

    # SacreBLEU with effective order to prevent artificial zeros on short sentences
    # ponytail: default 4-gram BLEU zeros out on 1-3 word sentences; use_effective_order adapts
    bleu_score = sacrebleu.corpus_bleu(
        h_norm, [r_norm], smooth_method="exp", use_effective_order=True
    )
    chrf_score = sacrebleu.corpus_chrf(h_norm, [r_norm])
    ter_score = sacrebleu.corpus_ter(h_norm, [r_norm])

    # Word Error Rate via jiwer, scaled to percentage [0-100+]
    raw_wer = float(jiwer.wer(r_norm, h_norm))
    wer_pct = raw_wer * 100.0

    # ROUGE-L via rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = float(
        np.mean([scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(r_norm, h_norm)])
    )

    em = float(np.mean([r == h for r, h in zip(r_norm, h_norm)]))
    nonempty = float(np.mean([bool(h) for h in h_norm]))

    gen_lens = [len(h.split()) for h in h_norm]
    ref_lens = [len(r.split()) for r in r_norm]

    metrics = {
        "WER": float(wer_pct),
        "BLEU": float(bleu_score.score),
        "BLEU-1": float(bleu_score.precisions[0]) if len(bleu_score.precisions) > 0 else 0.0,
        "BLEU-2": float(bleu_score.precisions[1]) if len(bleu_score.precisions) > 1 else 0.0,
        "BLEU-3": float(bleu_score.precisions[2]) if len(bleu_score.precisions) > 2 else 0.0,
        "BLEU-4": float(bleu_score.precisions[3]) if len(bleu_score.precisions) > 3 else 0.0,
        "chrF": float(chrf_score.score),
        "TER": float(ter_score.score),
        "ROUGE-L": rouge_l,
        "exact_match": em,
        "nonempty_output": nonempty,
        "per_class_recall": per_class_recall(r_norm, h_norm),
        "gen_len_mean": float(np.mean(gen_lens)),
        "ref_len_mean": float(np.mean(ref_lens)),
        "metric_scales": METRIC_SCALES,
    }

    # Verify no NaN or Inf in numeric outputs
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if not math.isfinite(v):
                raise ValueError(f"Non-finite metric computed for {k}: {v}")

    return metrics


def diversity_metrics(hyps: list[str], refs: list[str] | None = None) -> dict:
    """Principled diversity measurement rejecting collapsed output.

    Measures:
    - unique sentence fraction
    - mode concentration (top_prediction_freq)
    - distinct unigram (distinct_1) and bigram (distinct_2) ratios
    - reference-relative diversity comparison when references are provided
    """
    if hyps is None or len(hyps) == 0:
        return {
            "unique": 0,
            "unique_fraction": 0.0,
            "top_prediction": "",
            "top_prediction_freq": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "collapsed": True,
        }

    h_norm = [normalize_text(h) for h in hyps]
    n = len(h_norm)
    counts = Counter(h_norm)
    top_pred, top_count = counts.most_common(1)[0]
    unique_count = len(counts)
    unique_fraction = unique_count / n
    top_freq = top_count / n

    # Distinct-1 and Distinct-2 n-gram ratios
    unigrams = [w for h in h_norm for w in h.split()]
    distinct_1 = len(set(unigrams)) / max(1, len(unigrams))

    bigrams = [
        (words[i], words[i + 1])
        for h in h_norm
        for words in [h.split()]
        for i in range(len(words) - 1)
    ]
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams))

    # Reference mode and unique fraction for contextual grounding
    ref_unique_count = None
    ref_unique_frac = None
    ref_top_freq = 0.0
    if refs is not None and len(refs) > 0:
        r_norm = [normalize_text(r) for r in refs if r is not None and str(r).strip()]
        if r_norm:
            r_counts = Counter(r_norm)
            ref_unique_count = len(r_counts)
            ref_unique_frac = ref_unique_count / len(r_norm)
            ref_top_freq = r_counts.most_common(1)[0][1] / len(r_norm)

    # Principled collapse verdict:
    # 1. Extreme mode concentration: top prediction covers >50% of outputs
    #    (unless ground truth references themselves are that concentrated)
    # 2. Complete collapse: fewer than 2 unique predictions when reference has >= 2
    # 3. Severe under-diversity: <5% unique sentences when reference is diverse (>=20%)
    is_collapsed = False
    if n >= 4:
        if top_freq > max(0.50, ref_top_freq + 0.10):
            is_collapsed = True
        elif ref_unique_count is not None and ref_unique_count >= 2 and unique_count < 2:
            is_collapsed = True
        elif unique_fraction < 0.05 and (ref_unique_frac is None or ref_unique_frac >= 0.20):
            is_collapsed = True
    elif unique_count <= 1 and n > 1 and (ref_unique_count is None or ref_unique_count > 1):
        is_collapsed = True

    res = {
        "unique": unique_count,
        "unique_fraction": unique_fraction,
        "top_prediction": top_pred,
        "top_prediction_freq": top_freq,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "collapsed": bool(is_collapsed),
    }
    if ref_unique_frac is not None:
        res["ref_unique_fraction"] = ref_unique_frac
        res["diversity_ratio_to_ref"] = unique_fraction / max(1e-6, ref_unique_frac)

    return res


def build_trivial_baseline(refs: list[str], train_texts: list[str] | None = None) -> list[str]:
    """Constructs a meaningful trivial majority baseline over the same reference count."""
    pool = None
    if train_texts is not None and len(train_texts) > 0:
        cleaned_train = [normalize_text(t) for t in train_texts if t is not None and str(t).strip()]
        if cleaned_train:
            pool = cleaned_train

    if not pool:
        cleaned_refs = [normalize_text(r) for r in refs if r is not None and str(r).strip()]
        if not cleaned_refs:
            raise ValueError("Cannot build trivial baseline: no valid references or training texts.")
        pool = cleaned_refs

    majority = Counter(pool).most_common(1)[0][0]
    return [majority] * len(refs)


def evaluate_gate(test_metrics: dict, trivial_metrics: dict, diversity: dict) -> dict:
    """Defensible aggregate evaluation criteria replacing single exact match.

    A model passes ONLY if:
    1. Error rate is strictly lower than trivial baseline (WER < trivial_WER)
    2. Overlap is strictly higher than trivial baseline (BLEU > trivial_BLEU)
    3. Character fidelity is strictly higher than trivial baseline (chrF > trivial_chrF)
    4. Exact match accuracy is at least as high as the trivial majority baseline
    5. Output diversity is not collapsed
    6. Non-empty output rate is >= 90%
    """
    gate = {
        "beats_trivial_wer": bool(test_metrics["WER"] < trivial_metrics["WER"]),
        "beats_trivial_bleu": bool(test_metrics["BLEU"] > trivial_metrics["BLEU"]),
        "beats_trivial_chrf": bool(test_metrics["chrF"] > trivial_metrics["chrF"]),
        "beats_trivial_exact_match": bool(
            test_metrics["exact_match"] >= trivial_metrics["exact_match"]
        ),
        "diversity_not_collapsed": bool(not diversity["collapsed"]),
        "nonempty_adequate": bool(test_metrics["nonempty_output"] >= 0.90),
    }
    gate["GO"] = bool(all(gate.values()))
    return gate

