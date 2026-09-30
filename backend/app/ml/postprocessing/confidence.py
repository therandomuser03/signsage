"""Confidence scoring and acceptance gating for a decoded translation.

The confidence "proxy" is the geometric mean of per-token probabilities
from greedy decoding -- NOT a calibrated probability (the notebook's own
evaluation cell warns of this explicitly: "confidence is a diagnostic, not
a calibrated probability"). Treat it as a relative signal for gating
obviously-degenerate decodes, not as "% chance this is correct".
"""
from __future__ import annotations

import math


def compute_confidence(logps: list) -> float:
    if not logps:
        return 0.0
    return float(math.exp(sum(logps) / len(logps)))


def passes_acceptance_gate(text: str, motion_energy: float, confidence: float,
                            min_motion: float, min_confidence: float):
    """Returns (accepted, reason). Mirrors the notebook's
    predict_live_landmarks() acceptance logic exactly: motion gate,
    confidence gate, non-empty text.
    """
    if motion_energy < min_motion:
        return False, "insufficient_motion"
    if confidence < min_confidence:
        return False, "low_model_confidence"
    if not text.strip():
        return False, "empty_output"
    return True, "ok"
