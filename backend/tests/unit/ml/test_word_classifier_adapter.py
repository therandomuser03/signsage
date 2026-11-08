"""Unit tests for app.ml.word_classifier_adapter -- pure numpy + torch, no network needed."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.word_classifier_adapter import WordClassifierAdapter, WordClassificationResult

BUNDLE_DIR = Path(__file__).resolve().parents[4] / "models" / "word_v3"


def _golden_input() -> np.ndarray:
    return np.load(BUNDLE_DIR / "golden_test_vector.npy").astype(np.float32)


def _golden_logits() -> list[float]:
    return json.loads((BUNDLE_DIR / "golden_expected_logits.json").read_text())


@pytest.mark.skipif(not BUNDLE_DIR.is_dir(), reason="models/word_v3/ bundle not present")
def test_adapter_loads_word_v3_bundle():
    adapter = WordClassifierAdapter.load(BUNDLE_DIR, device="cpu")
    assert adapter.labels == ["hello", "me", "you", "yes", "no", "help", "water", "where", "rest"]
    assert not adapter.model.training
    assert adapter.promotion_eligible() is False
    assert adapter.confidence_threshold == 0.7
    assert adapter.model_version == "word_v3_8w_s123"


@pytest.mark.skipif(not BUNDLE_DIR.is_dir(), reason="models/word_v3/ bundle not present")
def test_golden_vector_reproduces_expected_logits():
    adapter = WordClassifierAdapter.load(BUNDLE_DIR, device="cpu")
    arr = _golden_input()
    result = adapter.classify(arr)
    assert result.status == "ok"
    assert result.label is not None
    # The golden vector is labelled "hello" (top logit index 0 = 7.43).
    assert result.label == "hello"
    # Reconstruct logits from top3 + confidence is lossy; instead verify via the
    # raw model forward pass against golden_expected_logits.json.
    import torch
    from app.ml.preprocessing.sign_features_v3 import extract_features as v3

    features = v3(arr, layout="signsage")
    with torch.inference_mode():
        logits = adapter.model(torch.from_numpy(features[None]).float())[0]
    expected = torch.tensor(_golden_logits())
    assert torch.allclose(logits, expected, atol=1e-4), f"maxdiff={(logits - expected).abs().max().item():.2e}"


@pytest.mark.skipif(not BUNDLE_DIR.is_dir(), reason="models/word_v3/ bundle not present")
def test_adapter_rejects_invalid_input():
    adapter = WordClassifierAdapter.load(BUNDLE_DIR, device="cpu")
    # Wrong shape
    with pytest.raises(ValueError, match="shaped"):
        adapter.classify(np.zeros((4, 10, 3), dtype=np.float32))
    # Too few frames
    with pytest.raises(ValueError, match="at least 2"):
        adapter.classify(np.zeros((1, 543, 3), dtype=np.float32))
    # NaN
    bad = np.zeros((4, 543, 3), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapter.classify(bad)


@pytest.mark.skipif(not BUNDLE_DIR.is_dir(), reason="models/word_v3/ bundle not present")
def test_no_hands_clip_returns_no_hands_status():
    adapter = WordClassifierAdapter.load(BUNDLE_DIR, device="cpu")
    # All-zero hands; only pose/face present
    arr = np.zeros((10, 543, 3), dtype=np.float32)
    arr[:, 23, 0] = 0.4  # left hip
    arr[:, 24, 0] = 0.6  # right hip
    result = adapter.classify(arr)
    assert result.status == "no_hands"
    assert result.label is None
    assert result.confidence == 0.0
    assert result.top3 == []
    assert result.frames_with_hand == 0


@pytest.mark.skipif(not BUNDLE_DIR.is_dir(), reason="models/word_v3/ bundle not present")
def test_classify_returns_top3_and_valid_confidence():
    adapter = WordClassifierAdapter.load(BUNDLE_DIR, device="cpu")
    arr = _golden_input()
    result = adapter.classify(arr)
    assert result.status == "ok"
    assert result.label in adapter.labels
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top3) == 3
    for entry in result.top3:
        assert "label" in entry and "probability" in entry
        assert entry["label"] in adapter.labels
        assert 0.0 <= entry["probability"] <= 1.0


def test_adapter_reports_not_promotable():
    adapter = WordClassifierAdapter.load(BUNDLE_DIR, device="cpu")
    assert adapter.promotion_eligible() is False
