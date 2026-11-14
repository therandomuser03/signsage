"""TranslationService composes gloss sequences into sentences for word-continuous pipelines."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import get_settings
from app.services.translation_service import TranslationService


class _FakePipeline:
    """Minimal word-continuous pipeline: transcribe() returns a fixed gloss text."""

    word_continuous = True

    def __init__(self, text: str):
        self.adapter = SimpleNamespace(labels=["hello", "water", "where", "you", "rest"])
        self._text = text

    def transcribe(self, landmarks_xyz, min_motion, min_confidence):
        frames = len(landmarks_xyz)
        return SimpleNamespace(
            text=self._text,
            accepted=bool(self._text.split()),
            reason="ok" if self._text.split() else "no_signs_detected",
            motion_energy=0.01,
            confidence=0.95 if self._text.split() else 0.0,
            feature_ms=0.0,
            inference_ms=0.0,
        )


def _settings():
    s = get_settings()
    s.groq_api_key = None  # deterministic: rule match / comma-join only
    return s


def test_word_continuous_translates_glosses_to_sentence():
    service = TranslationService(_FakePipeline("hello water"), _settings())
    landmarks = np.zeros((30, 543, 3), dtype=np.float32)
    result = service.translate_sign_to_text(landmarks)
    assert result.text == "Hello, water."
    assert result.accepted is True
    assert result.confidence == 0.95


def test_word_continuous_rule_pattern_where_you():
    service = TranslationService(_FakePipeline("where you"), _settings())
    landmarks = np.zeros((30, 543, 3), dtype=np.float32)
    result = service.translate_sign_to_text(landmarks)
    assert result.text == "Where are you?"
