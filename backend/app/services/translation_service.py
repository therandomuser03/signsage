"""Sign->Text translation use-case.

The single entry point api/websocket/translation.py (and
api/routes/translation.py, for the REST fallback) should call. Wraps the ML
adapter (app.ml.adapters.sign_to_text_adapter) with the domain-level result
type, so callers get a TranslationResult, not raw model output.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np

from app.core.config import Settings
from app.domain.translation.entities import Direction, TranslationResult
from app.ml.adapters.sign_to_text_adapter import SignToTextAdapter, TranscriptionResult
from app.services.gloss_composer import compose_words


class TranslationService:
    def __init__(self, pipeline, settings: Settings):
        self._pipeline = pipeline
        self._settings = settings

    def translate_sign_to_text(
        self,
        landmarks_xyz: np.ndarray,
        min_motion: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> TranslationResult:
        result = self._pipeline.transcribe(
            landmarks_xyz,
            min_motion=min_motion if min_motion is not None else self._settings.min_motion_energy,
            min_confidence=min_confidence if min_confidence is not None else self._settings.min_confidence,
        )
        text = result.text
        # The word-continuous pipeline emits an ordered sequence of glosses,
        # not a sentence. Compose one (deterministic rules -> LLM -> fallback)
        # so callers get sentence-level output, not "word word word".
        if getattr(self._pipeline, "word_continuous", False):
            adapter = getattr(self._pipeline, "adapter", None)
            vocab = set(getattr(adapter, "labels", []))
            if vocab:
                sentence, method = compose_words(text.split(), vocab, self._settings)
                if sentence:
                    import logging
                    logging.getLogger(__name__).info(
                        "[COMPOSE] composed sentence (method=%s): %r", method, sentence)
                    result = TranscriptionResult(
                        text=sentence,
                        accepted=result.accepted,
                        reason=result.reason,
                        motion_energy=result.motion_energy,
                        confidence=result.confidence,
                    )
                    text = sentence
        created_at = datetime.now(timezone.utc)
        return TranslationResult(
            direction=Direction.SIGN_TO_TEXT,
            text=text,
            accepted=result.accepted,
            reason=result.reason,
            confidence=result.confidence,
            motion_energy=result.motion_energy,
            created_at=created_at,
        )
