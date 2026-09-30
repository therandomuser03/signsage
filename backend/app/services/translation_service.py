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
from app.ml.adapters.sign_to_text_adapter import SignToTextAdapter


class TranslationService:
    def __init__(self, pipeline: SignToTextAdapter, settings: Settings):
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
        return TranslationResult(
            direction=Direction.SIGN_TO_TEXT,
            text=result.text,
            accepted=result.accepted,
            reason=result.reason,
            confidence=result.confidence,
            motion_energy=result.motion_energy,
            created_at=datetime.now(timezone.utc),
        )
