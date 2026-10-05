"""REST Sign->Text schemas (api/routes/translation.py)."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.ml.preprocessing.landmarks import COORDINATES_PER_LANDMARK, TOTAL_LANDMARKS

import math


def validate_landmark_rows(rows: List[List[float]]) -> List[List[float]]:
    """Every landmark must be exactly [x, y, z] of finite numbers; ragged or
    non-finite input would otherwise crash numpy conversion deep inside the
    pipeline (or poison the model with NaN)."""
    for row in rows:
        if len(row) != COORDINATES_PER_LANDMARK:
            raise ValueError(f"each landmark must have {COORDINATES_PER_LANDMARK} coordinates")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("landmark coordinates must be finite numbers")
    return rows


class LandmarkFramePayload(BaseModel):
    landmarks: List[List[float]] = Field(..., min_length=TOTAL_LANDMARKS, max_length=TOTAL_LANDMARKS)

    _check = field_validator("landmarks")(validate_landmark_rows)


class SignToTextRequest(BaseModel):
    """A complete recorded sequence (REST fallback / testing). Live capture
    should use the WebSocket route."""
    # Upper bound mirrors TranslationSession's buffer ceiling (4 * 160): an
    # unbounded REST sequence would hit the same memory/DoS concern, so it is
    # rejected at the schema boundary with a clear error.
    frames: List[LandmarkFramePayload] = Field(..., min_length=2, max_length=640)
    min_motion: Optional[float] = None
    min_confidence: Optional[float] = None
    save_history: bool = True  # only takes effect for signed-in users


class SignToTextResponse(BaseModel):
    text: str
    accepted: bool
    reason: str
    confidence: float
    motion_energy: float
    result_type: str  # "model" | "failed"
    history_id: Optional[str] = None
