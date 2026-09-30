"""REST Sign->Text schemas (api/routes/translation.py)."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.ml.preprocessing.landmarks import COORDINATES_PER_LANDMARK, TOTAL_LANDMARKS


def validate_landmark_rows(rows: List[List[float]]) -> List[List[float]]:
    """Every landmark must be exactly [x, y, z]; ragged input would otherwise
    crash numpy conversion deep inside the pipeline."""
    for row in rows:
        if len(row) != COORDINATES_PER_LANDMARK:
            raise ValueError(f"each landmark must have {COORDINATES_PER_LANDMARK} coordinates")
    return rows


class LandmarkFramePayload(BaseModel):
    landmarks: List[List[float]] = Field(..., min_length=TOTAL_LANDMARKS, max_length=TOTAL_LANDMARKS)

    _check = field_validator("landmarks")(validate_landmark_rows)


class SignToTextRequest(BaseModel):
    """A complete recorded sequence (REST fallback / testing). Live capture
    should use the WebSocket route."""
    frames: List[LandmarkFramePayload] = Field(..., min_length=2)
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
