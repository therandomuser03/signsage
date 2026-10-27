"""REST schemas for static ISL image classification (alphabet / word)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ImageClassifyRequest(BaseModel):
    """A single static image to classify.

    `image` is a base64-encoded JPEG/PNG, optionally as a data URI
    (data:image/jpeg;base64,...). `task` selects the classifier:
    "alphabet" (ISL fingerspelled letters A-Z) or "word" (static ISL word signs).
    """
    image: str = Field(..., description="base64-encoded image (optionally a data URI)")
    task: str = Field("alphabet", description="'alphabet' or 'word'")
    min_confidence: Optional[float] = Field(
        0.5, ge=0.0, le=1.0,
        description="minimum model confidence to accept a prediction",
    )

    @field_validator("task")
    @classmethod
    def _validate_task(cls, v: str) -> str:
        if v not in ("alphabet", "word"):
            raise ValueError("task must be 'alphabet' or 'word'")
        return v


class ImageClassifyResponse(BaseModel):
    label: str = Field(..., description="predicted class label (empty if not accepted)")
    accepted: bool
    reason: str = Field(..., description="'ok' or 'low_model_confidence'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    task: str
    index: int
    n_classes: int
    model_version: str
    input_size: list[int] = Field(..., description="[height, width] of preprocessed input")
