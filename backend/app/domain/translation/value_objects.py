"""Small value objects for the translation domain."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceProxy:
    """Wraps the model's confidence score with an explicit reminder that it
    is uncalibrated -- see ml/postprocessing/confidence.py."""
    value: float

    def __post_init__(self):
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"ConfidenceProxy must be in [0, 1], got {self.value}")
