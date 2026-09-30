"""Domain-level re-export of the ML landmark contract.

Keeps domain/ and services/ from importing app.ml directly for the one
piece of information they actually need (how many landmarks/coordinates the
pipeline expects) -- see ml/contracts/landmark_contract.py for the real,
bundle-driven implementation this wraps.
"""
from __future__ import annotations

from app.ml.preprocessing.landmarks import COORDINATES_PER_LANDMARK, TOTAL_LANDMARKS

EXPECTED_LANDMARK_COUNT = TOTAL_LANDMARKS
EXPECTED_COORDINATES = COORDINATES_PER_LANDMARK
