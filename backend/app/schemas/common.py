"""Shared response envelopes."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    code: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_promoted: bool
    model_bundle_found: bool  # a bundle directory exists at SIGNSAGE_MODEL_BUNDLE_DIR
    model_loaded: bool  # the checkpoint is loaded in this process
    word_recognition_enabled: Optional[bool] = None  # word-level recognition behind a feature flag


class CapabilitiesResponse(BaseModel):
    """What this deployment can do right now -- the frontend reads this once
    and hides/labels features accordingly (never guess client-side)."""
    api_version: str
    websocket_path: str
    auth_enabled: bool
    auth_required: bool
    history_enabled: bool
    sign_to_text_available: bool
    sign_to_text_unavailable_reason: Optional[str]  # "model_not_promoted" | "model_bundle_missing"
    sign_to_text_source: str = "unavailable"  # "transformer" | "isolated-word" | "heuristic-fallback" | "unavailable"
    fallback_vocabulary: List[str] = []  # gloss IDs the heuristic fallback recognizes
    text_to_sign_mode: str  # "avatar_gloss"
    avatar_vocabulary_size: int
    landmark_count: int
    max_frames_per_sequence: int
    # Word-level recognition (behind SIGNSAGE_WORD_MODE_ENABLED)
    word_recognition_enabled: Optional[bool] = None
    word_recognition_unavailable_reason: Optional[str] = None
    word_recognition_vocabulary: Optional[List[str]] = None
    word_recognition_threshold: Optional[float] = None
    word_recognition_model_version: Optional[str] = None
    word_recognition_label: str = "sign-sequence to sentence interpretation (8 words plus a rest (no-sign) class), adapted to one signer"


class ModelInfo(BaseModel):
    id: str
    direction: str
    promoted: bool
    loaded: bool
    evaluation_status: Optional[str] = None  # e.g. "NO-GO", read from the bundle's evaluation_results.json
    metrics: Dict[str, float] = {}


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
