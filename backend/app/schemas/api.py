from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings


# ---------------- REST / diagnostic schemas ----------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    device: str
    models_loaded: bool


class ModelInfo(BaseModel):
    name: str
    task: str
    input_features: int | None = None
    output_frames: int | None = None
    checkpoint: str | None = None
    status: str
    caveat: str | None = None


class ProvenanceInfo(BaseModel):
    """Which research run produced the served checkpoints, and its
    documented go/no-go status -- sourced from the knowledge-repository
    manifest (see DATA_MODEL.md). `None` fields mean the manifest was
    missing or didn't contain that value; this is diagnostic metadata, not
    something callers should rely on being present."""

    run_id: str | None = None
    phase_0_status: str | None = None
    warning: str | None = None


class PoseNormalizationStatsInfo(BaseModel):
    """Cumulative, process-lifetime counters from `normalize_pose_sequence`
    (see app/services/pose_preprocessing.py). This mirrors the notebook's own
    `POSE_NORMALIZATION_STATS` counter -- tracked there but, per
    research/outputs/phase0/RESULTS.md, 'never printed or logged... a known
    instrumentation gap.' The backend replicates the same counter; this
    field is what actually surfaces it.

    Only increments in `raw` landmark-input mode, since that is the only
    mode in which this backend runs `normalize_pose_sequence` itself (in
    `normalized` mode the client is contractually responsible for
    normalization already -- see docs/API.md). If `SIGNSAGE_LANDMARK_INPUT_MODE`
    stays at its default, these will simply stay at zero, which is correct,
    not a bug.
    """

    total_sequences: int
    bad_center_sequences: int
    degenerate_scale_sequences: int


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    provenance: ProvenanceInfo | None = None
    pose_normalization: PoseNormalizationStatsInfo


class CapabilityLimits(BaseModel):
    max_frames: int
    max_tokens: int
    pose_output_frames: int
    pose_fps: int
    max_websocket_frames: int
    max_ws_message_bytes: int
    max_text_chars: int


class CapabilityFeatures(BaseModel):
    noface_ablation_available: bool
    groq_gloss_enabled: bool
    ollama_gloss_enabled: bool
    landmark_input_mode: Literal["normalized", "raw"]
    demo_fallback_available: bool = False


class WebSocketMessageTypes(BaseModel):
    client: list[str]
    server: list[str]


class CapabilitiesResponse(BaseModel):
    """Static, non-secret configuration a client/integrator needs before
    opening a WebSocket session -- see docs/API.md. Deliberately separate
    from `ModelsResponse`: this is about what the *contract* supports
    (message types, limits, feature flags), not which checkpoints are
    loaded or their research provenance."""

    websocket_endpoint: str
    message_types: WebSocketMessageTypes
    limits: CapabilityLimits
    features: CapabilityFeatures


# ---------------- WebSocket schemas ----------------

class LandmarksFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["landmarks.frame"]
    session_id: Annotated[str, Field(min_length=1, max_length=128)]
    timestamp_ms: int
    landmarks: Annotated[list[float], Field(min_length=1629, max_length=1629)]

    @field_validator("landmarks")
    @classmethod
    def finite_landmarks(cls, value: list[float]) -> list[float]:
        out: list[float] = []
        for v in value:
            fv = float(v)
            if not math.isfinite(fv):
                raise ValueError("landmarks must contain finite numeric values.")
            out.append(fv)
        return out


class SentenceBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["sentence.boundary"]
    session_id: Annotated[str, Field(min_length=1, max_length=128)]
    trigger: Literal["motion_pause", "manual"]


class TextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text.input"]
    session_id: Annotated[str, Field(min_length=1, max_length=128)]
    # Max length is configurable (SIGNSAGE_MAX_TEXT_CHARS); enforced below.
    text: Annotated[str, Field(min_length=1)]

    @field_validator("text")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank.")
        max_chars = get_settings().max_text_chars
        if len(value) > max_chars:
            raise ValueError(f"text may contain at most {max_chars} characters.")
        return value


ClientMessage = Union[LandmarksFrame, SentenceBoundary, TextInput]


class TranslationText(BaseModel):
    type: Literal["translation.text"] = "translation.text"
    session_id: str
    text: str
    latency_ms: float
    confidence: float = 0.5
    is_fallback: bool = False
    gated_reason: str | None = None


class TranslationPose(BaseModel):
    type: Literal["translation.pose"] = "translation.pose"
    session_id: str
    gloss: str
    frames: list[list[float]]
    fps: int
    latency_ms: float
    is_fallback: bool = False


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    session_id: str
    message: str
    code: str
