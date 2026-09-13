from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings, get_settings
from app.schemas.api import (
    CapabilitiesResponse,
    CapabilityFeatures,
    CapabilityLimits,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PoseNormalizationStatsInfo,
    ProvenanceInfo,
    WebSocketMessageTypes,
)
from app.services.model_manager import ModelManager
from app.services.pose_preprocessing import POSE_NORMALIZATION_STATS


router = APIRouter(prefix="/api/v1")


# NOTE ON SCOPE: this router is diagnostics-only (health/models/capabilities).
# There is deliberately no REST endpoint for Sign->Text or Text->Pose
# translation itself -- docs/API.md is explicit that the WebSocket is the
# sole transport for "the core translation loop," specifically to avoid
# per-frame HTTP overhead. Backend-only testability without a frontend
# (tests/, scripts/verify_backend.py) comes from calling the service layer
# directly, not from an HTTP mirror of the WebSocket contract. See
# DECISIONS.md for this call.


def get_manager(request: Request) -> ModelManager:
    manager = getattr(request.app.state, "model_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    return manager


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    manager = getattr(request.app.state, "model_manager", None)
    return HealthResponse(
        status="ok" if manager is not None else "degraded",
        service="signsage-backend",
        device=str(manager.device) if manager else "unknown",
        models_loaded=manager is not None,
    )


@router.get("/models", response_model=ModelsResponse)
def models(manager: ModelManager = Depends(get_manager)):
    provenance = manager.provenance_summary()
    return ModelsResponse(
        models=[ModelInfo(**item) for item in manager.model_inventory()],
        provenance=ProvenanceInfo(**provenance) if provenance else None,
        pose_normalization=PoseNormalizationStatsInfo(
            **POSE_NORMALIZATION_STATS.snapshot()
        ),
    )


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities(
    settings: Settings = Depends(get_settings),
    manager: ModelManager = Depends(get_manager),
):
    """Static contract/configuration info an integrator needs before opening
    a WebSocket session -- message types, size/length limits, and which
    optional features (no-face model, Groq/Ollama gloss) are actually
    turned on in this deployment. See docs/API.md."""
    return CapabilitiesResponse(
        websocket_endpoint="/api/v1/ws/translate",
        message_types=WebSocketMessageTypes(
            client=["landmarks.frame", "sentence.boundary", "text.input"],
            server=["translation.text", "translation.pose", "error"],
        ),
        limits=CapabilityLimits(
            max_frames=settings.max_frames,
            max_tokens=settings.max_tokens,
            pose_output_frames=settings.pose_output_frames,
            pose_fps=25,
            max_websocket_frames=settings.max_websocket_frames,
            max_ws_message_bytes=settings.max_ws_message_bytes,
            max_text_chars=settings.max_text_chars,
        ),
        features=CapabilityFeatures(
            noface_ablation_available=manager.sign_to_text_noface is not None,
            groq_gloss_enabled=settings.enable_groq_gloss,
            ollama_gloss_enabled=settings.enable_ollama_gloss,
            landmark_input_mode=settings.landmark_input_mode,
        ),
    )
