"""Continuous ISL-to-English translation endpoint via verified ISLiM adapter.

Input: JSON keypoint sequence (B, T, 94, 4) or video upload (processed to landmarks).
Output: English sentence prediction + metadata (status, checkpoint, device, diversity).
"""
from __future__ import annotations

import tempfile, os, json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.errors import ModelNotPromotedError, ModelUnavailableError
from app.ml.adapters.islim_adapter import ISLiMAdapter, ISLiMAdapterError

router = APIRouter(prefix="/translation", tags=["translation"])

# Load once per process (lazy init on first request to avoid startup failure if model missing)
_adapter: Optional[ISLiMAdapter] = None

def _get_adapter() -> ISLiMAdapter:
    global _adapter
    if _adapter is None:
        settings = get_settings()
        ckpt = settings.model_bundle_dir / "checkpoints" / "ISLiM_4Loss_best_for_10kSpamoOF.pt" if hasattr(settings, "model_bundle_dir") else None
        # Fallback to verified local checkpoint path
        if ckpt is None or not ckpt.exists():
            ckpt = ISLiMAdapter.DEFAULT_CKPT
        _adapter = ISLiMAdapter(checkpoint_path=str(ckpt), device="cpu")
    return _adapter

@router.post("/islim/translate")
async def translate_islim_keypoints(payload: dict = None):
    """Accept preprocessed keypoint JSON (shape contract verified by adapter)."""
    try:
        adapter = _get_adapter()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status":"unavailable","reason":str(exc),"predictions":[]})
    if payload is None or not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"status":"invalid","reason":"Expected JSON payload with keypoints array."})
    keypoints = payload.get("keypoints")
    if keypoints is None:
        return JSONResponse(status_code=400, content={"status":"invalid","reason":"Missing 'keypoints'."})
    import numpy as np, torch
    try:
        arr = np.array(keypoints, dtype=np.float32)
        # Adapter validates shape; pass through
        result = adapter.evaluate_sample(arr, reference_text=payload.get("reference"))
    except ISLiMAdapterError as exc:
        return JSONResponse(status_code=400, content={"status":"invalid","reason":str(exc),"predictions":[]})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status":"error","reason":str(exc),"predictions":[]})
    # Never expose HF_TOKEN or internal checkpoint paths to client
    result.pop("checkpoint", None)
    result["model_version"] = "islim_mobile_export_v1"
    result["promoted"] = False  # SIGNSAGE_MODEL_PROMOTED remains false per rules
    return {"status":"ok","translation": result.get("predictions",[]),"metadata":{k:v for k,v in result.items() if k!="predictions"}}

@router.post("/islim/translate-video")
async def translate_islim_video(file: UploadFile = File(...)):
    """Video upload -> ISLiM continuous translation (placeholder integration)."""
    # Real pipeline requires MediaPipe landmark extraction to (T,94,4);
    # this endpoint reports the workflow and requires pre-processed keypoints
    # for now because landmark extraction is separate (video_service / predict_landmarks).
    return JSONResponse(status_code=202, content={
        "status":"accepted",
        "note":"Video-to-landmark extraction must complete first (use /translation/video for isolated-word pipeline, or pre-process to JSON keypoints for /translation/islim/translate).",
        "pipeline":"MediaPipe holistic -> select 94 landmarks + visibility -> ISLiM adapter",
        "promoted":False,
    })
