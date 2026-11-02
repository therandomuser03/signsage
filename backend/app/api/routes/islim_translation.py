"""Continuous ISL-to-English translation endpoint via verified ISLiM adapter.

Input: JSON keypoint sequence (B, T, 94, 4) or video upload (processed to
MediaPipe landmarks then converted to ISLiM's (T, 94, 4) contract).
Output: English sentence prediction + metadata (status, checkpoint, device,
diversity, collapsed, promoted=false).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.errors import ModelNotPromotedError, ModelUnavailableError
from app.ml.adapters.islim_adapter import ISLiMAdapter, ISLiMAdapterError
from app.ml.preprocessing.islim_landmarks import (
    MAX_FRAMES as ISLI_MAX_FRAMES,
    convert_landmarks_543_to_islim,
)
from app.services.video_processing.video_pipeline import process_video_file

router = APIRouter(prefix="/translation", tags=["translation"])

# Load once per process (lazy init on first request to avoid startup failure
# if the model is missing).
_adapter: Optional[ISLiMAdapter] = None


def _get_adapter() -> ISLiMAdapter:
    global _adapter
    if _adapter is None:
        settings = get_settings()
        ckpt = settings.model_bundle_dir / "checkpoints" / "ISLiM_4Loss_best_for_10kSpamoOF.pt" if hasattr(settings, "model_bundle_dir") else None
        if ckpt is None or not ckpt.exists():
            ckpt = ISLiMAdapter.DEFAULT_CKPT
        _adapter = ISLiMAdapter(checkpoint_path=str(ckpt), device="cpu")
    return _adapter


def _serialize_result(result: dict) -> dict:
    result = dict(result)
    result.pop("checkpoint", None)
    result["model_version"] = "islim_mobile_export_v1"
    result["promoted"] = False  # SIGNSAGE_MODEL_PROMOTED remains false
    return {"status": "ok", "translation": result.get("predictions", []),
            "metadata": {k: v for k, v in result.items() if k != "predictions"}}


@router.post("/islim/translate")
async def translate_islim_keypoints(payload: dict = None):
    """Accept preprocessed keypoint JSON (shape contract verified by adapter)."""
    try:
        adapter = _get_adapter()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "unavailable",
                                                      "reason": str(exc),
                                                      "predictions": []})
    if payload is None or not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"status": "invalid",
                                                      "reason": "Expected JSON payload with keypoints array."})
    keypoints = payload.get("keypoints")
    if keypoints is None:
        return JSONResponse(status_code=400, content={"status": "invalid",
                                                      "reason": "Missing 'keypoints'."})
    try:
        arr = np.array(keypoints, dtype=np.float32)
        result = adapter.evaluate_sample(arr, reference_text=payload.get("reference"))
    except ISLiMAdapterError as exc:
        return JSONResponse(status_code=400, content={"status": "invalid",
                                                      "reason": str(exc), "predictions": []})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error",
                                                      "reason": str(exc), "predictions": []})
    return _serialize_result(result)


@router.post("/islim/translate-video")
async def translate_islim_video(file: UploadFile = File(...)):
    """Video upload -> MediaPipe landmarks -> ISLiM (T,94,4) -> English sentence.

    Real pipeline: decode video -> extract pose+hand landmarks (T,543,3) ->
    convert to ISLiM's (T,94,4) [x,y,z,visibility] contract -> ISLiM adapter
    -> continuous English translation.
    """
    settings = get_settings()

    if not file.filename:
        return JSONResponse(status_code=422, content={"detail": "No file uploaded.",
                                                      "code": "INVALID_INPUT"})

    allowed_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_ext:
        return JSONResponse(status_code=422, content={
            "detail": f"Unsupported file type: {file_ext}. Supported: {', '.join(sorted(allowed_ext))}",
            "code": "INVALID_FILE_TYPE",
        })

    content = await file.read()
    max_size = 50 * 1024 * 1024  # 50 MB
    if len(content) > max_size:
        return JSONResponse(status_code=413, content={
            "detail": f"File exceeds maximum size ({max_size} bytes).",
            "code": "FILE_TOO_LARGE",
        })

    try:
        adapter = _get_adapter()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc),
                                                      "code": "MODEL_UNAVAILABLE"})

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Real video -> (T, 543, 3) MediaPipe landmarks.
        landmarks_543 = process_video_file(tmp_path, max_frames=ISLI_MAX_FRAMES)

        # Convert to ISLiM contract (T, 94, 4) [x, y, z, visibility].
        landmarks_94 = convert_landmarks_543_to_islim(landmarks_543,
                                                      max_frames=ISLI_MAX_FRAMES)

        # ISLiM continuous translation.
        result = adapter.evaluate_sample(landmarks_94[None, ...],
                                         reference_text=None)
        if result.get("status") != "ok":
            return JSONResponse(status_code=500, content={
                "detail": result.get("message", "ISLiM inference failed"),
                "code": "INFERENCE_ERROR",
            })

        payload = _serialize_result(result)
        payload["metadata"]["frames_processed"] = int(landmarks_543.shape[0])
        payload["metadata"]["landmarks_shape"] = [int(v) for v in landmarks_543.shape]
        payload["metadata"]["input_contract"] = "(T, 94, 4) [x, y, z, visibility]"
        return payload
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc),
                                                      "code": "VIDEO_PROCESSING_ERROR"})
    except ISLiMAdapterError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc),
                                                      "code": "INVALID_LANDMARKS"})
    except ModelNotPromotedError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc),
                                                      "code": "MODEL_NOT_PROMOTED"})
    except ModelUnavailableError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc),
                                                      "code": "MODEL_UNAVAILABLE"})
    except Exception as exc:  # noqa: BLE001 - surface a clean 500, never fabricate
        return JSONResponse(status_code=500, content={"detail": "Internal processing error.",
                                                      "code": "INTERNAL_ERROR"})
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
