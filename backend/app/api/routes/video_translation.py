"""REST endpoint for video upload -> isolated-word prediction."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.errors import ModelNotPromotedError, ModelUnavailableError, InvalidLandmarksError
from app.ml.models.registry import get_sign_to_text_pipeline

router = APIRouter(prefix="/translation", tags=["translation"])


@router.post("/video")
async def translate_video_upload(file: UploadFile = File(...)):
    """Accept an MP4/video file and return an isolated-word prediction.

    Preserves the existing REST contract: structured response with
    text, accepted, reason, confidence, and motion_energy.
    """
    settings = get_settings()

    # File validation
    if not file.filename:
        return JSONResponse(
            status_code=422,
            content={"detail": "No file uploaded.", "code": "INVALID_INPUT"},
        )

    allowed_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_ext:
        return JSONResponse(
            status_code=422,
            content={
                "detail": f"Unsupported file type: {file_ext}. Supported: {', '.join(allowed_ext)}",
                "code": "INVALID_FILE_TYPE",
            },
        )

    # Size guard (useful error; doesn't enforce a strict byte limit here)
    content = await file.read()
    max_size = 50 * 1024 * 1024  # 50 MB
    if len(content) > max_size:
        return JSONResponse(
            status_code=413,
            content={"detail": f"File exceeds maximum size ({max_size} bytes).", "code": "FILE_TOO_LARGE"},
        )

    # Write to temporary file (not retained by default)
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        adapter = get_sign_to_text_pipeline(settings, allow_unpromoted=settings.allow_unpromoted_dev)
        from app.services.video_processing.video_service import predict_from_video, VideoProcessingError
        result = predict_from_video(
            tmp_path,
            adapter=adapter,
            max_frames=64,
            min_motion=settings.min_motion_energy,
            min_confidence=settings.min_confidence,
        )
        return {
            "text": result["text"],
            "accepted": result["accepted"],
            "reason": result["reason"],
            "confidence": result["confidence"],
            "motion_energy": result["motion_energy"],
            "result_type": "model" if result["accepted"] else "failed",
            "frames_processed": result["frames_processed"],
            "model_version": result["model_version"],
            "feature_version": result["feature_version"],
        }
    except VideoProcessingError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "code": "VIDEO_PROCESSING_ERROR"},
        )
    except ModelNotPromotedError as exc:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "code": "MODEL_NOT_PROMOTED"},
        )
    except ModelUnavailableError as exc:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "code": "MODEL_UNAVAILABLE"},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal processing error.", "code": "INTERNAL_ERROR"},
        )
    finally:
        # Clean up temporary file safely
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
