"""Service that connects video file input to isolated-word predictions."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from app.ml.isolated_words import IsolatedWordAdapter
from app.services.video_processing.video_pipeline import process_video_file


class VideoProcessingError(Exception):
    pass


def predict_from_video(
    video_path: str | Path,
    adapter: IsolatedWordAdapter,
    max_frames: int = 64,
    min_motion: float = 0.002,
    min_confidence: float = 0.0,
) -> dict:
    """Run the full pipeline: MP4 -> landmarks -> adapter -> result dict."""
    try:
        landmarks = process_video_file(video_path, max_frames=max_frames)
    except Exception as exc:
        raise VideoProcessingError(f"Video processing failed: {exc}") from exc

    result = adapter.transcribe(landmarks, min_motion=min_motion, min_confidence=min_confidence)
    return {
        "text": result.text,
        "accepted": result.accepted,
        "reason": result.reason,
        "confidence": result.confidence,
        "motion_energy": result.motion_energy,
        "frames_processed": int(landmarks.shape[0]),
        "landmarks_shape": list(landmarks.shape),
        "model_version": adapter.metadata.get("format", "isolated_words_v1"),
        "feature_version": adapter.metadata.get("feature_version"),
    }
