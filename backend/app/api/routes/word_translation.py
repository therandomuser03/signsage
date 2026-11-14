"""Word/sentence recognition REST endpoints (behind SIGNSAGE_WORD_MODE_ENABLED).

POST /api/v1/translation/word
    input  = landmarks: list of frames, each 543 x [x, y, z]
    output = { word, confidence, accepted, top3, threshold, model_version, status }
    accepted=false and word=null when max prob < threshold or the predicted
    label is "rest". Returns 503 WORD_MODE_DISABLED when the flag is off.

POST /api/v1/translation/sentence
    input  = { frames: [...] } (same frame format as /word)
    output = { gloss_sequence, sentence, method, accepted, model_version, ... }
    Continuous sign stream (motion-gated sliding windows) -> ordered glosses
    -> sentence (deterministic rules -> Groq refinement -> fallback).

POST /api/v1/translation/compose
    input  = { words: [...] }
    output = { sentence, method }
    Validates each word against the loaded vocabulary. Uses Groq (when
    configured) with a strict prompt, falls back to a deterministic join.
    Handles history saving for signed-in users.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.core.dependencies import get_optional_history_service, get_request_user
from app.core.errors import SignSageError
from app.core.ratelimit import rate_limit_dependency
from app.core.security import AuthenticatedUser
from app.domain.translation.entities import Direction, NewHistoryRecord, ResultType
from app.ml.word_classifier_adapter import WordClassifierAdapter, WordClassificationResult, resolve_word_model_dir
from app.ml.word_sequence import WordSequencePipeline
from app.services.gloss_composer import compose_words
from app.services.history_service import HistoryService

logger = logging.getLogger("signsage.word_debug")
DEBUG_CAPTURES_DIR = Path(__file__).resolve().parents[3] / "debug_captures"
DEBUG_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
# Dev-only labeled captures (SIGNSAGE_CAPTURE_MODE_ENABLED, loopback only).
CAPTURES_LABELED_DIR = Path(__file__).resolve().parents[3] / "captures_labeled"
CAPTURES_LABELED_DIR.mkdir(parents=True, exist_ok=True)


def _is_loopback_address(host: str | None) -> bool:
    """True only for loopback clients (127.0.0.1 / ::1). Never Render."""
    return host in ("127.0.0.1", "::1")


def _save_labeled_capture(
    label: str,
    landmarks: np.ndarray,
    result: WordClassificationResult,
    model_version: str,
) -> None:
    """Save a labeled capture to backend/captures_labeled/<label>/<ts>.npz.

    Never raises: a failed save must never break the recognition response.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label_dir = CAPTURES_LABELED_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    top3 = [(e["label"], float(e["probability"])) for e in result.top3]
    np.savez(
        label_dir / f"{ts}.npz",
        landmarks=landmarks,
        true_label=np.array([label]),
        top3_labels=np.array([t[0] for t in top3]),
        top3_probs=np.array([t[1] for t in top3]),
        model_version=np.array([model_version]),
        timestamp=np.array([ts]),
    )


router = APIRouter(prefix="/translation", tags=["translation"])

# Process-wide singleton for the word classifier bundle.
_word_adapter: Optional[WordClassifierAdapter] = None
_word_adapter_loaded = False


class WordModeDisabledError(SignSageError):
    code = "WORD_MODE_DISABLED"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


def _warmup_golden_inference(adapter: WordClassifierAdapter, model_dir: str) -> None:
    """Run one dummy inference on the golden test vector so the model is warm.

    Silently skipped if the vector file is absent (e.g. a custom
    SIGNSAGE_WORD_MODEL_DIR without the golden vector next to it).
    """
    golden = Path(model_dir) / "golden_test_vector.npy"
    if not golden.is_file():
        logger.warning("WORD_MODE: golden test vector not found: %s", golden)
        return
    arr = np.load(golden).astype(np.float32)
    adapter.classify(arr)


def load_word_adapter_at_startup(settings: Settings) -> None:
    """Load the word classifier model at startup (avoids first-request latency).

    Engaged whenever SIGNSAGE_WORD_MODE_ENABLED=true -- independent of
    SIGNSAGE_WORD_DEBUG. Runs one dummy inference with the golden test vector
    (models/word_v3/golden_test_vector.npy) so the first real request is
    already warm. A failure is logged, not fatal: the first request still
    triggers a lazy load via _get_word_adapter().
    """
    global _word_adapter, _word_adapter_loaded
    if not settings.word_mode_enabled:
        return
    if _word_adapter_loaded:
        return
    model_dir = resolve_word_model_dir(settings)
    t0 = time.perf_counter()
    try:
        _word_adapter = WordClassifierAdapter.load(model_dir, device=settings.device)
        _warmup_golden_inference(_word_adapter, model_dir)
        _word_adapter_loaded = True
        warm_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "WORD_MODE: word model warmed at startup in %.1f ms (dir=%s; golden "
            "inference included) -- first real request should not pay the load cost",
            warm_ms,
            model_dir,
        )
    except Exception as exc:  # noqa: BLE001 -- never fail startup because of word mode
        logger.error("WORD_MODE: word model warm-up failed (dir=%s): %s", model_dir, exc)


@router.on_event("startup")
async def _startup_load_word_model() -> None:
    load_word_adapter_at_startup(get_settings())


def _resolve_word_model_dir(settings: Settings) -> str:
    """Alias kept for call sites/tests that import this name."""
    return resolve_word_model_dir(settings)


def _get_word_adapter(settings: Settings) -> WordClassifierAdapter:
    global _word_adapter, _word_adapter_loaded
    if not _word_adapter_loaded:
        model_dir = _resolve_word_model_dir(settings)
        _word_adapter = WordClassifierAdapter.load(model_dir, device=settings.device)
        _word_adapter_loaded = True
    return _word_adapter


def _reset_word_adapter() -> None:
    """Test-only: force the next _get_word_adapter() call to reload."""
    global _word_adapter, _word_adapter_loaded
    _word_adapter = None
    _word_adapter_loaded = False


def _extract_landmarks(payload: dict) -> np.ndarray:
    """Validate payload["frames"] and return a (T, 543, 3) float32 array."""
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_MESSAGE", "message": "frames must be a list of at least 2 landmark frames"},
        )

    try:
        landmarks = np.array(
            [f["landmarks"] for f in frames], dtype=np.float32
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_MESSAGE", "message": f"invalid landmarks: {exc}"},
        ) from exc

    if landmarks.ndim != 3 or landmarks.shape[1:] != (543, 3):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_MESSAGE",
                "message": f"each frame must be 543 x [x, y, z]; received shape {landmarks.shape}",
            },
        )
    if not np.isfinite(landmarks).all():
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_MESSAGE", "message": "landmarks must contain only finite coordinates"},
        )
    return landmarks


@router.post("/word", dependencies=[Depends(rate_limit_dependency("word"))])
async def word_recognition(
    request: Request,
    payload: dict,
    settings: Settings = Depends(get_settings),
    user: Optional[AuthenticatedUser] = Depends(get_request_user),
) -> dict:
    if not settings.word_mode_enabled:
        raise WordModeDisabledError(
            "Word recognition is disabled. Set SIGNSAGE_WORD_MODE_ENABLED=true "
            "to enable the word-level recognition path."
        )

    t0_total = time.perf_counter()
    t_parse_start = time.perf_counter()
    landmarks = _extract_landmarks(payload)
    t_parse_ms = (time.perf_counter() - t_parse_start) * 1000

    n_frames = int(landmarks.shape[0])
    payload_size_bytes = len(json.dumps(payload))
    left_present = float(np.any(landmarks[:, 501:522, :] != 0.0, axis=(1, 2)).mean())
    right_present = float(np.any(landmarks[:, 522:543, :] != 0.0, axis=(1, 2)).mean())

    adapter = _get_word_adapter(settings)

    # --- Dev-only labeled capture (SIGNSAGE_CAPTURE_MODE_ENABLED) ----------
    # Honored only from loopback (127.0.0.1 / ::1): enabling it on Render
    # saves nothing and logs a warning instead.
    save_capture = False
    true_label = payload.get("true_label")
    loopback = _is_loopback_address(request.client.host if request.client else None)
    if settings.capture_mode_enabled and isinstance(true_label, str) and true_label:
        if not loopback:
            logger.warning(
                "CAPTURE_MODE: ignoring true_label from non-loopback client %s; "
                "labeled captures are saved from loopback only",
                request.client.host if request.client else None,
            )
        elif true_label not in adapter.labels:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "INVALID_MESSAGE",
                    "message": "true_label must be one of the vocabulary labels: "
                    f"{sorted(adapter.labels)}",
                },
            )
        else:
            save_capture = True

    result = adapter.classify(landmarks)
    t_total_ms = (time.perf_counter() - t0_total) * 1000

    if save_capture:
        try:
            _save_labeled_capture(true_label, landmarks, result, adapter.model_version)
        except Exception as exc:  # noqa: BLE001 - never break the response for a capture save
            logger.warning("CAPTURE_MODE: failed to save capture: %s", exc)

    if settings.word_debug:
        logger.info(
            "WORD_DEBUG: parse=%.1fms feature=%.1fms inference=%.1fms total=%.1fms "
            "frames=%d payload_bytes=%d left_present=%.3f right_present=%.3f "
            "status=%s label=%s conf=%.3f",
            t_parse_ms,
            result.feature_ms,
            result.inference_ms,
            t_total_ms,
            n_frames,
            payload_size_bytes,
            left_present,
            right_present,
            result.status,
            result.label,
            result.confidence,
        )
        # Save debug capture (landmarks + top-3 result).
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            npz_path = DEBUG_CAPTURES_DIR / f"{ts}.npz"
            top3 = [(e["label"], float(e["probability"])) for e in result.top3]
            np.savez(
                npz_path,
                landmarks=landmarks,
                label=np.array([result.label or ""]),
                confidence=np.array([float(result.confidence)]),
                status=np.array([result.status]),
                top3_labels=np.array([t[0] for t in top3]),
                top3_probs=np.array([t[1] for t in top3]),
            )
        except Exception as exc:  # noqa: BLE001 - never break the response for debug capture
            logger.warning("WORD_DEBUG: failed to save capture: %s", exc)

    if result.status == "no_hands":
        return {
            "word": None,
            "confidence": 0.0,
            "accepted": False,
            "top3": [],
            "threshold": adapter.confidence_threshold,
            "model_version": adapter.model_version,
            "status": "no_hands",
        }

    if result.status != "ok":
        raise HTTPException(
            status_code=500,
            detail={"code": "INFERENCE_ERROR", "message": "word classification failed"},
        )

    accepted = bool(result.label is not None and result.label != "rest" and result.confidence >= adapter.confidence_threshold)
    return {
        "word": result.label if accepted else None,
        "confidence": result.confidence,
        "accepted": accepted,
        "top3": result.top3,
        "threshold": adapter.confidence_threshold,
        "model_version": adapter.model_version,
        "status": "ok",
    }


@router.post("/sentence", dependencies=[Depends(rate_limit_dependency("word"))])
async def sentence_recognition(
    request: Request,
    payload: dict,
    settings: Settings = Depends(get_settings),
    user: Optional[AuthenticatedUser] = Depends(get_request_user),
    history: Optional[HistoryService] = Depends(get_optional_history_service),
) -> dict:
    """Continuous sign stream -> ordered glosses -> sentence.

    Same frame format as /word ({frames: [...]}), same rate limiter
    ("word" scope). The stream is fed through the continuous pipeline
    (motion-gated sliding windows over the isolated word model), which
    finalises signs at low-motion boundaries. The ordered glosses are then
    composed into an English sentence (deterministic rules first, Groq
    strictly-guarded refinement, comma-join fallback).
    Returns { gloss_sequence, sentence, method, accepted, model_version, ... }.
    Rejects captures of more than 400 frames with 422.
    """
    if not settings.word_mode_enabled:
        raise WordModeDisabledError(
            "Sentence recognition is disabled. Set SIGNSAGE_WORD_MODE_ENABLED=true "
            "to enable the word-level recognition path."
        )

    landmarks = _extract_landmarks(payload)
    if len(landmarks) > 400:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_MESSAGE", "message": "too many frames; max 400"},
        )
    adapter = _get_word_adapter(settings)

    pipeline = WordSequencePipeline(
        adapter,
        min_motion=settings.min_motion_energy,
        min_confidence=settings.min_confidence,
    )
    result = pipeline.transcribe(landmarks)
    signs = pipeline.last_signs

    vocab = set(adapter.labels)
    gloss_sequence = [
        {"gloss": s.text, "confidence": round(float(s.confidence), 4)} for s in signs
    ]
    sentence, method = compose_words([s.text for s in signs], vocab, settings)
    accepted = bool(signs) and bool(sentence)

    # --- Best-effort history save for signed-in users only ---
    save_history = bool(payload.get("save_history", True))
    if user and history and save_history and accepted:
        joined = " ".join(s.text for s in signs)
        confidence = float(np.mean([s.confidence for s in signs]))
        await history.try_record_word(user, NewHistoryRecord(
            direction=Direction.SIGN_TO_TEXT,
            output_text=sentence,
            result_type=ResultType.MODEL,
            model_version=adapter.model_version,
            input_text=joined,
            gloss=joined,
            confidence=max(0.0, min(1.0, confidence)),
            sequence=1,
            title=f"Sign to text: {sentence[:60]}",
            status="completed",
        ))

    return {
        "gloss_sequence": gloss_sequence,
        "sentence": sentence,
        "method": method,
        "accepted": accepted,
        "model_version": adapter.model_version,
        "threshold": adapter.confidence_threshold,
        "motion_energy": result.motion_energy,
        "status": "ok",
    }


@router.post("/compose", dependencies=[Depends(rate_limit_dependency("compose"))])
async def compose_sentence(
    payload: dict,
    settings: Settings = Depends(get_settings),
    user: Optional[AuthenticatedUser] = Depends(get_request_user),
    history: Optional[HistoryService] = Depends(get_optional_history_service),
) -> dict:
    if not settings.word_mode_enabled:
        raise WordModeDisabledError(
            "Word composition is disabled. Set SIGNSAGE_WORD_MODE_ENABLED=true "
            "to enable the word-level recognition path."
        )

    words = payload.get("words")
    if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_MESSAGE", "message": "words must be a list of strings"},
        )

    adapter = _get_word_adapter(settings)
    vocab = set(adapter.labels)
    invalid = [w for w in words if w not in vocab]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_MESSAGE",
                "message": f"unknown words: {sorted(set(invalid))}; vocabulary: {sorted(vocab)}",
            },
        )

    sentence, method = compose_words(words, vocab, settings)

    # --- Best-effort history save for signed-in users only ---
    save_history = bool(payload.get("save_history", True))
    if user and history and save_history and words:
        confidences = payload.get("confidences")
        confidence = None
        if isinstance(confidences, list) and confidences:
            try:
                vals = [float(c) for c in confidences if isinstance(c, (int, float)) and not isinstance(c, bool)]
                if vals:
                    mean = sum(vals) / len(vals)
                    confidence = max(0.0, min(1.0, mean))
            except (TypeError, ValueError):
                confidence = None
        accepted_words = [w for w in words if w != "rest"]
        joined_words = " ".join(accepted_words)
        title = f"Word recognition: {sentence[:60]}" if sentence else "Word recognition"
        await history.try_record_word(user, NewHistoryRecord(
            direction=Direction.SIGN_TO_TEXT,
            output_text=sentence,
            result_type=ResultType.MODEL,
            model_version=adapter.model_version,
            input_text=joined_words,
            gloss=joined_words,
            confidence=confidence,
            sequence=1,
            title=title,
            status="completed",
        ))

    return {"sentence": sentence, "method": method}



