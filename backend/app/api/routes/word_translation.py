"""Word-level recognition REST endpoints (behind SIGNSAGE_WORD_MODE_ENABLED).

POST /api/v1/translation/word
    input  = landmarks: list of frames, each 543 x [x, y, z]
    output = { word, confidence, accepted, top3, threshold, model_version, status }
    accepted=false and word=null when max prob < threshold or the predicted
    label is "rest". Returns 503 WORD_MODE_DISABLED when the flag is off.

POST /api/v1/translation/compose
    input  = { words: [...] }
    output = { sentence, method }
    Validates each word against the loaded vocabulary. Uses Groq (when
    configured) with a strict prompt, falls back to a deterministic join.

Never presents word recognition as sentence translation.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.core.errors import SignSageError
from app.core.dependencies import get_request_user
from app.core.security import AuthenticatedUser
from app.ml.word_classifier_adapter import WordClassifierAdapter

logger = logging.getLogger("signsage.word_debug")
DEBUG_CAPTURES_DIR = Path(__file__).resolve().parents[3] / "debug_captures"
DEBUG_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/translation", tags=["translation"])

# Process-wide singleton for the word classifier bundle.
_word_adapter: Optional[WordClassifierAdapter] = None
_word_adapter_loaded = False


class WordModeDisabledError(SignSageError):
    code = "WORD_MODE_DISABLED"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


def load_word_adapter_at_startup(settings: Settings) -> None:
    """Load the word classifier model at startup (avoids first-request latency).

    Only engaged when SIGNSAGE_WORD_DEBUG=true so production behaviour is
    unchanged (lazy first-request load otherwise).
    """
    global _word_adapter, _word_adapter_loaded
    if not settings.word_mode_enabled or not settings.word_debug:
        return
    if _word_adapter_loaded:
        return
    model_dir = _resolve_word_model_dir(settings)
    t0 = time.perf_counter()
    _word_adapter = WordClassifierAdapter.load(model_dir, device=settings.device)
    _word_adapter_loaded = True
    logger.info(
        "WORD_DEBUG: model loaded at startup in %.3f ms (dir=%s)",
        (time.perf_counter() - t0) * 1000,
        model_dir,
    )


@router.on_event("startup")
async def _startup_load_word_model() -> None:
    load_word_adapter_at_startup(get_settings())


def _resolve_word_model_dir(settings: Settings) -> str:
    """Resolve word_model_dir relative to the repo root when relative.

    The default ``models/word_v3`` lives next to ``research/`` and ``backend/``
    at the repository root, so a relative value is resolved against the
    repository root (not the ``backend/`` package directory).
    """
    d = settings.word_model_dir or "models/word_v3"
    p = Path(d)
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[4]
        p = repo_root / d
    return str(p)


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


@router.post("/word")
async def word_recognition(
    payload: dict,
    settings: Settings = Depends(get_settings),
    user: Optional[AuthenticatedUser] = Depends(get_request_user),
) -> dict:
    if not settings.word_mode_enabled:
        raise WordModeDisabledError(
            "Word recognition is disabled. Set SIGNSAGE_WORD_MODE_ENABLED=true "
            "to enable the demo word-level recognition path."
        )

    t0_total = time.perf_counter()
    t_parse_start = time.perf_counter()
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
    t_parse_ms = (time.perf_counter() - t_parse_start) * 1000

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

    n_frames = int(landmarks.shape[0])
    payload_size_bytes = len(json.dumps(payload))
    left_present = float(np.any(landmarks[:, 501:522, :] != 0.0, axis=(1, 2)).mean())
    right_present = float(np.any(landmarks[:, 522:543, :] != 0.0, axis=(1, 2)).mean())

    adapter = _get_word_adapter(settings)
    result = adapter.classify(landmarks)
    t_total_ms = (time.perf_counter() - t0_total) * 1000

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


@router.post("/compose")
async def compose_sentence(
    payload: dict,
    settings: Settings = Depends(get_settings),
    user: Optional[AuthenticatedUser] = Depends(get_request_user),
) -> dict:
    if not settings.word_mode_enabled:
        raise WordModeDisabledError(
            "Word composition is disabled. Set SIGNSAGE_WORD_MODE_ENABLED=true "
            "to enable the demo word-level recognition path."
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

    sentence, method = _compose(words, settings)
    return {"sentence": sentence, "method": method}


def _compose(words: list[str], settings: Settings) -> tuple[str, str]:
    """Compose a sentence from recognised words.

    Tries Groq with a strict prompt when configured; falls back to a
    deterministic join (capitalised, ending punctuation) when the LLM is
    unavailable or the output fails the sanity check.
    """
    joined = " ".join(w for w in words if w != "rest").strip()
    if not joined:
        return "", "fallback"

    # --- LLM path (Groq) when configured ---
    if settings.groq_api_key:
        try:
            import httpx

            system = (
                "Use only the meaning of these words, one short natural English "
                "sentence, add no new facts, names or details. "
                "Reply with only the sentence, nothing else."
            )
            body = {
                "model": settings.groq_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": joined},
                ],
                "temperature": 0,
                "max_tokens": 64,
            }
            headers = {
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            }
            url = f"{settings.groq_base_url.rstrip('/')}/chat/completions"
            with httpx.Client(timeout=settings.groq_timeout_seconds) as client:
                resp = client.post(url, json=body, headers=headers)
                resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content and len(content) <= 200 and not any(ch in content for ch in "<>{}"):
                return content, "llm"
        except Exception:
            # fall through to deterministic
            pass

    # --- Deterministic fallback ---
    sentence = joined[0].upper() + joined[1:]
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence, "fallback"
