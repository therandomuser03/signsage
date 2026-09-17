from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.api import (
    LandmarksFrame,
    SentenceBoundary,
    TextInput,
    TranslationPose,
    TranslationText,
    ErrorMessage,
)
from app.core.auth import verify_supabase_token
from app.services.model_manager import ModelManager
from app.services.session import SessionError, TranslationSession


logger = logging.getLogger("signsage.websocket")
router = APIRouter()


def _manager(websocket: WebSocket) -> ModelManager:
    manager = getattr(websocket.app.state, "model_manager", None)
    if manager is None:
        raise RuntimeError("Models are not loaded.")
    return manager


async def _send_error(
    websocket: WebSocket,
    session_id: str,
    message: str,
    code: str,
) -> None:
    payload = ErrorMessage(
        session_id=session_id,
        message=message,
        code=code,
    )
    await websocket.send_json(payload.model_dump())


@router.websocket("/api/v1/ws/translate")
async def translate(websocket: WebSocket):
    """One WebSocket connection == one shared-device session (see
    app/services/session.py). Handles both directions independently:
    Person A's `landmarks.frame`/`sentence.boundary` stream and Person B's
    `text.input` are unrelated message flows sharing one socket -- neither
    blocks or waits on the other, matching the shared-device turn-taking
    model in ARCHITECTURE.md."""
    settings = websocket.app.state.settings
    # Optional auth: token via query param ?token=...
    user_id: str | None = None
    token = websocket.query_params.get("token")
    if token and settings.enable_auth and settings.supabase_url:
        user_id = await verify_supabase_token(token, settings.supabase_url, settings.supabase_anon_key)
        if user_id is None:
            await websocket.close(code=4401, reason="Unauthorized")
            return
    elif token and not settings.enable_auth:
        # Auth disabled — trust token-less but allow passthrough if frontend sent token (no verification)
        # ponytail: no JWT dep needed when disabled; frontend guard is primary
        user_id = None

    await websocket.accept()

    manager = _manager(websocket)
    session = TranslationSession(user_id=user_id)

    try:
        while True:
            raw = await websocket.receive_text()

            if len(raw.encode("utf-8")) > settings.max_ws_message_bytes:
                await _send_error(
                    websocket,
                    session.session_id or "",
                    "WebSocket message is too large.",
                    "MESSAGE_TOO_LARGE",
                )
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(
                    websocket,
                    session.session_id or "",
                    "Message must be valid JSON.",
                    "INVALID_JSON",
                )
                continue

            msg_type = payload.get("type")
            if not isinstance(msg_type, str):
                await _send_error(
                    websocket,
                    session.session_id or "",
                    "Message type is required.",
                    "INVALID_MESSAGE",
                )
                continue

            try:
                if msg_type == "landmarks.frame":
                    msg = LandmarksFrame.model_validate(payload)
                    session.bind_session_id(msg.session_id)
                    frame_arr = np.asarray(msg.landmarks, dtype=np.float32).reshape(543, 3)
                    session.add_frame(frame_arr, settings.max_websocket_frames)
                    logger.info("landmarks.frame received session=%s frame_idx=%d total_frames=%d", session.session_id, session.buffered_frame_count, len(session.frames) if hasattr(session, 'frames') else '?')

                elif msg_type == "sentence.boundary":
                    msg = SentenceBoundary.model_validate(payload)
                    session.bind_session_id(msg.session_id)
                    sequence = session.take_sentence()
                    logger.info("sentence.boundary received session=%s seq_len=%d", session.session_id, len(sequence) if hasattr(sequence, '__len__') else '?')

                    started = time.perf_counter()
                    # PyTorch inference is synchronous. Run it off the event
                    # loop so another connection is not blocked by CPU
                    # inference.
                    text, frames_used, model_latency, is_fallback, confidence, gated_reason = await asyncio.to_thread(
                        manager.predict_sign_to_text_with_meta,
                        sequence,
                        no_face=False,
                    )
                    total_latency = (time.perf_counter() - started) * 1000

                    logger.info("S2T inference session=%s text=%s fallback=%s reason=%s", session.session_id, text[:30] if text else "None", is_fallback, gated_reason)

                    await websocket.send_json(
                        TranslationText(
                            session_id=session.session_id,
                            text=text,
                            latency_ms=round(max(model_latency, total_latency), 3),
                            confidence=round(float(confidence), 3),
                            is_fallback=bool(is_fallback),
                            gated_reason=gated_reason,
                        ).model_dump()
                    )

                elif msg_type == "text.input":
                    msg = TextInput.model_validate(payload)
                    session.bind_session_id(msg.session_id)

                    started = time.perf_counter()
                    gloss, poses, model_latency, _provider = await asyncio.to_thread(
                        manager.text_to_pose.predict,
                        msg.text,
                    )
                    total_latency = (time.perf_counter() - started) * 1000
                    session.record_text_to_pose_turn()

                    logger.info("T2P inference session=%s text=%s gloss=%s poses_len=%d", session.session_id, msg.text[:30], gloss, len(poses) if hasattr(poses, '__len__') else '?')

                    await websocket.send_json(
                        TranslationPose(
                            session_id=session.session_id,
                            gloss=gloss,
                            frames=poses.tolist(),
                            fps=25,
                            latency_ms=round(max(model_latency, total_latency), 3),
                        ).model_dump()
                    )

                else:
                    await _send_error(
                        websocket,
                        session.session_id or "",
                        f"Unknown message type: {msg_type!r}.",
                        "UNKNOWN_MESSAGE_TYPE",
                    )

            except SessionError as exc:
                await _send_error(
                    websocket,
                    session.session_id or str(payload.get("session_id", "")),
                    str(exc),
                    "SESSION_ERROR",
                )
            except ValueError as exc:
                await _send_error(
                    websocket,
                    session.session_id or str(payload.get("session_id", "")),
                    str(exc),
                    "INVALID_MESSAGE",
                )
            except FileNotFoundError:
                logger.exception("Model artifact missing during WebSocket inference.")
                await _send_error(
                    websocket,
                    session.session_id or "",
                    "Required model artifact is unavailable.",
                    "MODEL_UNAVAILABLE",
                )
            except Exception:
                logger.exception("Unhandled WebSocket inference error.")
                await _send_error(
                    websocket,
                    session.session_id or str(payload.get("session_id", "")),
                    "Translation inference failed.",
                    "INFERENCE_FAILED",
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected session=%s", session.session_id)
    except Exception:
        logger.exception("WebSocket connection failed session=%s", session.session_id)
        try:
            await _send_error(
                websocket,
                session.session_id or "",
                "WebSocket connection failed.",
                "CONNECTION_FAILED",
            )
        except Exception:
            pass
