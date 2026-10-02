"""WebSocket endpoint for live Sign->Text translation. Protocol: schemas/websocket.py."""
from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.dependencies import build_translation_service
from app.core.errors import SignSageError
from app.core.security import verify_access_token
from app.domain.translation.entities import Direction, NewHistoryRecord
from app.domain.translation.policies import classify_result, should_persist
from app.schemas.websocket import (
    AuthOkMessage, ErrorCode, ErrorMessage, TranslationTextMessage,
    UnknownMessageTypeError, parse_incoming,
)
from app.services.history_service import build_history_service
from app.services.session_service import SequenceTooLongError, TranslationSession

logger = logging.getLogger(__name__)
router = APIRouter()


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_json(ErrorMessage(code=code, message=message).model_dump())


@router.websocket("/ws/translate")
async def translate_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()
    session = TranslationSession(session_id=str(uuid.uuid4()), settings=settings)
    user = None

    try:
        # First load reads a ~100 MB checkpoint: keep it off the event loop.
        service = await run_in_threadpool(build_translation_service, settings)
    except Exception as exc:  # model not promoted / bundle missing or invalid
        logger.warning("[MODEL] translation unavailable for session %s: %s", session.session_id, exc)
        await _send_error(websocket, ErrorCode.MODEL_UNAVAILABLE, str(exc))
        await websocket.close()
        return
    history = build_history_service(settings)
    logger.info("[WEBSOCKET] session %s connected", session.session_id)

    try:
        while True:
            text = await websocket.receive_text()
            if len(text.encode("utf-8")) > settings.max_message_bytes:
                await _send_error(websocket, ErrorCode.MESSAGE_TOO_LARGE, "Message exceeds the size limit.")
                continue
            try:
                raw = json.loads(text)
            except ValueError:
                await _send_error(websocket, ErrorCode.INVALID_JSON, "Message is not valid JSON.")
                continue
            if not isinstance(raw, dict):
                await _send_error(websocket, ErrorCode.INVALID_MESSAGE, "Message must be a JSON object.")
                continue
            try:
                message = parse_incoming(raw)
            except UnknownMessageTypeError as exc:
                await _send_error(websocket, ErrorCode.UNKNOWN_MESSAGE_TYPE, str(exc))
                continue
            except ValueError as exc:
                await _send_error(websocket, ErrorCode.INVALID_MESSAGE, str(exc))
                continue

            if message.type == "auth":
                try:
                    user = await verify_access_token(message.access_token, settings)
                except SignSageError as exc:
                    await _send_error(websocket, exc.code, exc.message)
                    continue
                await websocket.send_json(AuthOkMessage(user_id=user.id).model_dump())

            elif message.type == "landmarks.frame":
                try:
                    session.add_frame(message.landmarks)
                except SequenceTooLongError as exc:
                    await _send_error(websocket, ErrorCode.SESSION_ERROR, str(exc))
                    session.reset()
                    continue
                # Rolling interim inference on a trailing window every stride
                # frames: partial results without waiting for sentence.boundary.
                # Best-effort by design -- a failure here must never kill the
                # socket or disturb the final buffer.
                count = session.frame_count()
                window = settings.stream_window_frames
                stride = settings.stream_stride_frames
                if window > 0 and stride > 0 and count >= window and (count - window) % stride == 0:
                    frames = session.window(window)
                    if frames is not None:
                        try:
                            interim = await run_in_threadpool(service.translate_sign_to_text, frames)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("[INFERENCE] session %s interim failed: %s",
                                           session.session_id, exc)
                            continue
                        if interim.accepted and session.note_interim(interim.text):
                            await websocket.send_json(TranslationTextMessage(
                                text=interim.text, accepted=interim.accepted, reason=interim.reason,
                                confidence=interim.confidence,
                                result_type=classify_result(interim).value,
                                history_id=None, final=False,
                            ).model_dump())

            elif message.type == "sentence.boundary":
                if settings.auth_required and user is None:
                    await _send_error(websocket, ErrorCode.UNAUTHORIZED, "Send an auth message first.")
                    session.reset()
                    continue
                if not session.has_frames():
                    await _send_error(websocket, ErrorCode.EMPTY_SEQUENCE, "No frames buffered.")
                    continue
                sequence = session.take_sequence()
                n_frames = sequence.shape[0]
                started = time.perf_counter()
                try:
                    result = await run_in_threadpool(service.translate_sign_to_text, sequence)
                except ValueError as exc:
                    await _send_error(websocket, ErrorCode.INVALID_MESSAGE, str(exc))
                    continue
                latency_ms = (time.perf_counter() - started) * 1000
                # No landmark data, no translation text: counts and gate
                # outcome only (text is user content; frames are geometry).
                logger.info(
                    "[INFERENCE] session %s frames=%d accepted=%s reason=%s "
                    "confidence=%.3f latency_ms=%.1f",
                    session.session_id, n_frames, result.accepted, result.reason,
                    result.confidence, latency_ms,
                )
                history_id = None
                if user and history and message.save_history and should_persist(result):
                    history_id = await history.try_record(user, NewHistoryRecord(
                        direction=Direction.SIGN_TO_TEXT, output_text=result.text,
                        result_type=classify_result(result), model_version=settings.model_version,
                        confidence=result.confidence,
                    ))
                await websocket.send_json(TranslationTextMessage(
                    text=result.text, accepted=result.accepted, reason=result.reason,
                    confidence=result.confidence, result_type=classify_result(result).value,
                    history_id=history_id, final=True,
                ).model_dump())
    except WebSocketDisconnect:
        logger.info("[WEBSOCKET] session %s disconnected", session.session_id)
