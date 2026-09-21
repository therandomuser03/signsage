from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.api import (
    ErrorMessage,
    LandmarksFrame,
    SentenceBoundary,
    TextInput,
    TranslationPose,
    TranslationText,
)
from app.core.auth import verify_supabase_token
from app.services.model_manager import ModelManager
from app.services.session import SessionError, TranslationSession


logger = logging.getLogger("signsage.websocket")

router = APIRouter()


# ---------------------------------------------------------------------------
# SignSage landmark contract
# ---------------------------------------------------------------------------
# MediaPipe Holistic:
#
#   Pose       = 33
#   Face       = 468
#   Left hand  = 21
#   Right hand = 21
#
# Total:
#
#   543 landmarks × 3 coordinates = 1629 values/frame
#
# IMPORTANT:
# websocket.py receives RAW MediaPipe coordinates.
# Canonical training-time normalization must happen exactly once downstream
# through pose_preprocessing.normalize_pose_sequence().
# ---------------------------------------------------------------------------

EXPECTED_LANDMARKS = 543
COORDS_PER_LANDMARK = 3
EXPECTED_FEATURE_DIM = (
    EXPECTED_LANDMARKS * COORDS_PER_LANDMARK
)


def _manager(websocket: WebSocket) -> ModelManager:
    manager = getattr(
        websocket.app.state,
        "model_manager",
        None,
    )

    if manager is None:
        raise RuntimeError(
            "Models are not loaded."
        )

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

    await websocket.send_json(
        payload.model_dump()
    )


def _decode_landmark_frame(
    landmarks: list[float],
) -> np.ndarray:
    """
    Validate and convert one WebSocket landmark frame.

    Expected input:
        flat list of 1629 finite numeric values

    Output:
        float32 ndarray shaped [543, 3]

    This function intentionally DOES NOT normalize the coordinates.
    Normalization belongs to the model preprocessing layer so that the
    exact training-time transform is applied once and only once.
    """

    if len(landmarks) != EXPECTED_FEATURE_DIM:
        raise ValueError(
            "Invalid landmark frame: expected "
            f"{EXPECTED_FEATURE_DIM} values "
            f"(543 landmarks × 3), received "
            f"{len(landmarks)}."
        )

    try:
        frame = np.asarray(
            landmarks,
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Landmark frame contains non-numeric values."
        ) from exc

    if frame.size != EXPECTED_FEATURE_DIM:
        raise ValueError(
            "Invalid landmark frame size."
        )

    if not np.isfinite(frame).all():
        raise ValueError(
            "Landmark frame contains NaN or infinite values."
        )

    return np.ascontiguousarray(
        frame.reshape(
            EXPECTED_LANDMARKS,
            COORDS_PER_LANDMARK,
        ),
        dtype=np.float32,
    )


@router.websocket("/api/v1/ws/translate")
async def translate(websocket: WebSocket):
    """
    One WebSocket connection = one shared-device SignSage session.

    Supported client -> server messages:

        landmarks.frame
            Raw MediaPipe Holistic landmark frame.

        sentence.boundary
            Indicates that the current landmark sequence should be
            sent through Sign -> Text inference.

        text.input
            Natural-language input for Text -> ISL.

    The WebSocket itself does not run MediaPipe and does not receive
    raw camera/video data. The browser performs MediaPipe extraction
    locally and sends only numeric landmarks.
    """

    settings = websocket.app.state.settings

    # -----------------------------------------------------------------------
    # Optional authentication
    # -----------------------------------------------------------------------

    user_id: str | None = None

    token = websocket.query_params.get(
        "token"
    )

    if (
        token
        and settings.enable_auth
        and settings.supabase_url
    ):
        user_id = await verify_supabase_token(
            token,
            settings.supabase_url,
            settings.supabase_anon_key,
        )

        if user_id is None:
            await websocket.close(
                code=4401,
                reason="Unauthorized",
            )
            return

    elif token and not settings.enable_auth:
        # Authentication is disabled. The presence of a token is ignored.
        user_id = None

    await websocket.accept()

    manager = _manager(websocket)

    session = TranslationSession(
        user_id=user_id
    )

    try:
        while True:
            # ----------------------------------------------------------------
            # Receive WebSocket message
            # ----------------------------------------------------------------

            raw = await websocket.receive_text()

            if (
                len(raw.encode("utf-8"))
                > settings.max_ws_message_bytes
            ):
                await _send_error(
                    websocket,
                    session.session_id or "",
                    "WebSocket message is too large.",
                    "MESSAGE_TOO_LARGE",
                )
                continue

            # ----------------------------------------------------------------
            # Parse JSON
            # ----------------------------------------------------------------

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

            if not isinstance(payload, dict):
                await _send_error(
                    websocket,
                    session.session_id or "",
                    "WebSocket payload must be a JSON object.",
                    "INVALID_MESSAGE",
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

            # ----------------------------------------------------------------
            # Message dispatch
            # ----------------------------------------------------------------

            try:

                # ============================================================
                # ISL -> English
                # ============================================================

                if msg_type == "landmarks.frame":

                    msg = LandmarksFrame.model_validate(
                        payload
                    )

                    session.bind_session_id(
                        msg.session_id
                    )

                    # --------------------------------------------------------
                    # Validate the exact 1629-value contract.
                    #
                    # IMPORTANT:
                    # This is RAW MediaPipe data.
                    # Do not normalize here.
                    # --------------------------------------------------------

                    frame_arr = (
                        _decode_landmark_frame(
                            msg.landmarks
                        )
                    )

                    # --------------------------------------------------------
                    # Add raw frame to the sentence buffer.
                    # --------------------------------------------------------

                    session.add_frame(
                        frame_arr,
                        settings.max_websocket_frames,
                    )

                    logger.debug(
                        "landmarks.frame received "
                        "session=%s "
                        "frame_idx=%d "
                        "total_frames=%d",
                        session.session_id,
                        session.buffered_frame_count,
                        (
                            len(session.frames)
                            if hasattr(
                                session,
                                "frames",
                            )
                            else session.buffered_frame_count
                        ),
                    )

                # ============================================================
                # Sentence boundary
                # ============================================================

                elif msg_type == "sentence.boundary":

                    msg = SentenceBoundary.model_validate(
                        payload
                    )

                    session.bind_session_id(
                        msg.session_id
                    )

                    sequence = (
                        session.take_sentence()
                    )

                    sequence_length = (
                        len(sequence)
                        if hasattr(
                            sequence,
                            "__len__",
                        )
                        else 0
                    )

                    logger.info(
                        "sentence.boundary received "
                        "session=%s "
                        "trigger=%s "
                        "seq_len=%d",
                        session.session_id,
                        getattr(
                            msg,
                            "trigger",
                            None,
                        ),
                        sequence_length,
                    )

                    # --------------------------------------------------------
                    # Don't run inference on an empty sentence.
                    # --------------------------------------------------------

                    if sequence_length == 0:
                        await _send_error(
                            websocket,
                            session.session_id
                            or "",
                            "No landmark frames were captured for this sentence.",
                            "EMPTY_SEQUENCE",
                        )
                        continue

                    # --------------------------------------------------------
                    # Convert the buffered sequence into the canonical
                    # ndarray representation expected by the model service.
                    #
                    # The model service is responsible for applying:
                    #
                    #   normalize_pose_sequence()
                    #
                    # exactly once.
                    # --------------------------------------------------------

                    sequence_array = np.asarray(
                        sequence,
                        dtype=np.float32,
                    )

                    if (
                        sequence_array.ndim != 3
                        or sequence_array.shape[1:]
                        != (
                            EXPECTED_LANDMARKS,
                            COORDS_PER_LANDMARK,
                        )
                    ):
                        raise ValueError(
                            "Buffered landmark sequence has invalid shape: "
                            f"{tuple(sequence_array.shape)}. "
                            "Expected [T, 543, 3]."
                        )

                    if not np.isfinite(
                        sequence_array
                    ).all():
                        raise ValueError(
                            "Buffered landmark sequence contains "
                            "NaN or infinite values."
                        )

                    started = time.perf_counter()

                    # --------------------------------------------------------
                    # PyTorch inference is synchronous.
                    #
                    # Run it outside the FastAPI event loop so another
                    # WebSocket connection is not blocked by inference.
                    #
                    # IMPORTANT:
                    # predict_sign_to_text_with_meta() must call
                    # prepare_landmarks(..., input_mode="raw")
                    # internally.
                    # --------------------------------------------------------

                    # Hand-only framing is supported. Use matching ablation
                    # model when face landmarks are absent instead of feeding
                    # zero-filled face features into full-model inference.
                    has_face = bool(np.any(sequence_array[:, 33:501]))
                    use_noface = not has_face and manager.sign_to_text_noface is not None

                    (
                        text,
                        frames_used,
                        model_latency,
                        is_fallback,
                        confidence,
                        gated_reason,
                    ) = await asyncio.to_thread(
                        manager.predict_sign_to_text_with_meta,
                        sequence_array,
                        no_face=use_noface,
                    )

                    total_latency = (
                        time.perf_counter()
                        - started
                    ) * 1000

                    logger.info(
                        "S2T inference "
                        "session=%s "
                        "text=%s "
                        "frames_used=%s "
                        "fallback=%s "
                        "confidence=%s "
                        "reason=%s",
                        session.session_id,
                        (
                            text[:80]
                            if text
                            else "None"
                        ),
                        frames_used,
                        is_fallback,
                        confidence,
                        gated_reason,
                    )

                    await websocket.send_json(
                        TranslationText(
                            session_id=session.session_id,
                            text=text,
                            latency_ms=round(
                                max(
                                    model_latency,
                                    total_latency,
                                ),
                                3,
                            ),
                            confidence=round(
                                float(
                                    confidence
                                ),
                                3,
                            ),
                            is_fallback=bool(
                                is_fallback
                            ),
                            gated_reason=gated_reason,
                        ).model_dump()
                    )

                # ============================================================
                # English/Text -> ISL
                # ============================================================

                elif msg_type == "text.input":

                    msg = TextInput.model_validate(
                        payload
                    )

                    session.bind_session_id(
                        msg.session_id
                    )

                    started = time.perf_counter()

                    (
                        gloss,
                        poses,
                        model_latency,
                        provider,
                        is_fallback,
                    ) = await asyncio.to_thread(
                        manager.text_to_pose.predict,
                        msg.text,
                    )

                    total_latency = (
                        time.perf_counter()
                        - started
                    ) * 1000

                    session.record_text_to_pose_turn()

                    logger.info(
                        "T2P inference "
                        "session=%s "
                        "text=%s "
                        "gloss=%s "
                        "poses_len=%s "
                        "provider=%s "
                        "fallback=%s",
                        session.session_id,
                        msg.text[:30],
                        gloss,
                        (
                            len(poses)
                            if hasattr(
                                poses,
                                "__len__",
                            )
                            else "?"
                        ),
                        provider,
                        is_fallback,
                    )

                    await websocket.send_json(
                        TranslationPose(
                            session_id=session.session_id,
                            gloss=gloss,
                            frames=poses.tolist(),
                            fps=25,
                            latency_ms=round(
                                max(
                                    model_latency,
                                    total_latency,
                                ),
                                3,
                            ),
                            is_fallback=bool(
                                is_fallback
                            ),
                        ).model_dump()
                    )

                # ============================================================
                # Unknown message
                # ============================================================

                else:

                    await _send_error(
                        websocket,
                        session.session_id or "",
                        (
                            "Unknown message type: "
                            f"{msg_type!r}."
                        ),
                        "UNKNOWN_MESSAGE_TYPE",
                    )

            # ----------------------------------------------------------------
            # Known application errors
            # ----------------------------------------------------------------

            except SessionError as exc:

                await _send_error(
                    websocket,
                    session.session_id
                    or str(
                        payload.get(
                            "session_id",
                            "",
                        )
                    ),
                    str(exc),
                    "SESSION_ERROR",
                )

            except ValueError as exc:

                await _send_error(
                    websocket,
                    session.session_id
                    or str(
                        payload.get(
                            "session_id",
                            "",
                        )
                    ),
                    str(exc),
                    "INVALID_MESSAGE",
                )

            except FileNotFoundError:

                logger.exception(
                    "Model artifact missing during "
                    "WebSocket inference."
                )

                await _send_error(
                    websocket,
                    session.session_id or "",
                    "Required model artifact is unavailable.",
                    "MODEL_UNAVAILABLE",
                )

            # ----------------------------------------------------------------
            # Unexpected inference/application error
            # ----------------------------------------------------------------

            except Exception:

                logger.exception(
                    "Unhandled WebSocket inference error."
                )

                await _send_error(
                    websocket,
                    session.session_id
                    or str(
                        payload.get(
                            "session_id",
                            "",
                        )
                    ),
                    "Translation inference failed.",
                    "INFERENCE_FAILED",
                )

    # ------------------------------------------------------------------------
    # Socket closed by client
    # ------------------------------------------------------------------------

    except WebSocketDisconnect:

        logger.info(
            "WebSocket disconnected session=%s",
            session.session_id,
        )

    # ------------------------------------------------------------------------
    # Unexpected connection-level failure
    # ------------------------------------------------------------------------

    except Exception:

        logger.exception(
            "WebSocket connection failed session=%s",
            session.session_id,
        )

        try:
            await _send_error(
                websocket,
                session.session_id or "",
                "WebSocket connection failed.",
                "CONNECTION_FAILED",
            )

        except Exception:
            pass
