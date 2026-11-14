"""Continuous sign-sequence pipeline built on top of the isolated-word recognizer.

Wraps :class:`app.ml.isolated_words.IsolatedWordAdapter` to handle a
continuous stream of landmark frames. It runs inference on a sliding
window, detects sign boundaries via motion-energy gating, suppresses
repeated predictions of the same held gesture, and emits an ordered
sequence of recognized signs.

This is NOT sentence translation: it recognizes individual isolated signs
from a continuous camera stream and preserves their order. Continuous
ISL-to-English translation (grammar-aware, sentence-level) remains a
separate unresolved milestone requiring sentence-level training data
(ISLTranslate, gated on Hugging Face).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.ml.adapters.sign_to_text_adapter import TranscriptionResult
from app.ml.isolated_words import (
    FRAME_COUNT,
    IsolatedWordAdapter,
    prepare_isolated_features,
)
from app.ml.preprocessing.sequence import compute_motion_energy


#: Frames of low-motion required to confirm a sign boundary (end of one sign,
#: start of the next). At ~10 fps, 15 frames ~= 1.5 s of stillness.
BOUNDARY_MOTION_FRAMES = 15

#: Minimum frames a sign must be held before it can be accepted. Prevents
#: transient detections from flickering. At ~10 fps, 8 frames ~= 0.8 s.
MIN_SIGN_HOLD_FRAMES = 8

#: Cooldown frames after accepting a sign before the next sign can be
#: accepted. Prevents flooding from a held gesture. At ~10 fps, 20 frames
#: ~= 2 s.
SIGN_COOLDOWN_FRAMES = 20

#: Maximum frames buffered before forcing a boundary check (prevents
#: unbounded memory growth from a never-ending sign).
MAX_BUFFER_FRAMES = 300


@dataclass
class RecognizedSign:
    """One recognized sign in a continuous sequence."""

    text: str
    confidence: float
    reason: str
    start_frame: int
    end_frame: int
    frame_index: int  # global frame index when the sign was finalized


@dataclass
class ContinuousSignResult:
    """Result of processing one frame through the continuous pipeline."""

    accepted_signs: list[RecognizedSign]  # all finalized signs so far
    current_sign: Optional[str]  # sign currently being held (not yet finalized)
    current_confidence: float
    reason: str
    motion_energy: float
    boundary_detected: bool
    total_frames: int


class ContinuousSignPipeline:
    """Stateful continuous sign-sequence recognizer.

    Consumes one landmark frame at a time. Maintains an internal buffer and
    runs inference on sliding windows. When a sign boundary (low-motion
    period) is detected, the buffered sign segment is classified and, if
    accepted, appended to the sequence.

    Usage::

        pipe = ContinuousSignPipeline(adapter)
        for frame in landmark_stream:
            result = pipe.feed_frame(frame)
            # result.accepted_signs grows as signs are finalized
    """

    def __init__(
        self,
        adapter: IsolatedWordAdapter,
        *,
        window_frames: int = FRAME_COUNT,
        stride_frames: int | None = None,
        min_motion: float = 0.002,
        min_confidence: float = 0.5,
        boundary_motion_frames: int = BOUNDARY_MOTION_FRAMES,
        min_sign_hold_frames: int = MIN_SIGN_HOLD_FRAMES,
        sign_cooldown_frames: int = SIGN_COOLDOWN_FRAMES,
        max_buffer_frames: int = MAX_BUFFER_FRAMES,
    ):
        self.adapter = adapter
        self.window_frames = window_frames
        self.stride_frames = stride_frames or max(1, window_frames // 2)
        self.min_motion = min_motion
        self.min_confidence = min_confidence
        self.boundary_motion_frames = boundary_motion_frames
        self.min_sign_hold_frames = min_sign_hold_frames
        self.sign_cooldown_frames = sign_cooldown_frames
        self.max_buffer_frames = max_buffer_frames

        # Buffer of raw landmark frames (T, 543, 3).
        self._buffer: list[np.ndarray] = []
        # Global frame counter.
        self._frame_index = 0
        # Motion energy per buffered frame (for boundary detection).
        self._motion_history: deque[float] = deque(maxlen=max_buffer_frames)
        # Finalized sign sequence.
        self.accepted_signs: list[RecognizedSign] = []
        # State for the sign currently being held.
        self._current_sign: Optional[str] = None
        self._current_sign_start_frame: int = 0
        self._current_sign_frames: int = 0
        self._current_best_confidence: float = 0.0
        # Last token accepted by flush(): [{text, confidence, ...}], kept so
        # callers can inspect per-gloss confidences after a flush.
        self.last_flushed: list[RecognizedSign] = []
        # Cooldown counter (frames remaining before next sign can be accepted).
        self._cooldown_remaining: int = 0
        # Low-motion run length (for boundary detection).
        self._low_motion_run: int = 0
        # Last accepted sign text (for repeat suppression).
        self._last_accepted_text: Optional[str] = None

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    def feed_frame(self, landmarks_xyz: np.ndarray) -> ContinuousSignResult:
        """Feed one (543, 3) landmark frame. Returns pipeline result."""
        arr = np.asarray(landmarks_xyz, dtype=np.float32)
        if arr.shape != (543, 3):
            raise ValueError(
                f"Expected (543, 3) landmark frame, received {arr.shape}"
            )

        motion = float(compute_motion_energy(
            np.asarray(self._buffer + [arr], dtype=np.float32)
            if self._buffer
            else arr[None]
        ))
        self._motion_history.append(motion)
        self._buffer.append(arr)
        self._frame_index += 1

        result = self._process_buffer(motion)

        # Enforce max buffer size.
        if len(self._buffer) > self.max_buffer_frames:
            self._buffer = self._buffer[-self.max_buffer_frames :]
            self._motion_history = deque(
                self._motion_history[-self.max_buffer_frames :],
                maxlen=self.max_buffer_frames,
            )

        return result

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _process_buffer(self, motion: float) -> ContinuousSignResult:
        """Run inference on the buffered window and detect boundaries."""
        # Decrement cooldown.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Track low-motion run for boundary detection.
        if motion < self.min_motion:
            self._low_motion_run += 1
        else:
            self._low_motion_run = 0

        boundary_detected = self._low_motion_run >= self.boundary_motion_frames

        # If we have enough frames for inference, run the model.
        current_sign: Optional[str] = None
        current_confidence = 0.0
        reason = "collecting"

        if len(self._buffer) >= self.window_frames:
            window = np.stack(self._buffer[-self.window_frames :])
            result = self.adapter.transcribe(
                window,
                min_motion=self.min_motion,
                min_confidence=self.min_confidence,
            )
            if result.accepted:
                current_sign = result.text
                current_confidence = result.confidence
                reason = result.reason

            # Update current sign tracking.
            if current_sign is not None:
                if self._current_sign == current_sign:
                    self._current_sign_frames += 1
                    self._current_best_confidence = max(
                        self._current_best_confidence, current_confidence
                    )
                else:
                    # New sign detected - only adopt if we've held it long enough
                    # or if the previous sign wasn't stable.
                    if (
                        self._current_sign is None
                        or self._current_sign_frames >= self.min_sign_hold_frames
                    ):
                        self._current_sign = current_sign
                        self._current_sign_start_frame = (
                            self._frame_index - self.window_frames
                        )
                        self._current_sign_frames = 1
                        self._current_best_confidence = current_confidence
            else:
                # No sign recognized - reset current sign if held long enough.
                if self._current_sign is not None:
                    if self._current_sign_frames >= self.min_sign_hold_frames:
                        self._finalize_current_sign()
                    self._current_sign = None
                    self._current_sign_frames = 0
                    self._current_best_confidence = 0.0
        else:
            reason = "insufficient_frames"

        # On boundary, finalize the current sign.
        if boundary_detected and self._current_sign is not None:
            if self._current_sign_frames >= self.min_sign_hold_frames:
                self._finalize_current_sign()
            self._current_sign = None
            self._current_sign_frames = 0
            self._current_best_confidence = 0.0
            self._low_motion_run = 0

        return ContinuousSignResult(
            accepted_signs=list(self.accepted_signs),
            current_sign=self._current_sign,
            current_confidence=self._current_best_confidence,
            reason=reason,
            motion_energy=motion,
            boundary_detected=boundary_detected,
            total_frames=self._frame_index,
        )

    def _finalize_current_sign(self) -> None:
        """Move the current held sign into the accepted sequence."""
        if self._current_sign is None:
            return
        # Repeat suppression: don't accept the same sign twice in a row.
        if self._current_sign == self._last_accepted_text:
            return
        # Cooldown check.
        if self._cooldown_remaining > 0:
            return

        sign = RecognizedSign(
            text=self._current_sign,
            confidence=self._current_best_confidence,
            reason="ok",
            start_frame=self._current_sign_start_frame,
            end_frame=self._frame_index,
            frame_index=self._frame_index,
        )
        self.accepted_signs.append(sign)
        self._last_accepted_text = self._current_sign
        self._cooldown_remaining = self.sign_cooldown_frames

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def flush(self) -> list[RecognizedSign]:
        """Finalize any held sign and return all accepted signs.

        Call this at session end (sentence.boundary) to flush the buffer.
        """
        if self._current_sign is not None:
            if self._current_sign_frames >= self.min_sign_hold_frames:
                self._finalize_current_sign()
        self._current_sign = None
        self._current_sign_frames = 0
        self._current_best_confidence = 0.0
        flushed = list(self.accepted_signs)
        # Kept for callers that want the finalized sequence (e.g. the REST
        # /sentence endpoint reads the per-gloss confidences from here).
        self.last_flushed = flushed
        self.accepted_signs = []
        self._buffer = []
        self._motion_history = deque(maxlen=self.max_buffer_frames)
        self._frame_index = 0
        self._low_motion_run = 0
        self._cooldown_remaining = 0
        self._last_accepted_text = None
        return flushed

    def reset(self) -> None:
        """Reset all pipeline state without flushing accepted signs."""
        self._buffer = []
        self._motion_history = deque(maxlen=self.max_buffer_frames)
        self._frame_index = 0
        self._current_sign = None
        self._current_sign_start_frame = 0
        self._current_sign_frames = 0
        self._current_best_confidence = 0.0
        self._low_motion_run = 0
        self._cooldown_remaining = 0
        self._last_accepted_text = None
        self.last_flushed = []

    @property
    def current_sign(self) -> Optional[str]:
        return self._current_sign

    # ------------------------------------------------------------------
    # Batch interface (compatible with IsolatedWordAdapter.transcribe)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        landmarks_xyz: np.ndarray,
        min_motion: float = 0.002,
        min_confidence: float = 0.5,
    ) -> TranscriptionResult:
        """Process a full landmark sequence as a continuous sign stream.

        Feeds each frame incrementally through the pipeline, then flushes
        to finalize any held sign. Returns a single ``TranscriptionResult``
        whose ``text`` is the space-joined sequence of accepted signs.

        This maintains interface compatibility with
        :class:`app.ml.adapters.sign_to_text_adapter.SignToTextAdapter` /
        :class:`app.ml.isolated_words.IsolatedWordAdapter` so the existing
        ``TranslationService`` can wrap either without changes.
        """
        arr = np.asarray(landmarks_xyz, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (543, 3):
            raise ValueError(
                f"Expected landmarks shaped (T, 543, 3), received {arr.shape}"
            )
        if len(arr) < 2:
            raise ValueError(
                f"Need at least two frames, received {len(arr)}"
            )

        # Reset pipeline state for a fresh sequence.
        self.reset()

        motion_energy = 0.0
        reason = "collecting"
        for frame in arr:
            result = self.feed_frame(frame)
            motion_energy = result.motion_energy
            reason = result.reason

        # For sequences shorter than the window, run direct adapter inference
        # as a batch-mode fallback (simulating a single-held-sign stream).
        if len(self._buffer) < self.window_frames and len(self._buffer) >= 2:
            direct_result = self.adapter.transcribe(
                np.stack(self._buffer),
                min_motion=min_motion,
                min_confidence=min_confidence,
            )
            if direct_result.accepted and direct_result.text:
                sign = RecognizedSign(
                    text=direct_result.text,
                    confidence=direct_result.confidence,
                    reason=direct_result.reason,
                    start_frame=0,
                    end_frame=len(self._buffer),
                    frame_index=self._frame_index,
                )
                self.accepted_signs.append(sign)
                self._last_accepted_text = direct_result.text
                self._cooldown_remaining = self.sign_cooldown_frames

        flushed = self.flush()
        text = " ".join(s.text for s in flushed)
        confidence = (
            float(np.mean([s.confidence for s in flushed])) if flushed else 0.0
        )
        accepted = len(flushed) > 0

        return TranscriptionResult(
            text=text,
            accepted=accepted,
            reason=reason if not accepted else "ok",
            motion_energy=motion_energy,
            confidence=confidence,
        )
