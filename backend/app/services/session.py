from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


class SessionError(ValueError):
    """Raised for session/turn-state problems (wrong session_id, empty
    sentence, buffer overflow, etc). Callers translate this into the
    WebSocket `error` message type -- see docs/API.md."""


@dataclass
class TranslationSession:
    """Per-WebSocket-connection state for one shared-device conversation.

    SignSage's MVP is explicitly single-device (see ARCHITECTURE.md's
    "single-device, not two-device relay" rationale and AGENTS.md's
    deferred-features list): one browser tab, one shared camera/mic, two
    people taking turns. That means one WebSocket connection *is* one
    session for the lifetime of this class -- there is deliberately no
    session registry, no room code, and no cross-connection lookup here.
    Adding one would be scaffolding the two-device relay feature that is
    explicitly deferred until a real remote-communication need is
    demonstrated.

    This exists so the turn-taking / frame-buffering state that used to be
    loose local variables inside the WebSocket handler is instead a small,
    plain object that can be unit-tested without booting FastAPI or a real
    socket -- see tests/test_session.py.

    Turn model: Person A (signing) and Person B (typing/speaking-via-STT)
    share one connection and are *not* mutually exclusive -- either can
    produce a message at any time, matching the "conversation mode passed
    between two people" UX in ARCHITECTURE.md. This class only tracks the
    state needed for the *signing* side (an in-progress buffered sentence),
    since text.input has no buffering step; it also counts completed turns
    in both directions for /api/v1/health-style visibility.
    """

    session_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    frames: list[np.ndarray] = field(default_factory=list)
    sign_to_text_turns: int = 0
    text_to_pose_turns: int = 0

    def bind_session_id(self, session_id: str) -> None:
        """Adopt `session_id` on first use; reject a change mid-connection.

        One WebSocket connection is one session for this MVP, so a
        mid-connection session_id change almost certainly indicates a
        client bug (e.g. reusing a socket across browser tabs) rather than
        a legitimate use case -- fail loudly per DESIGN.md rather than
        silently re-binding.
        """
        if not session_id:
            raise SessionError("session_id must not be empty.")
        if self.session_id is None:
            self.session_id = session_id
        elif self.session_id != session_id:
            raise SessionError(
                "session_id changed within one WebSocket connection; one "
                "connection is one session for the single-device MVP."
            )
        self.last_activity_at = time.time()

    def add_frame(self, frame: np.ndarray, max_buffered_frames: int) -> None:
        if len(self.frames) >= max_buffered_frames:
            raise SessionError(
                "Maximum buffered frames reached; send a sentence boundary."
            )
        self.frames.append(frame)
        self.last_activity_at = time.time()

    @property
    def buffered_frame_count(self) -> int:
        return len(self.frames)

    def take_sentence(self) -> np.ndarray:
        """Consume and clear the buffered frames for one completed sentence."""
        if not self.frames:
            raise SessionError("No landmark frames are buffered for this sentence.")
        sequence = np.stack(self.frames, axis=0)
        self.frames.clear()
        self.sign_to_text_turns += 1
        self.last_activity_at = time.time()
        return sequence

    def record_text_to_pose_turn(self) -> None:
        self.text_to_pose_turns += 1
        self.last_activity_at = time.time()

    def snapshot(self) -> dict:
        """Small dict for logging/diagnostics -- not part of the wire contract."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "buffered_frames": self.buffered_frame_count,
            "sign_to_text_turns": self.sign_to_text_turns,
            "text_to_pose_turns": self.text_to_pose_turns,
        }
