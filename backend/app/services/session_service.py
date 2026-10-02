"""Per-connection translation session state.

Buffers incoming landmark frames until a sentence.boundary message arrives,
then hands the full sequence to TranslationService. One instance per
WebSocket connection -- not shared across connections/threads.
"""
from __future__ import annotations

import numpy as np

from app.core.config import Settings


class SequenceTooLongError(ValueError):
    pass


class TranslationSession:
    def __init__(self, session_id: str, settings: Settings):
        self.session_id = session_id
        self._settings = settings
        self._frames = []
        # Rolling-interim stabilization state (repeat-agreement): the last
        # emitted interim text and the current candidate awaiting agreement.
        self._sent_interim: str | None = None
        self._pending_interim: str | None = None

    def add_frame(self, landmarks) -> None:
        # Generous hard ceiling -- resize_sequence() will later downsample to
        # max_frames_per_sequence regardless, but an unbounded buffer from a
        # client that never sends sentence.boundary is a real memory/DoS
        # concern, so cap it well above the model's own max_frames.
        if len(self._frames) >= self._settings.max_frames_per_sequence * 4:
            raise SequenceTooLongError(
                f"Session {self.session_id} exceeded the maximum buffered frame count."
            )
        self._frames.append(landmarks)

    def has_frames(self) -> bool:
        return len(self._frames) >= 2

    def frame_count(self) -> int:
        return len(self._frames)

    def window(self, n: int):
        """Trailing n frames as an (n, 543, 3) array, or None if buffered
        fewer. The buffer is NOT consumed: the final sentence.boundary still
        sees the full sequence."""
        if len(self._frames) < n:
            return None
        return np.asarray(self._frames[-n:], dtype=np.float32)

    def note_interim(self, text: str) -> bool:
        """Repeat-agreement stabilization: emit the first accepted prediction
        immediately, then only emit a *different* text after it repeats on two
        consecutive windows. Returns True when the caller should send it."""
        if text == self._sent_interim:
            return False
        if self._sent_interim is None or text == self._pending_interim:
            self._sent_interim = text
            self._pending_interim = None
            return True
        self._pending_interim = text
        return False

    def take_sequence(self) -> np.ndarray:
        """Pop the buffered frames as a (T, 543, 3) array and reset the buffer."""
        arr = np.asarray(self._frames, dtype=np.float32)
        self._frames = []
        self._sent_interim = None
        self._pending_interim = None
        return arr

    def reset(self) -> None:
        self._frames = []
        self._sent_interim = None
        self._pending_interim = None
