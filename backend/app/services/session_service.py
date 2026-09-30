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

    def take_sequence(self) -> np.ndarray:
        """Pop the buffered frames as a (T, 543, 3) array and reset the buffer."""
        arr = np.asarray(self._frames, dtype=np.float32)
        self._frames = []
        return arr

    def reset(self) -> None:
        self._frames = []
