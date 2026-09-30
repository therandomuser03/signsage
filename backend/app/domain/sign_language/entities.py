"""Sign-language domain entities -- pure data, no ML/framework dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LandmarkFrame:
    """One frame of 543 (x, y, z) landmarks, already validated for shape."""
    values: tuple

    def __post_init__(self):
        if len(self.values) != 543:
            raise ValueError(f"LandmarkFrame requires 543 landmarks, got {len(self.values)}")


@dataclass
class LandmarkSequence:
    """An ordered, mutable buffer of frames for one signing attempt."""
    session_id: str
    frames: list = field(default_factory=list)

    def append(self, frame: LandmarkFrame) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)
