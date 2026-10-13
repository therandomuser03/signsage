"""Heuristic ISL-fallback recognizer: a tiny, explicitly limited sign classifier.

Scope: 5 static one-hand shapes voted over a landmark sequence. This is NOT
general ISL translation -- it exists so the live webcam path can demonstrate
real landmark-derived recognition while the transformer remains NO-GO (see
backend/models/README.md and MODEL_DEPLOYMENT_DIAGNOSIS.md). The transformer
gate is untouched: this pipeline never loads a checkpoint and is always
reported as ``heuristic-fallback`` via /capabilities, never as the model.

Canonical input: (T, 543, 3) float array in COMPONENT_ORDER
(POSE 0:33, FACE 33:501, LEFT_HAND 501:522, RIGHT_HAND 522:543).
Missing hands are all-zero (frontend landmarks.ts sends zeros).
"""
from __future__ import annotations

import numpy as np

from app.ml.preprocessing.landmarks import (
    COMPONENT_RANGES,
    assert_landmark_dtype,
    assert_landmark_shape,
)

# MediaPipe hand topology (local indices within one 21-point hand).
_WRIST = 0
_THUMB = (2, 3, 4)  # (mcp, ip, tip)
_FINGERS = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),  # (mcp, pip, tip)
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

#: Gloss ID -> English word. All IDs are in the avatar vocabulary, so a
#: recognized sign can be played back on the Text-to-Sign page (round-trip).
GLOSS_TEXT = {
    "HELLO": "Hello",
    "YES": "Yes",
    "NO": "No",
    "YOU": "You",
    "HELP": "Help",
}

FALLBACK_VOCABULARY = tuple(GLOSS_TEXT)

#: Gloss ID -> how to perform it for the demo hint in the UI.
GLOSS_HINTS = {
    "HELLO": "Open palm, fingers spread, facing the camera",
    "YES": "Thumbs up (only the thumb extended)",
    "NO": "Closed fist",
    "YOU": "Point with the index finger only",
    "HELP": "Index + middle fingers only (peace shape)",
}

_MIN_SUPPORT_FRAMES = 10
_MIN_AGREEMENT = 0.6


def _hand_present(hand: np.ndarray) -> bool:
    """A hand counts as present when most of its 21 points are non-zero."""
    norms = np.linalg.norm(hand, axis=-1)
    return bool((norms > 1e-6).sum() >= 11)


def _extended_fingers(hand: np.ndarray) -> frozenset[str]:
    """Which fingers are extended: tip farther from wrist than base joint."""
    wrist = hand[_WRIST]
    extended = set()
    for name, (base, joint, tip) in _FINGERS.items():
        d_tip = np.linalg.norm(hand[tip] - wrist)
        d_joint = np.linalg.norm(hand[joint] - wrist)
        if d_tip > d_joint * 1.15:
            extended.add(name)
    return frozenset(extended)


def classify_hand_shape(hand: np.ndarray) -> str | None:
    """Single-frame hand shape -> gloss ID (right hand preferred by caller)."""
    if not _hand_present(hand):
        return None
    ext = _extended_fingers(hand)
    if ext == frozenset({"thumb", "index", "middle", "ring", "pinky"}):
        return "HELLO"
    if len(ext) == 0:
        return "NO"
    if ext == frozenset({"thumb"}):
        return "YES"
    if ext == frozenset({"index"}):
        return "YOU"
    if ext == frozenset({"index", "middle"}):
        return "HELP"
    return None


def classify_frame(frame_543: np.ndarray) -> str | None:
    """One (543, 3) frame -> gloss ID, or None when unrecognized.

    Right hand first; a present-but-unclassifiable right hand must not veto
    a supported shape on the left (fall through instead of returning None).
    """
    l_start, l_end = COMPONENT_RANGES["LEFT_HAND_LANDMARKS"]
    r_start, r_end = COMPONENT_RANGES["RIGHT_HAND_LANDMARKS"]
    label = classify_hand_shape(frame_543[r_start:r_end])
    if label is None:
        label = classify_hand_shape(frame_543[l_start:l_end])
    return label


def recognize_sequence(landmarks_xyz: np.ndarray) -> tuple[str | None, float, str]:
    """Majority vote over frames. Returns (gloss_id|None, agreement, reason)."""
    arr = np.asarray(landmarks_xyz)
    assert_landmark_dtype(arr)
    assert_landmark_shape(arr)
    if len(arr) < 2:
        raise ValueError(f"Incompatible representation: need at least 2 frames, received {len(arr)}")

    votes: dict[str, int] = {}
    recognized = 0
    for frame in arr:
        label = classify_frame(frame)
        if label is not None:
            votes[label] = votes.get(label, 0) + 1
            recognized += 1

    if recognized < _MIN_SUPPORT_FRAMES:
        return None, 0.0, "no_hand_detected"
    best = max(votes, key=votes.__getitem__)
    agreement = votes[best] / recognized
    if agreement < _MIN_AGREEMENT:
        return None, float(agreement), "unrecognized_shape"
    return best, float(agreement), "heuristic-majority"


class HeuristicSignPipeline:
    """Duck-compatible with SignToTextAdapter.transcribe() so
    TranslationService can wrap either without changes."""

    def transcribe(self, landmarks_xyz, min_motion: float = 0.0, min_confidence: float = 0.0):
        from app.ml.adapters.sign_to_text_adapter import TranscriptionResult
        from app.ml.preprocessing.sequence import compute_motion_energy

        motion = compute_motion_energy(landmarks_xyz)
        gloss, agreement, reason = recognize_sequence(landmarks_xyz)
        floor = max(min_confidence, 0.0)
        # Static poses are valid signs: acceptance needs shape agreement, not
        # motion (min_motion is accepted for interface compatibility only).
        if gloss is None or agreement < floor:
            return TranscriptionResult(
                text="", accepted=False, reason=reason,
                motion_energy=motion, confidence=float(agreement),
            )
        return TranscriptionResult(
            text=GLOSS_TEXT[gloss], accepted=True, reason=reason,
            motion_energy=motion, confidence=float(agreement),
        )
