from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


# Small alias table mapping common rephrasings to one of the eight supported
# demo phrases. Keys/values are compared after `_slug()` normalization, so
# case and punctuation don't matter. This is a bounded, hand-authored lookup
# -- not general English understanding -- matching the "bounded phrase
# lookup, not arbitrary generation" scope for English->ISL (see PLAN.md).
PHRASE_TEXT_ALIASES: dict[str, str] = {
    "hi": "hello",
    "hey": "hello",
    "hiya": "hello",
    "namaste": "hello",
    "thanks": "thank you",
    "thank u": "thank you",
    "thankyou": "thank you",
    "ty": "thank you",
    "yeah": "yes",
    "yep": "yes",
    "yup": "yes",
    "nope": "no",
    "nah": "no",
    "pls": "please",
    "plz": "please",
    "sry": "sorry",
    "im sorry": "sorry",
    "apologies": "sorry",
    "gm": "good morning",
    "goodmorning": "good morning",
    "howre you": "how are you",
    "how are you doing": "how are you",
    "hows it going": "how are you",
    "need help": "i need help",
    "i need some help": "i need help",
    "help me": "help",
}

SUPPORTED_DEMO_PHRASES = frozenset({
    "Hello",
    "Thank you",
    "Yes",
    "No",
    "Please",
    "Sorry",
    "Help",
    "Good morning",
})


def _slug(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used to compare
    free-form input text against the fixed demo-phrase inventory."""
    normalized = re.sub(r"[^a-z0-9\s]", "", str(text).strip().lower())
    return re.sub(r"\s+", " ", normalized).strip()


def match_demo_phrase(text: str, available_phrases: list[str]) -> str | None:
    """Map free-form English text to one of the bounded demo phrases actually
    loaded from demo_prototypes/, if it matches exactly or via the small
    alias table above. Returns the phrase exactly as stored (e.g. "Hello"),
    or None if there's no confident bounded match -- callers should fall
    back to the general (unreliable) generative model in that case, not
    guess.
    """
    normalized_input = _slug(text)
    if not normalized_input:
        return None
    candidate = PHRASE_TEXT_ALIASES.get(normalized_input, normalized_input)
    lookup = {_slug(phrase): phrase for phrase in available_phrases}
    return lookup.get(candidate)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean DTW distance for 2D sequences a[Na,D], b[Nb,D] using L2 per frame."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    # ponytail: O(n*m) DP with float32, n,m <=60 so cheap; no fastdtw dep
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / (n + m))


class DemoLexicon:
    """DTW matcher for controlled live sign recognition.

    Prototypes are 60x225 noface features (pose 33 + hands 21+21) as stored in demo_prototypes/*.json
    Each file is {phrase: str, frames: list[list[float]]} with frames as 60x225.
    ponytail: noface ignores collapsed face (468) per ablation finding noface val loss better.
    """

    def __init__(self, prototypes: dict[str, list[np.ndarray]]):
        # phrase -> list[np[ T, 225]]
        self.prototypes = prototypes

    @classmethod
    def load(cls, directory: Path) -> "DemoLexicon":
        protos: dict[str, list[np.ndarray]] = {}
        if not directory.is_dir():
            return cls(protos)
        for p in sorted(directory.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                phrase = str(data.get("phrase", p.stem)).strip()
                if phrase not in SUPPORTED_DEMO_PHRASES:
                    continue
                frames = np.asarray(data.get("frames"), dtype=np.float32)
                if frames.ndim == 2 and frames.shape[1] == 225:
                    protos.setdefault(phrase, []).append(frames)
                elif frames.ndim == 2 and frames.shape[1] == 1629:
                    # allow 1629 and auto-select noface indices
                    from app.services.pose_preprocessing import NOFACE_FEATURE_INDICES

                    protos.setdefault(phrase, []).append(frames[:, NOFACE_FEATURE_INDICES])
            except Exception:
                continue
        return cls(protos)

    def is_available(self) -> bool:
        return len(self.prototypes) > 0

    def match(self, seq_noface: np.ndarray, threshold: float = 420.0) -> tuple[str | None, float]:
        """Return (phrase, distance) best match under threshold, else (None, best_dist)."""
        if not self.is_available() or seq_noface.size == 0:
            return None, float("inf")
        # ensure 2D [T,225]
        if seq_noface.ndim == 1:
            seq_noface = seq_noface.reshape(1, -1)
        if seq_noface.shape[1] != 225:
            # try to slice if 1629
            if seq_noface.shape[1] == 1629:
                from app.services.pose_preprocessing import NOFACE_FEATURE_INDICES

                seq_noface = seq_noface[:, NOFACE_FEATURE_INDICES]
            else:
                return None, float("inf")
        best_phrase = None
        best_dist = float("inf")
        query = seq_noface.astype(np.float64)
        for phrase, templates in self.prototypes.items():
            for proto in templates:
                d = _dtw_distance(query, proto.astype(np.float64))
                if d < best_dist:
                    best_dist = d
                    best_phrase = phrase
        if best_dist <= threshold:
            return best_phrase, best_dist
        return None, best_dist

    def match_raw(
        self,
        landmarks: np.ndarray,
        *,
        threshold: float,
        min_frames: int = 8,
    ) -> tuple[str | None, float, float, str | None]:
        """Match raw [T,543,3] MediaPipe landmarks against templates.

        Match score is a normalized threshold distance, not model accuracy.
        """
        if not self.is_available():
            return None, float("inf"), 0.0, "templates_unavailable"

        from app.services.pose_preprocessing import (
            NOFACE_FEATURE_INDICES,
            normalize_pose_sequence,
            validate_landmark_array,
        )

        try:
            sequence = validate_landmark_array(landmarks)
        except ValueError:
            return None, float("inf"), 0.0, "invalid_landmarks"

        if sequence.shape[0] < min_frames:
            return None, float("inf"), 0.0, "too_short"

        normalized = normalize_pose_sequence(sequence)
        demo_features = normalized[:, NOFACE_FEATURE_INDICES]
        phrase, distance = self.match(demo_features, threshold=threshold)
        confidence = max(0.0, min(1.0, 1.0 - (distance / threshold))) if np.isfinite(distance) else 0.0
        return (
            phrase,
            distance,
            confidence,
            None if phrase is not None else "below_threshold",
        )

    def inventory(self) -> list[str]:
        return sorted(self.prototypes.keys())

    def get_render_sequence(self, phrase: str) -> np.ndarray | None:
        """Return a [T, 1629] full-feature sequence for skeleton playback of
        `phrase`, or None if it isn't loaded.

        Only the two-hand landmarks (the last 126 of the stored 225 noface
        features) are real captured motion -- see AUDIT_NOTES.md for why the
        remaining 99 stored values are not usable pose data. Those real hand
        landmarks are embedded at their correct position in an otherwise-zero
        1629-length frame; the frontend renders this as a hands-only
        skeleton for bounded demo phrases rather than a (fabricated) full
        body pose.
        """
        templates = self.prototypes.get(phrase)
        if not templates:
            return None
        from app.services.pose_preprocessing import (
            TRUE_HAND_FEATURE_DIM,
            reconstruct_full_frame_from_hands,
        )

        hand_features = templates[0][:, -TRUE_HAND_FEATURE_DIM:]
        return reconstruct_full_frame_from_hands(hand_features)
