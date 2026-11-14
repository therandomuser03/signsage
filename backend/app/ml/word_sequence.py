"""Continuous sign stream -> ordered gloss sequence, via windowed voting.

Strategy (works with the data we actually have — signer corpora where each
sign is a ~2-3 s gesture separated by low-motion rest, same cadence as the
recorder takes):

  1. Slide a short window over the whole stream (default 20 frames, stride 5).
  2. Classify each window; drop "rest" and low-confidence windows.
  3. Smooth the label timeline with a small majority vote (+/- 2 windows) to
     suppress single-window flip-flops at sign boundaries.
  4. Collapse consecutive equal labels into one gloss, then prune singleton
     glosses flanked by two OTHER glosses (a singleton between two different
     neighbours is almost always a boundary artefact of the classifier,
     not a real sign).

Per-gloss confidence is the mean confidence of the windows that agreed with
the final gloss. The ordered glosses are what the composer turns into a
sentence.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from app.ml.adapters.sign_to_text_adapter import TranscriptionResult
from app.ml.continuous_sign import RecognizedSign
from app.ml.preprocessing.sequence import compute_motion_energy
from app.ml.word_classifier_adapter import WordClassifierAdapter

WINDOW_FRAMES = 20
WINDOW_STRIDE = 5
VOTE_RADIUS = 2


class WordSequencePipeline:
    """Segment a landmark stream into ordered glosses.

    ``word_continuous`` is set so the translation service knows that
    ``result.text`` is an ordered sequence of glosses to compose into a
    sentence (see app.services.translation_service).
    """

    word_continuous = True

    def __init__(
        self,
        word_adapter: WordClassifierAdapter,
        *,
        window_frames: int = WINDOW_FRAMES,
        stride_frames: int = WINDOW_STRIDE,
        vote_radius: int = VOTE_RADIUS,
        min_motion: float = 0.002,
        min_confidence: float = 0.5,
    ):
        self._word = word_adapter
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.vote_radius = vote_radius
        self.min_motion = min_motion
        self.min_confidence = min_confidence
        self.last_signs: list[RecognizedSign] = []

    @property
    def adapter(self) -> WordClassifierAdapter:
        return self._word

    @property
    def labels(self) -> list[str]:
        return list(self._word.labels)

    @property
    def confidence_threshold(self) -> float:
        return float(self._word.confidence_threshold)

    def _window_preds(
        self, arr: np.ndarray,
    ) -> list[Optional[tuple[str, float]]]:
        """Classify each window; None = rest / uncertain."""
        threshold = max(float(self.min_confidence), float(self._word.confidence_threshold))

        def predict(seg: np.ndarray) -> Optional[tuple[str, float]]:
            result = self._word.classify(seg)
            if result.status != "ok":
                return None
            if result.label == "rest" or result.label not in self._word.labels:
                return None
            if result.confidence < threshold:
                return None
            return (result.label, float(result.confidence))

        preds: list[Optional[tuple[str, float]]] = []
        if len(arr) < self.window_frames:
            return [predict(arr)]
        for start in range(0, len(arr) - self.window_frames + 1, self.stride_frames):
            preds.append(predict(arr[start:start + self.window_frames]))
        return preds

    def _majority_vote(
        self, preds: list[Optional[tuple[str, float]]],
    ) -> list[Optional[tuple[str, float]]]:
        """+/-VOTE_RADIUS majority smoothing on (label, conf); keeps conf of the
        supporting windows inside each final run via a second pass."""
        n = len(preds)
        labels = [p[0] if p else None for p in preds]
        out: list[Optional[tuple[str, float]]] = []
        for i in range(n):
            vote = labels[max(0, i - self.vote_radius):min(n, i + self.vote_radius + 1)]
            support_idx = [
                max(0, i - self.vote_radius) + j
                for j, v in enumerate(vote)
                if v is not None
            ]
            support = [labels[j] for j in support_idx]
            if not support:
                out.append(None)
                continue
            label = Counter(support).most_common(1)[0][0]
            conf = float(np.mean([preds[j][1] for j in support_idx if labels[j] == label]))
            out.append((label, conf))
        return out

    def _sequence_with_conf(
        self, preds: list[Optional[tuple[str, float]]],
    ) -> list[tuple[str, float, int, int, int]]:
        """Collapse runs of equal labels into (gloss, mean_conf, start_i, end_i, n_windows)."""
        runs: list[list] = []
        for i, p in enumerate(preds):
            if p is None:
                continue
            if runs and runs[-1][0] == p[0]:
                runs[-1][1].append(p[1])
                runs[-1][3] = i
                runs[-1][4] += 1
            else:
                runs.append([p[0], [p[1]], i, i, 1])
        return [(r[0], float(np.mean(r[1])), r[2], r[3], r[4]) for r in runs]

    def _prune_singletons(
        self, runs: list[tuple[str, float, int, int, int]],
    ) -> list[tuple[str, float, int, int, int]]:
        """Drop a gloss run with < MIN_SUPPORT windows when both neighbours are
        sufficiently supported and different — a 1-window run between two real
        signs is a boundary artefact, not a sign."""
        MIN_SUPPORT = 3
        changed = True
        seq = list(runs)
        for _ in range(3):
            if not changed:
                break
            changed = False
            pruned = []
            for i, run in enumerate(seq):
                prev_ = seq[i - 1] if i > 0 else None
                next_ = seq[i + 1] if i < len(seq) - 1 else None
                if (
                    prev_ is not None
                    and next_ is not None
                    and run[4] < MIN_SUPPORT
                    and prev_[0] != run[0]
                    and next_[0] != run[0]
                    and prev_[0] != next_[0]
                    and prev_[4] >= MIN_SUPPORT
                    and next_[4] >= MIN_SUPPORT
                ):
                    changed = True
                    continue
                pruned.append(run)
            seq = pruned
        return seq

    def transcribe(
        self,
        landmarks_xyz: np.ndarray,
        min_motion: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> TranscriptionResult:
        arr = np.asarray(landmarks_xyz, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (543, 3):
            raise ValueError(
                f"Expected landmarks shaped (T, 543, 3), received {arr.shape}"
            )
        if len(arr) < 2:
            raise ValueError(f"Need at least two frames, received {len(arr)}")

        motion = compute_motion_energy(arr)
        eff_min_motion = self.min_motion if min_motion is None else min_motion
        if motion < eff_min_motion and len(arr) >= self.window_frames:
            self.last_signs = []
            return TranscriptionResult("", False, "insufficient_motion", motion, 0.0)

        preds = self._window_preds(arr)
        preds = self._majority_vote(preds)
        runs = self._sequence_with_conf(preds)
        runs = self._prune_singletons(runs)

        self.last_signs = [
            RecognizedSign(
                text=gloss,
                confidence=conf,
                reason="ok",
                start_frame=start * self.stride_frames,
                end_frame=min(end * self.stride_frames + self.window_frames, len(arr)),
                frame_index=end * self.stride_frames,
            )
            for gloss, conf, start, end, n_windows in runs
        ]

        text = " ".join(s.text for s in self.last_signs)
        confidence = (
            float(np.mean([s.confidence for s in self.last_signs])) if self.last_signs else 0.0
        )
        accepted = bool(self.last_signs)
        return TranscriptionResult(
            text, accepted, "ok" if accepted else "no_signs_detected", motion, confidence
        )
