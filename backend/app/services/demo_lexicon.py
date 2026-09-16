from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


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
    """DTW matcher for functional demo phrases.

    Prototypes are 60x225 noface features (pose 33 + hands 21+21) as stored in demo_prototypes/*.json
    Each file is {phrase: str, frames: list[list[float]]} with frames as 60x225.
    ponytail: noface ignores collapsed face (468) per ablation finding noface val loss better.
    """

    def __init__(self, prototypes: dict[str, np.ndarray]):
        # phrase -> np[ T, 225]
        self.prototypes = prototypes

    @classmethod
    def load(cls, directory: Path) -> "DemoLexicon":
        protos: dict[str, np.ndarray] = {}
        if not directory.is_dir():
            return cls(protos)
        for p in sorted(directory.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                phrase = str(data.get("phrase", p.stem)).strip()
                frames = np.asarray(data.get("frames"), dtype=np.float32)
                if frames.ndim == 2 and frames.shape[1] == 225:
                    protos[phrase] = frames
                elif frames.ndim == 2 and frames.shape[1] == 1629:
                    # allow 1629 and auto-select noface indices
                    from app.services.pose_preprocessing import NOFACE_FEATURE_INDICES

                    protos[phrase] = frames[:, NOFACE_FEATURE_INDICES]
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
        for phrase, proto in self.prototypes.items():
            d = _dtw_distance(seq_noface.astype(np.float64), proto.astype(np.float64))
            if d < best_dist:
                best_dist = d
                best_phrase = phrase
        if best_dist <= threshold:
            return best_phrase, best_dist
        return None, best_dist

    def inventory(self) -> list[str]:
        return sorted(self.prototypes.keys())
