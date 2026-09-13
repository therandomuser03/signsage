from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Allow `python scripts/benchmark.py` to work from any working directory,
# without requiring the caller to set PYTHONPATH or use `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.core.config import Settings
from app.services.model_manager import ModelManager
from app.services.pose_preprocessing import normalize_pose_sequence


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    settings = Settings(landmark_input_mode="normalized")
    manager = ModelManager.load(settings)

    rng = np.random.default_rng(123)
    raw = rng.normal(size=(80, 543, 3)).astype(np.float32)
    normalized = normalize_pose_sequence(raw).reshape(80, 543, 3)

    s2t = []
    t2p = []

    # Warm-up.
    manager.predict_sign_to_text(normalized)
    manager.text_to_pose.predict("hello how are you")

    for _ in range(args.runs):
        t0 = time.perf_counter()
        manager.predict_sign_to_text(normalized)
        s2t.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        manager.text_to_pose.predict("hello how are you")
        t2p.append((time.perf_counter() - t0) * 1000)

    def report(name, values):
        print(
            f"{name}: mean={statistics.mean(values):.2f} ms "
            f"median={statistics.median(values):.2f} ms "
            f"p95={percentile(values, 95):.2f} ms"
        )

    report("Sign->Text", s2t)
    report("Text->Pose", t2p)
    print("These are backend/model timings only; they are not end-to-end browser latency.")


if __name__ == "__main__":
    main()
