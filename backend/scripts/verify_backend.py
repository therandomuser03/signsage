from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `python scripts/verify_backend.py` to work from any working directory,
# without requiring the caller to set PYTHONPATH or use `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.core.config import Settings
from app.services.model_manager import ModelManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SignSage backend artifacts and inference.")
    parser.add_argument("--raw", action="store_true", help="Treat generated landmarks as raw coordinates.")
    args = parser.parse_args()

    settings = Settings(
        landmark_input_mode="raw" if args.raw else "normalized",
    )

    print(f"Model directory: {settings.resolved_model_dir}")
    manager = ModelManager.load(settings)
    print(f"Device: {manager.device}")

    rng = np.random.default_rng(42)
    raw = rng.normal(size=(12, 543, 3)).astype(np.float32)

    if args.raw:
        landmarks = raw
    else:
        # Produce deterministic training-compatible input without importing
        # notebook code: this is only a smoke fixture, not a real sign sample.
        from app.services.pose_preprocessing import normalize_pose_sequence
        landmarks = normalize_pose_sequence(raw).reshape(12, 543, 3)

    started = time.perf_counter()
    text, frames_used, latency = manager.predict_sign_to_text(landmarks)
    print(f"Sign->Text: {text!r}")
    print(f"Frames used: {frames_used}; model latency: {latency:.2f} ms")
    print(f"Wall time: {(time.perf_counter() - started) * 1000:.2f} ms")

    gloss, poses, latency, provider, is_fallback = manager.text_to_pose.predict(
        "hello how are you"
    )
    print(f"Text->Pose gloss: {gloss!r} (provider={provider}, demo_fallback={is_fallback})")
    print(f"Pose shape: {poses.shape}; latency: {latency:.2f} ms")

    gloss2, poses2, latency2, provider2, is_fallback2 = manager.text_to_pose.predict("hello")
    print(f"Text->Pose gloss: {gloss2!r} (provider={provider2}, demo_fallback={is_fallback2})")
    print(f"Pose shape: {poses2.shape}; latency: {latency2:.2f} ms")
    assert is_fallback2, "Expected 'hello' to hit the bounded demo phrase lexicon."

    print("Backend artifact/inference smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
