"""CLI: verify the configured deployment bundle loads and behaves correctly.

Usage:
    python -m scripts.model_check

Loads the bundle pointed at by SIGNSAGE_MODEL_BUNDLE_DIR (bypassing the
model_promoted gate, since this is explicitly a diagnostic tool) and runs
the same activity-gate self-test the training notebook runs on export
(Section 15a): an all-zero landmark sequence must be REJECTED, not
transcribed with false confidence.
"""
from __future__ import annotations

import sys

import numpy as np

from app.core.config import get_settings
from app.ml.models.registry import get_sign_to_text_pipeline


def main() -> int:
    settings = get_settings()
    try:
        pipeline = get_sign_to_text_pipeline(settings, allow_unpromoted=True)
    except Exception as exc:
        print(f"FAILED to load model bundle: {exc}")
        return 1

    print(
        f"Loaded bundle OK. max_frames={pipeline.contract.model.max_frames}, "
        f"vocab_size={pipeline.contract.model.vocab_size}, "
        f"model_promoted={settings.model_promoted}"
    )

    static_landmarks = np.zeros((10, 543, 3), dtype=np.float32)
    result = pipeline.transcribe(static_landmarks)
    if result.accepted:
        print("FAILED: static/no-motion landmarks were accepted as a translation.")
        return 1
    print(f"Activity gate self-test passed (reason={result.reason!r}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
