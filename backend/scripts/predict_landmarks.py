"""CLI: transcribe one landmark file with an exported SignSage deployment bundle.

Usage (from backend/):
    python -m scripts.predict_landmarks --bundle /path/to/signsage_deployment_bundle \\
        --input landmarks.npy [--min-motion 0.002] [--min-confidence 0.0] [--device cpu]

INPUT CONTRACT (the only form the trained model supports):
    A numpy array shaped (T, 543, 3), T >= 2, in MediaPipe Holistic component order
    (pose 33 + face 468 + left hand 21 + right hand 21; x, y, z), RAW -- do not normalise or
    compute deltas yourself. Missing landmarks may be NaN or 0 (training treats both the same).
    Accepted containers: .npy, or .npz holding one array (key "landmarks" if several).

THIS IS NOT A VIDEO MODEL. Raw camera frames must first go through MediaPipe Holistic (as the
SignSage frontend does) to produce the (T, 543, 3) array. This CLI does not do that.

Exit codes: 0 = ran (inspect "accepted"/"reason"), 1 = bundle could not be loaded,
            2 = input file malformed / not the canonical representation.
Output: one JSON object on stdout. The bundle is loaded directly (bypassing the
SIGNSAGE_MODEL_PROMOTED gate) because this is an explicit diagnostic tool.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from app.ml.adapters.sign_to_text_adapter import ModelLoadError, SignToTextAdapter
from app.ml.contracts.inference_contract import InferenceContractError
from app.ml.contracts.landmark_contract import LandmarkContractError
from app.ml.contracts.model_contract import ModelContractError

SCOPE_WARNINGS = {
    "smoke_synthetic": "SMOKE-TEST BUNDLE (synthetic data): not a trained model; output is meaningless.",
    "pilot_subset": "PILOT BUNDLE (trained on a small subset): pipeline validation only; not production quality.",
    "unknown": "Training scope not recorded in this bundle: treat quality as unverified.",
}


def load_landmarks(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"input file not found: {path}")
    try:
        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                names = list(z.files)
                if not names:
                    raise ValueError("npz archive is empty")
                key = "landmarks" if "landmarks" in names else (names[0] if len(names) == 1 else None)
                if key is None:
                    raise ValueError(f"npz has several arrays {names}; name the landmark array 'landmarks'")
                arr = z[key]
        else:
            raise ValueError(f"unsupported input type {path.suffix!r}; use .npy or .npz")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 -- corrupt/pickled files must become a clean contract error
        raise ValueError(f"could not read {path.name}: {type(exc).__name__}") from exc
    return arr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Transcribe a (T,543,3) landmark file with an exported bundle.")
    ap.add_argument("--bundle", required=True, help="path to signsage_deployment_bundle/")
    ap.add_argument("--input", required=True, help=".npy/.npz landmark file")
    ap.add_argument("--min-motion", type=float, default=0.002)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")
    args = ap.parse_args(argv)

    try:
        adapter = SignToTextAdapter.load(args.bundle, device=args.device)
    except (ModelLoadError, LandmarkContractError, ModelContractError, InferenceContractError,
            FileNotFoundError) as exc:
        print(json.dumps({"ok": False, "stage": "load_bundle", "error": str(exc)}))
        return 1

    try:
        landmarks = load_landmarks(Path(args.input))
        result = adapter.transcribe(landmarks, min_motion=args.min_motion, min_confidence=args.min_confidence)
    except ValueError as exc:
        print(json.dumps({"ok": False, "stage": "input", "error": str(exc)}))
        return 2

    print(json.dumps({
        "ok": True,
        "text": result.text,
        "accepted": result.accepted,
        "reason": result.reason,
        "confidence": result.confidence,
        "motion_energy": result.motion_energy,
        "frames": int(len(landmarks)),
        "model_scope": adapter.training_scope,
        "warning": SCOPE_WARNINGS.get(adapter.training_scope),
        "note": "confidence is a decoding-probability proxy, not a calibrated accuracy.",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
