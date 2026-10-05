"""Landmark layout constants for the Sign->Text pipeline.

These numbers come directly from landmark_contract.json (component_order /
component_ranges), which is itself derived from the real MediaPipe Holistic
pose-file header at training time (signsage.ipynb, Section 5b). Do not
change these without regenerating that contract and re-validating (see
ml/contracts/landmark_contract.py) -- this module never invents ordering,
it just names the constants the trained model already assumes.
"""
from __future__ import annotations

TOTAL_LANDMARKS = 543
COORDINATES_PER_LANDMARK = 3
RAW_FEATURE_DIM = TOTAL_LANDMARKS * COORDINATES_PER_LANDMARK  # 1629

# Canonical component layout: global landmark indices 0..542 are assigned by
# walking COMPONENT_ORDER and concatenating each component's points. Pose
# *world* landmarks are excluded (backend mask covers range(543) only).
# This is the single owner of the ordering -- frontend landmarks.ts,
# the training reader, and the bundle contract must all match these values.
POSE_LANDMARK_COUNT = 33
FACE_LANDMARK_COUNT = 468
HAND_LANDMARK_COUNT = 21
COMPONENT_ORDER = (
    "POSE_LANDMARKS",
    "FACE_LANDMARKS",
    "LEFT_HAND_LANDMARKS",
    "RIGHT_HAND_LANDMARKS",
)
COMPONENT_RANGES = {
    "POSE_LANDMARKS": (0, 33),
    "FACE_LANDMARKS": (33, 501),
    "LEFT_HAND_LANDMARKS": (501, 522),
    "RIGHT_HAND_LANDMARKS": (522, 543),
}
COORDINATE_ORDER = ("x", "y", "z")

# The model's FrameProjection concatenates each frame with its own
# frame-to-frame delta before its first linear layer -- see
# ml/models/sign_to_text.py:FrameProjection. Preprocessing (this package)
# must NOT precompute that delta; it ships plain normalized 1629-D frames
# and the model produces the 3258-D projection input internally.
MODEL_INPUT_DIM = RAW_FEATURE_DIM * 2  # 3258

# MediaPipe pose landmarks: global index 23 = left hip, 24 = right hip.
# normalize_pose_sequence() root-centers on the midpoint of these two points.
HIP_LANDMARK_INDICES = (23, 24)
BODY_LANDMARK_COUNT = 33  # first 33 global indices = the "pose" component

# Bump this whenever normalize_pose_sequence()'s algorithm changes.
# ml/contracts/model_contract.py asserts the deployment manifest's declared
# normalization_version equals this constant, so a stale bundle paired with
# new preprocessing code (or vice versa) is caught at load time.
NORMALIZATION_VERSION = "pose_normalize_v1"


def assert_landmark_shape(arr) -> None:
    """Raise ValueError if `arr` is not an ndarray shaped (T, 543, 3).

    Fails loudly on non-array input (no AttributeError) so a caller passing
    a silently incompatible representation gets a clear contract error.
    """
    import numpy as np

    if not isinstance(arr, np.ndarray):
        raise ValueError(
            f"Expected landmarks as numpy.ndarray shaped "
            f"(T, {TOTAL_LANDMARKS}, {COORDINATES_PER_LANDMARK}), "
            f"received {type(arr).__name__}"
        )
    if arr.ndim != 3 or arr.shape[1:] != (TOTAL_LANDMARKS, COORDINATES_PER_LANDMARK):
        raise ValueError(
            f"Expected landmarks shaped (T, {TOTAL_LANDMARKS}, {COORDINATES_PER_LANDMARK}), "
            f"received {arr.shape}"
        )


def assert_landmark_dtype(arr) -> None:
    """Raise ValueError if `arr` has a non-numeric dtype.

    Integer and floating dtypes are accepted (JSON transport yields ints for
    whole numbers; float32 conversion happens downstream). Bool, string,
    and object dtypes are rejected loudly instead of being silently cast --
    a bool/object array here means the representation is incompatible.
    """
    import numpy as np

    if not isinstance(arr, np.ndarray):
        raise ValueError(
            f"Expected landmarks as numpy.ndarray, received {type(arr).__name__}"
        )
    if arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(
            f"Expected landmark dtype integer or floating, received {arr.dtype} "
            f"(incompatible representation -- refusing to cast)"
        )
