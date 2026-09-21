from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# SignSage landmark contract
# ---------------------------------------------------------------------------
# MediaPipe Holistic:
#   pose       = 33
#   face       = 468
#   left hand  = 21
#   right hand = 21
#
# Total:
#   543 landmarks × XYZ = 1629 features/frame
#
# The canonical training-time normalization is implemented in
# normalize_pose_sequence(). This is the backend source of truth.
# ---------------------------------------------------------------------------

FULL_KEYPOINTS = 543
COORDS_PER_KEYPOINT = 3
FULL_FEATURE_DIM = FULL_KEYPOINTS * COORDS_PER_KEYPOINT


# ---------------------------------------------------------------------------
# Historical no-face / DTW compatibility contract
# ---------------------------------------------------------------------------
# IMPORTANT:
# The shipped DTW/demo path historically selected indices 468:543.
# Because SignSage's actual component order is:
#
#   pose(0:33)
#   face(33:501)
#   left_hand(501:522)
#   right_hand(522:543)
#
# indices 468:543 are NOT pose + both hands. They contain:
#
#   face[435:468] + left_hand + right_hand
#
# Do not silently change this selector without migrating the stored
# demo prototypes at the same time.
FACE_END = 468

NOFACE_KEYPOINTS = list(range(FACE_END, FULL_KEYPOINTS))

NOFACE_FEATURE_INDICES = np.array(
    [
        3 * i + j
        for i in NOFACE_KEYPOINTS
        for j in range(3)
    ],
    dtype=np.int64,
)

NOFACE_FEATURE_DIM = int(
    NOFACE_FEATURE_INDICES.size
)


# ---------------------------------------------------------------------------
# Verified real component boundaries
# ---------------------------------------------------------------------------

TRUE_POSE_LANDMARKS = 33
TRUE_FACE_LANDMARKS = 468
TRUE_HAND_LANDMARKS = 21

TRUE_LEFT_HAND_LANDMARK_START = (
    TRUE_POSE_LANDMARKS + TRUE_FACE_LANDMARKS
)  # 501

TRUE_RIGHT_HAND_LANDMARK_START = (
    TRUE_LEFT_HAND_LANDMARK_START + TRUE_HAND_LANDMARKS
)  # 522

TRUE_HAND_FEATURE_START = (
    TRUE_LEFT_HAND_LANDMARK_START * COORDS_PER_KEYPOINT
)  # 1503

TRUE_HAND_FEATURE_DIM = (
    2 * TRUE_HAND_LANDMARKS * COORDS_PER_KEYPOINT
)  # 126


@dataclass(frozen=True)
class ProcessedPose:
    features: np.ndarray
    true_length: int
    feature_dim: int


class PoseNormalizationStats:
    def __init__(self) -> None:
        self.total_sequences = 0
        self.bad_center_sequences = 0
        self.degenerate_scale_sequences = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "total_sequences": self.total_sequences,
            "bad_center_sequences": self.bad_center_sequences,
            "degenerate_scale_sequences": self.degenerate_scale_sequences,
        }


POSE_NORMALIZATION_STATS = PoseNormalizationStats()


def validate_landmark_array(
    points: np.ndarray,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    """
    Validate and canonicalize landmark input.

    Accepted input:
        [543, 3]       -> one frame
        [T, 543, 3]    -> sequence

    Returns:
        contiguous float32 array shaped [T, 543, 3]
    """
    x = np.asarray(points, dtype=np.float32)

    if x.ndim == 2:
        if x.shape != (
            FULL_KEYPOINTS,
            COORDS_PER_KEYPOINT,
        ):
            raise ValueError(
                "Expected one frame with shape "
                f"(543, 3), received {tuple(x.shape)}."
            )

        x = x[None, ...]

    if x.ndim != 3 or x.shape[1:] != (
        FULL_KEYPOINTS,
        COORDS_PER_KEYPOINT,
    ):
        raise ValueError(
            "Expected [frames, 543, 3], "
            f"received shape={tuple(x.shape)}."
        )

    if not allow_empty and x.shape[0] == 0:
        raise ValueError(
            "Empty landmark sequence."
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Landmarks contain NaN or infinite values."
        )

    return np.ascontiguousarray(
        x,
        dtype=np.float32,
    )


def normalize_pose_sequence(
    points: np.ndarray,
) -> np.ndarray:
    """
    Exact training-time root-center + body-scale normalization.

    Source of truth:
        research/notebooks/signsage.ipynb

    Input:
        [T, 543, 3]

    Output:
        [T, 1629]

    The normalization is intentionally kept here as the canonical backend
    implementation. The browser should send raw MediaPipe landmark
    coordinates rather than applying a second, potentially divergent
    normalization implementation.
    """
    x = validate_landmark_array(points)

    POSE_NORMALIZATION_STATS.total_sequences += 1

    # Hip center from MediaPipe pose landmarks 23 and 24.
    hip = x[:, [23, 24], :]
    center = hip.mean(
        axis=1,
        keepdims=True,
    )

    bad_center = ~np.isfinite(center).all(
        axis=(1, 2)
    )

    if bad_center.any():
        # Matches the notebook's counter semantics:
        # one increment per sequence containing at least one
        # bad-center frame.
        POSE_NORMALIZATION_STATS.bad_center_sequences += 1

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Mean of empty slice",
            )

            center[bad_center] = np.nanmean(
                x[bad_center, :33, :],
                axis=1,
                keepdims=True,
            )

    x = x - center

    # Body scale is computed from the 33 pose landmarks.
    body = x[:, :33, :]

    scale_per_frame = np.nanstd(
        body.reshape(
            body.shape[0],
            -1,
        ),
        axis=1,
    )

    valid_scale = scale_per_frame[
        np.isfinite(scale_per_frame)
        & (scale_per_frame > 1e-6)
    ]

    if valid_scale.size == 0:
        POSE_NORMALIZATION_STATS.degenerate_scale_sequences += 1
        scale = 1.0
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Mean of empty slice",
            )

            scale = np.nanmedian(valid_scale)

        if (
            not np.isfinite(scale)
            or scale <= 1e-6
        ):
            POSE_NORMALIZATION_STATS.degenerate_scale_sequences += 1
            scale = 1.0

    x = x / float(scale)

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return x.reshape(
        x.shape[0],
        FULL_FEATURE_DIM,
    ).astype(np.float32)


def resize_sequence(
    seq: np.ndarray,
    target_len: int,
) -> tuple[np.ndarray, int]:
    """
    Resize a [T, F] sequence to target_len.

    Longer sequences are temporally sampled.
    Shorter sequences are zero-padded.

    Returns:
        resized sequence
        true length before padding/truncation
    """
    if target_len <= 0:
        raise ValueError(
            "target_len must be positive."
        )

    src_len = int(seq.shape[0])

    if src_len <= 0:
        raise ValueError(
            "Empty pose sequence."
        )

    true_len = min(
        src_len,
        target_len,
    )

    if src_len == target_len:
        return (
            seq.astype(np.float32),
            true_len,
        )

    if src_len > target_len:
        idx = np.linspace(
            0,
            src_len - 1,
            target_len,
        )

        return (
            seq[
                np.round(idx).astype(int)
            ].astype(np.float32),
            true_len,
        )

    out = np.zeros(
        (
            target_len,
            seq.shape[1],
        ),
        dtype=np.float32,
    )

    out[:src_len] = seq

    return out, true_len


def reconstruct_full_frame_from_hands(
    hand_features: np.ndarray,
) -> np.ndarray:
    """
    Embed a captured [T, 126] two-hand feature block:

        left hand  = 21 * 3
        right hand = 21 * 3

    into a [T, 1629] full-feature sequence.

    Used only for the bounded English -> ISL demo rendering path.

    Pose and face remain zero because the shipped demo clips contain
    usable hand motion but not complete body-pose information.
    """
    hand_features = np.asarray(
        hand_features,
        dtype=np.float32,
    )

    if (
        hand_features.ndim != 2
        or hand_features.shape[1]
        != TRUE_HAND_FEATURE_DIM
    ):
        raise ValueError(
            f"Expected [T, {TRUE_HAND_FEATURE_DIM}] "
            "two-hand features, "
            f"got {tuple(hand_features.shape)}."
        )

    out = np.zeros(
        (
            hand_features.shape[0],
            FULL_FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    out[
        :,
        TRUE_HAND_FEATURE_START:
        TRUE_HAND_FEATURE_START
        + TRUE_HAND_FEATURE_DIM,
    ] = hand_features

    return out


def prepare_landmarks(
    landmarks: np.ndarray,
    *,
    max_frames: int,
    no_face: bool = False,
    input_mode: str = "raw",
) -> ProcessedPose:
    """
    Prepare a landmark sequence for model inference.

    Parameters:
        landmarks:
            [T, 543, 3] or [543, 3]

        max_frames:
            Model input temporal length.

        no_face:
            Apply the historical 468:543 feature selector.
            Keep this only for checkpoints/prototypes trained with
            that exact selector.

        input_mode:
            "raw":
                canonical training-time normalization is applied here.

            "normalized":
                input is assumed to already be training-compatible
                normalized landmark data.

    Returns:
        ProcessedPose(features=[max_frames, feature_dim], ...)
    """
    if input_mode not in {
        "normalized",
        "raw",
    }:
        raise ValueError(
            "input_mode must be 'normalized' or 'raw'."
        )

    x = validate_landmark_array(
        landmarks
    )

    if input_mode == "raw":
        normalized = normalize_pose_sequence(x)
    else:
        normalized = x.reshape(
            x.shape[0],
            FULL_FEATURE_DIM,
        ).astype(np.float32)

    if no_face:
        normalized = normalized[
            :,
            NOFACE_FEATURE_INDICES,
        ]

    resized, true_len = resize_sequence(
        normalized,
        max_frames,
    )

    return ProcessedPose(
        features=np.ascontiguousarray(
            resized,
            dtype=np.float32,
        ),
        true_length=true_len,
        feature_dim=int(
            resized.shape[1]
        ),
    )
