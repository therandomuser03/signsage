from __future__ import annotations
import numpy as np

from typing import Optional, Tuple, Protocol


class ConfidenceProvider(Protocol):
    def get_confidence(self, landmark_index: int, component_name: str) -> float:
        ...


class FallbackConfidenceProvider:
    def get_confidence(self, landmark_index: int, component_name: str) -> float:
        return 1.0


POSE_START, POSE_END = 0, 33
LEFT_HAND_START, LEFT_HAND_END = 501, 522
RIGHT_HAND_START, RIGHT_HAND_END = 522, 543

SPATIAL_DIM = 300
MOTION_DIM = 600


class SignST5Adapter:
    def __init__(self, confidence_provider=None, use_motion=False, max_frames=500):
        self.confidence_provider = confidence_provider or FallbackConfidenceProvider()
        self.use_motion = use_motion
        self.max_frames = max_frames

    def convert(self, landmarks_543):
        if not isinstance(landmarks_543, np.ndarray):
            raise TypeError("Expected numpy.ndarray")
        if landmarks_543.ndim != 3 or landmarks_543.shape[1] != 543 or landmarks_543.shape[2] != 3:
            raise ValueError("Expected (T, 543, 3)")
        T = landmarks_543.shape[0]

        pose_33 = landmarks_543[:, POSE_START:POSE_END, :]
        left_21 = landmarks_543[:, LEFT_HAND_START:LEFT_HAND_END, :]
        right_21 = landmarks_543[:, RIGHT_HAND_START:RIGHT_HAND_END, :]

        def frame_to_4values(pose_33, left_21, right_21, t):
            p = pose_33[t].reshape(-1)
            l = left_21[t].reshape(-1)
            r = right_21[t].reshape(-1)

            def add_conf(values_flat, count):
                c = self.confidence_provider.get_confidence(0, "POSE")
                reshaped = values_flat.reshape(count, 3)
                conf_col = np.full((count, 1), c, dtype=np.float32)
                return np.concatenate([reshaped, conf_col], axis=1).reshape(-1)

            pose_conf = add_conf(p, 33)
            left_conf = add_conf(l, 21)
            right_conf = add_conf(r, 21)
            return np.concatenate([pose_conf, left_conf, right_conf])

        combined = np.stack([frame_to_4values(pose_33, left_21, right_21, t)
                             for t in range(T)], axis=0)

        # Z-score normalization (matching external exactly)
        mean = np.nanmean(combined, axis=0, keepdims=True)
        std = np.nanstd(combined, axis=0, keepdims=True)
        combined = (combined - mean) / np.where(std == 0, 1.0, std)
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.use_motion:
            velocity = np.zeros_like(combined, dtype=np.float32)
            velocity[1:] = combined[1:] - combined[:-1]
            combined = np.concatenate([combined, velocity], axis=1).astype(np.float32)

        T = combined.shape[0]
        if T >= self.max_frames:
            padded = combined[:self.max_frames, :].copy()
            actual_length = self.max_frames
        else:
            padded = np.zeros((self.max_frames, combined.shape[1]), dtype=np.float32)
            padded[:T, :] = combined.astype(np.float32)
            actual_length = T

        return padded, actual_length


def signage_to_external(landmarks_543, *, use_motion=False, max_frames=500):
    adapter = SignST5Adapter(use_motion=use_motion, max_frames=max_frames)
    return adapter.convert(landmarks_543)


if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    one_frame = np.random.rand(1, 543, 3).astype(np.float32)
    t_sp, l_sp = signage_to_external(one_frame, use_motion=False)
    print(f"Spatial: shape={t_sp.shape}, length={l_sp}")
    t_mot, l_mot = signage_to_external(one_frame, use_motion=True)
    print(f"Motion: shape={t_mot.shape}, length={l_mot}")
    print("Smoke test passed")
