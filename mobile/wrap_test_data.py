#!/usr/bin/env python3
"""Wrap the preprocessed test keypoints as a TorchScript module for Android."""
import torch

tensor = torch.load("mobile/android_test/app/src/main/assets/test_data/keypoints.pt")
print(f"Loaded keypoints: {tensor.shape}")

class KeypointWrapper(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.register_buffer("data", data)

    def forward(self, idx: int):
        return self.data[idx:idx+1]  # (1, 4, 94, 128)

wrapper = KeypointWrapper(tensor).eval()
scripted = torch.jit.script(wrapper)
out_path = "mobile/android_test/app/src/main/assets/test_data/keypoints_module.pt"
scripted.save(out_path)
print(f"Saved: {out_path}")
# Also export as .ptl for mobile
from torch.utils.mobile_optimizer import optimize_for_mobile
opt = optimize_for_mobile(scripted)
ptl_path = out_path.with_suffix(".ptl")
opt._save_for_lite_interpreter(str(ptl_path))
print(f"Saved: {ptl_path}")
print(f"Test: load(0) → {scripted(0).shape}")
