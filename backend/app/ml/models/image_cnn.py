"""Small CNN for ISL image classification.

Architecture must match research/image_classification/train.py:SmallCNN exactly:
  3 conv blocks (32->64->128 filters, 3x3 conv, ReLU, 2x2 maxpool)
  flatten -> Linear(128*8*8, 256) -> ReLU -> Dropout(0.5) -> Linear(256, n_classes)

Input: (N, 3, img_size, img_size) float32 in [0, 1], RGB.
Output: (N, n_classes) logits.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from app.ml.device import resolve_torch_device


class SmallImageCNN(nn.Module):
    def __init__(self, n_classes: int, img_size: int = 64, n_channels: int = 3):
        super().__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        self.n_channels = n_channels
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feature_map = img_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feature_map * feature_map, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @classmethod
    def load_from_bundle(
        cls,
        bundle_dir,
        device: Optional[str] = None,
    ):
        """Load model + labels from a bundle dir containing model.pt and labels.json.

        Bundle layout (produced by research/image_classification/train.py):
          <bundle_dir>/
            model.pt              -- {'state_dict': ..., 'model_config': {...}}
            labels.json           -- {label_to_index, index_to_label, classes, ...}
        """
        import json

        from pathlib import Path

        bundle_dir = Path(bundle_dir)
        model_path = bundle_dir / "model.pt"
        labels_path = bundle_dir / "labels.json"

        if not model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        torch_device = resolve_torch_device(device)
        payload = torch.load(model_path, map_location=torch_device)
        if "state_dict" not in payload or "model_config" not in payload:
            raise ValueError("Checkpoint is missing 'state_dict'/'model_config'.")

        mc = payload["model_config"]
        n_classes = int(mc["n_classes"])
        img_size = int(mc.get("img_size", 64))
        n_channels = int(mc.get("n_channels", 3))

        model = cls(n_classes=n_classes, img_size=img_size, n_channels=n_channels)
        model.load_state_dict(payload["state_dict"])
        model.eval()

        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        label_to_index = labels["label_to_index"]
        index_to_label = {int(v): k for k, v in label_to_index.items()}
        classes = labels["classes"]

        torch_device = resolve_torch_device(device)
        model.to(torch_device)

        return cls._BundleModel(model=model, device=torch_device,
                                label_to_index=label_to_index,
                                index_to_label=index_to_label,
                                classes=classes,
                                model_config=mc,
                                img_size=img_size)

    class _BundleModel:
        def __init__(self, model, device, label_to_index, index_to_label,
                     classes, model_config, img_size):
            self.model = model
            self.device = device
            self.label_to_index = label_to_index
            self.index_to_label = index_to_label
            self.classes = classes
            self.model_config = model_config
            self.img_size = img_size
            self.n_classes = len(classes)

        def predict(self, image_tensor: torch.Tensor):
            """image_tensor: (1, 3, img_size, img_size) float [0,1].
            Returns (label, confidence, index, logits)."""
            with torch.no_grad():
                logits = self.model(image_tensor.to(self.device))
                probs = torch.softmax(logits, dim=-1)[0]
                conf, idx = probs.max(dim=0)
            label = self.index_to_label[int(idx.item())]
            return label, float(conf.item()), int(idx.item()), logits.cpu()
