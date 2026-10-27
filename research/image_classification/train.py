"""SignSage lightweight ISL image classifier training.

Trains two small CNNs from scratch (no pretrained weights, no torchvision):

1. Alphabet classifier -- 26 classes (A-Z) on RealSign data.
2. Word classifier -- 20 classes on Real-life Words data.

CPU-friendly: small batches, 64x64 RGB input, early stopping on val loss.

Artifacts (under research/image_classification/artifacts/<task>/):
  model.pt              -- {'state_dict': ..., 'model_config': {...}}
  labels.json           -- {label_to_index, index_to_label, classes, ...}
  evaluation.json       -- test metrics, confusion matrix, per-class metrics
  training_history.json -- per-epoch train/val loss & accuracy

No large checkpoints or datasets are committed to Git.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "research" / "image_classification" / "artifacts"
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
IMG_SIZE = 64


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    """Compact CNN: 3 conv blocks + classifier head. CPU-friendly."""

    def __init__(self, n_classes: int, img_size: int = IMG_SIZE, n_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        # After 3 pools: img_size / 8. For 64 -> 8.
        feature_map = img_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feature_map * feature_map, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class NpzImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images).float() / 255.0  # [0,1]
        self.images = self.images.permute(0, 3, 1, 2)  # (N,H,W,C) -> (N,C,H,W)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["images"], d["labels"]


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes: int) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == y).sum().item()
        total_n += x.size(0)
        for t, p in zip(y.tolist(), preds.tolist()):
            confusion[t, p] += 1

    acc = total_correct / total_n
    avg_loss = total_loss / total_n

    # Per-class precision, recall, F1
    per_class = {}
    for c in range(n_classes):
        tp = int(confusion[c, c])
        fn = int(confusion[c, :].sum()) - tp
        fp = int(confusion[:, c].sum()) - tp
        support = int(confusion[c, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[str(c)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    # Macro-F1
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    macro_precision = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()]))

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "n_samples": total_n,
    }


def train_task(task: str, classes: list, n_classes: int,
               train_path: Path, val_path: Path, test_path: Path,
               out_dir: Path) -> dict:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training {task} classifier ({n_classes} classes) on {device} ===")

    train_images, train_labels = load_npz(train_path)
    val_images, val_labels = load_npz(val_path)
    test_images, test_labels = load_npz(test_path)

    train_ds = NpzImageDataset(train_images, train_labels)
    val_ds = NpzImageDataset(val_images, val_labels)
    test_ds = NpzImageDataset(test_images, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    model = SmallCNN(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, n_classes)
        dt = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": dt,
        })

        print(f"  epoch {epoch:2d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_f1={val_metrics['macro_f1']:.4f} ({dt:.1f}s)")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    # Load best and evaluate on test
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device, n_classes)
    print(f"\n  TEST: loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}")

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "n_classes": n_classes,
        "img_size": IMG_SIZE,
        "n_channels": 3,
        "architecture": "SmallCNN",
        "classes": classes,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "epochs_trained": len(history),
        "early_stopping_patience": PATIENCE,
        "learning_rate": LR,
    }
    torch.save({
        "state_dict": best_state,
        "model_config": model_config,
    }, out_dir / "model.pt")

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    evaluation = {
        "task": task,
        "n_classes": n_classes,
        "classes": classes,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "n_test_samples": test_metrics["n_samples"],
        "per_class": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "best_val_loss": best_val_loss,
        "device": str(device),
    }
    (out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))

    return evaluation


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Alphabet
    alpha_labels = json.loads((ARTIFACT_DIR / "alphabet" / "labels.json").read_text())
    alpha_classes = alpha_labels["classes"]
    e1 = train_task(
        task="alphabet", classes=alpha_classes, n_classes=len(alpha_classes),
        train_path=ARTIFACT_DIR / "alphabet" / "train.npz",
        val_path=ARTIFACT_DIR / "alphabet" / "val.npz",
        test_path=ARTIFACT_DIR / "alphabet" / "test.npz",
        out_dir=ARTIFACT_DIR / "alphabet",
    )

    # Words
    word_labels = json.loads((ARTIFACT_DIR / "words" / "labels.json").read_text())
    word_classes = word_labels["classes"]
    e2 = train_task(
        task="word", classes=word_classes, n_classes=len(word_classes),
        train_path=ARTIFACT_DIR / "words" / "train.npz",
        val_path=ARTIFACT_DIR / "words" / "val.npz",
        test_path=ARTIFACT_DIR / "words" / "test.npz",
        out_dir=ARTIFACT_DIR / "words",
    )

    print("\n=== Training complete ===")
    print(f"Alphabet: acc={e1['test_accuracy']:.4f} macro_f1={e1['test_macro_f1']:.4f}")
    print(f"Words:    acc={e2['test_accuracy']:.4f} macro_f1={e2['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
