#!/usr/bin/env python3
"""
Prepare the 57 intersection test samples for Android testing.

Outputs:
  mobile/android_test/app/src/main/assets/test_data/
    keypoints.pt   — torch tensor (57, 4, 94, 128)
    references.txt — one reference per line (lowercased)
    vocab.json     — T5 tokenizer vocab for decoding
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from evaluate_subsety_intersection import IntersectionDataset

KP_DIR = Path("/home/shoaib/phd/Datasets/iSign/misc/iSign_Keypoints")
ASSETS = Path(__file__).parent / "android_test/app/src/main/assets/test_data"
ASSETS.mkdir(parents=True, exist_ok=True)

ds = IntersectionDataset(
    str(Path(__file__).parent.parent / "data/pose_stitch_split/isign_test_pose_stitch.csv"),
    str(Path(__file__).parent.parent / "data/subset_y/isign_subset_y_test.csv"),
)

all_kp = []
all_refs = []
for i in range(len(ds)):
    item = ds[i]
    kp = item["keypoints"]  # (4, 94, 128) already preprocessed
    all_kp.append(kp)
    all_refs.append(item["text"].lower())

tensor = torch.stack(all_kp)  # (57, 4, 94, 128)
torch.save(tensor, str(ASSETS / "keypoints.pt"))
print(f"Saved keypoints: {tensor.shape}")

with open(ASSETS / "references.txt", "w") as f:
    for ref in all_refs:
        f.write(ref + "\n")
print(f"Saved {len(all_refs)} references")

# Save a minimal T5 vocab (just for decoding token IDs → text)
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
vocab = {id: token for token, id in tokenizer.get_vocab().items()}
# Special tokens
vocab[tokenizer.pad_token_id] = "<pad>"
vocab[tokenizer.eos_token_id] = "</s>"
# Sort by ID for cleaner JSON
vocab_sorted = {str(k): v for k, v in sorted(vocab.items())}
with open(ASSETS / "vocab.json", "w") as f:
    json.dump(vocab_sorted, f)
print(f"Saved vocab: {len(vocab_sorted)} tokens")
print(f"Done. Files in {ASSETS}")
