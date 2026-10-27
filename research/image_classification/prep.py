"""SignSage image dataset preparation.

Prepares two lightweight ISL image-classification datasets:

1. Alphabet classifier -- RealSign-Indian-Sign-Language-Dataset (CC0-1.0)
   26 classes (A-Z), pre-split Training/Validation/Testing.
2. Word classifier -- Indian Sign Language Real-life Words (Mendeley)
   20 word classes, signer-aware (User_N) split for signer-independent eval.

Outputs (under research/image_classification/artifacts/):
  artifacts/alphabet/{train,val,test}.npz   -- {'images': (N,64,64,3) uint8, 'labels': (N,) int64}
  artifacts/words/{train,val,test}.npz      -- same layout
  artifacts/alphabet/labels.json            -- {index: "A", ...} plus source meta
  artifacts/words/labels.json
  artifacts/manifest.json                   -- dataset provenance, counts, checksums

No external image datasets or large checkpoints are committed to Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "research" / "image_classification" / "artifacts"
ALPHABET_SRC = Path("/tmp/opencode/dataset-audit/ds3/extracted")  # RealSign extracted
WORDS_SRC_ZIP_DIR = REPO_ROOT / "Indian sign Language-Real-life Words.zip"  # outer zip
WORDS_EXTRACT_DIR = Path("/tmp/opencode/dataset-audit/ds2/extracted")

IMG_SIZE = 64
N_CHANNELS = 3
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Image I / O helpers
# ---------------------------------------------------------------------------

def load_image_rgb(path: Path, size: int = IMG_SIZE) -> Optional[np.ndarray]:
    """Load one image as RGB uint8 (size, size, 3). Returns None on failure."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((size, size), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dataset 1: RealSign alphabet (CC0-1.0)
# ---------------------------------------------------------------------------

def prepare_alphabet() -> dict:
    """Build train/val/test arrays from RealSign's pre-existing splits.

    RealSign ships Training/Validation/Testing directories, one per letter A-Z.
    We use them as-is (no re-split) to preserve the documented leakage-free
    split. No signer metadata is available per image (documented limitation).
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    root = ALPHABET_SRC
    splits = ("Training", "Validation", "Testing")
    class_dirs = {
        split: sorted(
            d for d in (root / split).iterdir()
            if d.is_dir() and len(d.name) == 1 and d.name.isalpha()
        )
        for split in splits
    }

    # Verify all splits have the same classes.
    classes = sorted({d.name.upper() for split in splits for d in class_dirs[split]})
    for split in splits:
        split_classes = sorted({d.name.upper() for d in class_dirs[split]})
        if split_classes != classes:
            raise ValueError(
                f"Class mismatch in {split}: {split_classes} != {classes}"
            )

    label_to_index = {c: i for i, c in enumerate(classes)}

    result = {}
    for split in splits:
        images: list[np.ndarray] = []
        labels: list[int] = []
        discarded: list[tuple[str, str]] = []  # (path, reason)
        for class_dir in class_dirs[split]:
            cls = class_dir.name.upper()
            cls_idx = label_to_index[cls]
            for img_path in sorted(class_dir.glob("*.jpg")):
                arr = load_image_rgb(img_path)
                if arr is None:
                    discarded.append((str(img_path), "unreadable"))
                    continue
                images.append(arr)
                labels.append(cls_idx)
        result[split] = {
            "images": np.stack(images) if images else np.zeros((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
            "labels": np.asarray(labels, dtype=np.int64),
            "discarded": discarded,
            "class_dir": str(class_dir),
        }

    manifest = {
        "task": "alphabet",
        "source": "RealSign-Indian-Sign-Language-Dataset",
        "source_url": "https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset",
        "license": "CC0-1.0",
        "license_url": "http://creativecommons.org/publicdomain/zero/1.0/",
        "classes": classes,
        "n_classes": len(classes),
        "label_to_index": label_to_index,
        "img_size": IMG_SIZE,
        "n_channels": N_CHANNELS,
        "counts": {split: int(len(result[split]["labels"])) for split in splits},
        "discarded": {split: len(result[split]["discarded"]) for split in splits},
        "signer_metadata_available": False,
        "split_note": "Uses RealSign's pre-existing Training/Validation/Testing directories "
                      "(leakage-free by construction). No per-image signer identity available.",
    }
    return {"manifest": manifest, "splits": result, "label_to_index": label_to_index, "classes": classes}


# ---------------------------------------------------------------------------
# Dataset 2: Real-life Words (Mendeley)
# ---------------------------------------------------------------------------

def _extract_words_zip() -> Path:
    """Ensure the words outer zip is extracted to WORDS_EXTRACT_DIR."""
    WORDS_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    marker = WORDS_EXTRACT_DIR / ".extracted_ok"
    if marker.is_file():
        return WORDS_EXTRACT_DIR
    import zipfile
    with zipfile.ZipFile(WORDS_SRC_ZIP_DIR) as zf:
        zf.extractall(WORDS_EXTRACT_DIR)
    marker.write_text("ok")
    return WORDS_EXTRACT_DIR


def prepare_words() -> dict:
    """Build train/val/test arrays from Real-life Words.

    Each word class is a separate inner zip containing <word>/User_N/*.jpg.
    We extract all inner zips, then split by User_N (signer-aware) so that
    no signer appears in more than one split -- signer-independent eval.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    root = _extract_words_zip()
    # Find all inner <word>.zip files under New folder/
    new_folder = root / "Indian sign Language-Real-life Words" / "New folder"
    if not new_folder.is_dir():
        # Try alternative: the zip may extract directly
        candidates = list(root.rglob("*.zip"))
        if not candidates:
            raise FileNotFoundError(f"No inner word zips found under {root}")
        inner_zips = candidates
    else:
        inner_zips = sorted(new_folder.glob("*.zip"))

    if not inner_zips:
        raise FileNotFoundError(f"No inner word zips found in {new_folder or root}")

    # Extract each inner zip into a per-word directory, collect (word, user, path)
    word_user_images: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    discarded: list[tuple[str, str]] = []

    for inner_zip in inner_zips:
        word = inner_zip.stem  # e.g. "afraid"
        extract_dir = WORDS_EXTRACT_DIR / "words_extracted" / word
        if not any(extract_dir.iterdir()) if extract_dir.is_dir() else True:
            extract_dir.mkdir(parents=True, exist_ok=True)
            import zipfile
            with zipfile.ZipFile(inner_zip) as zf:
                zf.extractall(extract_dir)
        # Scan for images, infer user from path: .../<word>/User_N/<file>
        for img_path in extract_dir.rglob("*.jpg"):
            # Determine user id from path components
            user = None
            parts = img_path.relative_to(extract_dir).parts
            for p in parts:
                if p.startswith("User_"):
                    user = p
                    break
            if user is None:
                discarded.append((str(img_path), "no_user_id"))
                continue
            word_user_images[word][user].append(img_path)

    words = sorted(word_user_images.keys())
    label_to_index = {w: i for i, w in enumerate(words)}

    # Signer-aware split: assign whole users to train/val/test per word.
    # Use 70/15/15 by user count, stratified so each split has users for every word.
    split_data: dict[str, dict[str, list[np.ndarray]]] = {
        split: defaultdict(list) for split in ("train", "val", "test")
    }
    split_counts: dict[str, dict[str, int]] = {s: {} for s in ("train", "val", "test")}

    for word in words:
        users = sorted(word_user_images[word].keys())
        n_users = len(users)
        if n_users < 3:
            # Too few users for a 3-way split; put all in train, note it.
            n_train, n_val, n_test = n_users, 0, 0
        else:
            n_train = max(1, int(round(n_users * 0.70)))
            n_val = max(1, int(round(n_users * 0.15)))
            n_test = n_users - n_train - n_val
            if n_test < 1:
                n_test = 1
                n_val = max(0, n_users - n_train - n_test)

        train_users = users[:n_train]
        val_users = users[n_train:n_train + n_val]
        test_users = users[n_train + n_val:]

        for split, usrs in (("train", train_users), ("val", val_users), ("test", test_users)):
            cls_idx = label_to_index[word]
            n_imgs = 0
            for u in usrs:
                for img_path in word_user_images[word][u]:
                    arr = load_image_rgb(img_path)
                    if arr is None:
                        discarded.append((str(img_path), "unreadable"))
                        continue
                    split_data[split][word].append((arr, cls_idx))
                    n_imgs += 1
            split_counts[split][word] = n_imgs

    # Convert to arrays
    result = {}
    for split in ("train", "val", "test"):
        images: list[np.ndarray] = []
        labels: list[int] = []
        for word in words:
            for arr, cls_idx in split_data[split][word]:
                images.append(arr)
                labels.append(cls_idx)
        random.shuffle(list(zip(images, labels))) if images else None
        # reshuffle deterministically
        combined = list(zip(images, labels))
        random.shuffle(combined)
        if combined:
            images, labels = zip(*combined)
        result[split] = {
            "images": np.stack(images) if images else np.zeros((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
            "labels": np.asarray(labels, dtype=np.int64),
            "counts_per_class": split_counts[split],
        }

    manifest = {
        "task": "word",
        "source": "Indian Sign Language -- Real-life Words (Mendeley)",
        "source_url": "https://data.mendeley.com/datasets/s6kgb6r3ss/2",
        "license": "To verify on Mendeley dataset page",
        "classes": words,
        "n_classes": len(words),
        "label_to_index": label_to_index,
        "img_size": IMG_SIZE,
        "n_channels": N_CHANNELS,
        "counts": {split: int(len(result[split]["labels"])) for split in ("train", "val", "test")},
        "counts_per_class": split_counts,
        "discarded": len(discarded),
        "signer_metadata_available": True,
        "split_note": "Signer-aware split by User_N (70/15/15) so no signer appears "
                      "in more than one split (signer-independent eval).",
    }
    return {"manifest": manifest, "splits": result, "label_to_index": label_to_index, "classes": words}


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_split(npz_path: Path, images: np.ndarray, labels: np.ndarray) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, images=images, labels=labels)


def save_labels_json(path: Path, label_to_index: dict, classes: list, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index_to_label = {v: k for k, v in label_to_index.items()}
    payload = {
        "label_to_index": label_to_index,
        "index_to_label": {str(v): k for k, v in index_to_label.items()},
        "classes": classes,
        "n_classes": len(classes),
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Alphabet ----
    alpha = prepare_alphabet()
    for split in ("Training", "Validation", "Testing"):
        out_name = {"Training": "train", "Validation": "val", "Testing": "test"}[split]
        save_split(
            ARTIFACT_DIR / "alphabet" / f"{out_name}.npz",
            alpha["splits"][split]["images"],
            alpha["splits"][split]["labels"],
        )
    save_labels_json(
        ARTIFACT_DIR / "alphabet" / "labels.json",
        alpha["label_to_index"],
        alpha["classes"],
        {"source": alpha["manifest"]["source"], "license": alpha["manifest"]["license"]},
    )

    # ---- Words ----
    words = prepare_words()
    for split in ("train", "val", "test"):
        save_split(
            ARTIFACT_DIR / "words" / f"{split}.npz",
            words["splits"][split]["images"],
            words["splits"][split]["labels"],
        )
    save_labels_json(
        ARTIFACT_DIR / "words" / "labels.json",
        words["label_to_index"],
        words["classes"],
        {"source": words["manifest"]["source"], "license": words["manifest"]["license"]},
    )

    # ---- Overall manifest ----
    manifest = {
        "img_size": IMG_SIZE,
        "n_channels": N_CHANNELS,
        "random_seed": RANDOM_SEED,
        "tasks": {
            "alphabet": alpha["manifest"],
            "word": words["manifest"],
        },
        "artifact_paths": {
            "alphabet_train": "artifacts/alphabet/train.npz",
            "alphabet_val": "artifacts/alphabet/val.npz",
            "alphabet_test": "artifacts/alphabet/test.npz",
            "words_train": "artifacts/words/train.npz",
            "words_val": "artifacts/words/val.npz",
            "words_test": "artifacts/words/test.npz",
        },
    }
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("=== Dataset preparation complete ===")
    print(f"Artifact dir: {ARTIFACT_DIR}")
    print(f"Alphabet classes: {alpha['classes']}")
    print(f"Alphabet counts: {manifest['tasks']['alphabet']['counts']}")
    print(f"Word classes: {words['classes']}")
    print(f"Word counts: {manifest['tasks']['word']['counts']}")


if __name__ == "__main__":
    main()
