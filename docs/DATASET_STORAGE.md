# SignSage Dataset Storage Workflow

## Overview

This document describes the dataset storage, integrity, and workflow patterns for SignSage.
The old laptop serves as a **storage/data node only** — it does not train models,
and the system must function correctly when the laptop is offline.

No dataset files are committed to this Git repository. All dataset paths reference
external storage locations.

---

## 1. Expected Dataset Directory Structure

The iSign dataset (or approved subset) should reside on the old laptop at:

```
/signsage-data/
  ├── isign_2024/
  │   ├── annotations/
  │   │   └── isign_annotations_train.json
  │   │   └── isign_annotations_val.json
  │   │   └── isign_annotations_test.json
  │   ├── videos/
  │   │   ├── train/
  │   │   ├── val/
  │   │   └── test/
  │   ├── labels/
  │   │   └── sign_vocabulary.json  ← 14 avatar gloss IDs (EN→ISL generation)
  │   └── README.md
  ├── signed_subsets/
  │   └── trial_run_202408/
  │       ├── train/
  │       │   ├── class_0001/  ← 4 samples
  │       │   ├── class_0002/  ← 4 samples
  │       │   └── class_0003/
  │       ├── val/
  │       │   ├── class_0001/
  │       │   └── class_0002/
  │       └── test/
  │           ├── class_0001/
  │           └── class_0002/
  ├── checksums/
  │   ├── isign_2024_sha256.txt
  │   └── trial_run_202408_sha256.txt
  └── metadata/
      └── dataset_manifest.json
```

## 2. iSign Metadata Location

- `signsage-data/isign_2024/labels/sign_vocabulary.json`
  - Contains 14 avatar gloss IDs used for EN→ISL generation
  - **Not** recognition labels — these are generation labels only
- `signsage-data/metadata/dataset_manifest.json`
  - Primary metadata entry point (see Section 5)

## 3. Dataset Integrity / Checksum Verification

A SHA-256 checksum file must accompany any distributed dataset:

```
signsage-data/isign_2024/ isign_2024_sha256.txt
```

Example `isign_2024_sha256.txt` format:

```
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  isign_2024/labels/sign_vocabulary.json
aaf4c6f0a6d6...  isign_2024/videos/train/video_001.mp4
...
```

Verification command (run on any machine, including the old laptop):

```bash
cd /signsage-data
sha256sum -c isign_2024_sha256.txt
```

## 4. Dataset Version Identification

The `signsage-data/metadata/dataset_manifest.json` must contain a version field:

```json
{
  "dataset_name": "isign_2024",
  "version": "1.0.0",
  "description": "iSign dataset subset for SignSage training",
  "source": "iSign-2024-release",
  "checksum_sha256": "isign_2024_sha256.txt",
  "created": "2024-08-15",
  "notes": "14 avatar gloss IDs are EN→ISL generation labels, not recognition labels"
}
```

The version string must be updated whenever the dataset composition changes.
The backend references this version via `SIGNSAGE_DATASET_VERSION` env var.

## 5. Materialized Subset Storage

When a reduced training subset is materialized (e.g., for Kaggle or quick
experiments), it lives under `signsage-data/signed_subsets/<run_name>/`:

```
signed_subsets/trial_run_202408/
  train/
    class_0001/   ← exactly 4 video samples
    class_0002/   ← exactly 4 video samples
    class_0003/   ← exactly 4 video samples
  val/
    class_0001/
    class_0002/
  test/
    class_0001/
    class_0002/
```

**Rules for materialized subsets:**

- Each class must have exactly `N` samples (default `N=4` for trial runs).
- Subsets are **immutable** once created — do not add/remove samples after
  materialization without incrementing the version.
- Subsets are stored as regular video files (e.g., `.mp4`, `.avi`).
- A `subset_manifest.json` accompanies each materialized run:

```json
{
  "run_name": "trial_run_202408",
  "version": "1.0.0",
  "samples_per_class": 4,
  "classes": ["class_0001", "class_0002", "class_0003"],
  "source": "full isign_2024/train",
  "checksum_sha256": "trial_run_202408_sha256.txt"
}
```

## 6. Training Artifact Storage

Training artifacts (PyTorch `.pt` files, logs, configs) are stored separately:

```
/signsage-artifacts/
  checkpoints/
    isign_2024/
      v1.0.0/
        model_0001.pt
        model_0002.pt
        best.pt
  logs/
    isign_2024/
      v1.0.0/
        training.log
        metrics.json
  configs/
    isign_2024/
      v1.0.0/
        training_config.yaml
```

**Important:** Artifacts are **not** in the Git repo. They reside on the old
laptop or a separate artifact server. The `SIGNSAGE_ARTIFACTS_DIR` env var
points to this location.

## 7. Model Checkpoint Storage

Model checkpoints follow the pattern in Section 6. The promoted model bundle
is referenced by `SIGNSAGE_MODEL_BUNDLE_DIR`. If no model is promoted,
`SIGNSAGE_MODEL_PROMOTED=false` and the pipeline honestly returns
`MODEL_UNAVAILABLE`.

**Never promote checkpoint 352868924** — it has been verified NO-GO (WER 1.0046,
EM 0.0, diversity 0.99%).

## 8. Evaluation Report Storage

Evaluation reports are stored as JSON alongside checkpoints:

```
/signsage-artifacts/
  evaluations/
    isign_2024/
      v1.0.0/
        evaluation.json
        confusion_matrix.png
        diversity_plot.png
```

`evaluation.json` must contain at minimum:

```json
{
  "version": "1.0.0",
  "wer": 0.123,
  "em": 0.456,
  "diversity": 0.789,
  "samples_evaluated": 1234,
  "metadata": "See MODEL_READINESS.md for promotion gate requirements"
}
```

## 9. Backup Strategy

**On the old laptop:**

- The `/signsage-data/` directory is **rsynced** to a secondary location
  (external HDD or cloud bucket) on a weekly schedule.
- A cron job runs `sha256sum -c` after each sync to verify integrity.
- **Retention**: Keep 3 weekly snapshots (e.g., `isign_2024_2024-08-07.tar.gz`,
  `isign_2024_2024-08-14.tar.gz`, `isign_2024_2024-08-21.tar.gz`).

**Off-laptop policy:**

- The system **must function** if the old laptop is offline.
- All dataset checksums and metadata are versioned; if the laptop is unavailable,
  the pipeline returns `MODEL_UNAVAILABLE` honestly (no fabricated results).
- Kaggle independent operation is the fallback — see Section 10.

## 10. Tailscale / SSH Access Assumptions

- The old laptop has a stable Tailscale IP (e.g., `100.x.x.x`).
- SSH access: `ssh signsage-laptop.local` (via Tailscale SSH alias).
- All dataset verification commands assume the laptop is reachable via
  Tailscale, but **must degrade gracefully** if the laptop is offline.
- Example verification (run from any machine on the Tailscale network):

```bash
ssh signsage-laptop.local "sha256sum -c /signsage-data/isign_2024_sha256.txt"
```

- If the laptop is offline, the command fails — the pipeline simply returns
  `MODEL_UNAVAILABLE`. No error is raised.

## 11. Offline / Laptop-Unavailable Policy

The system must continue to work if the old laptop is offline. The pipeline
behavior is:

1. On startup, if `SIGNSAGE_DATA_DIR` is not accessible → `MODEL_UNAVAILABLE`
2. If checksum verification fails → `MODEL_UNAVAILABLE`
3. If no promoted model exists (`SIGNSAGE_MODEL_PROMOTED=false`) → `MODEL_UNAVAILABLE`
4. The frontend renders "Translation unavailable" panel honestly.

**No fluent English sentences are ever fabricated.** The system always tells
the user when translation is unavailable.

## 12. Kaggle Independent Operation

Kaggle must be capable of receiving and materializing the training subset
independently. The workflow:

1. Kaggle developer downloads the checksum-verified `isign_2024/` subset
   from the external source.
2. Materializes it under `signed_subsets/kaggle_run_2024/` following the
   structure in Section 5.
3. Sets `SIGNSAGE_DATA_DIR=/path/to/materialized/subset`.
4. Runs training or evaluation independently — no old laptop required.

The `subset_manifest.json` and `dataset_manifest.json` ensure version
and composition consistency across environments.

---

## Helper Scripts (Optional)

### `verify_dataset.sh` — Checksum verification (run anywhere)

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/signsage-data}"
CHECKSUM_FILE="${DATA_DIR}/isign_2024_sha256.txt"

if [[ ! -f "$CHECKSUM_FILE" ]]; then
  echo "ERROR: Checksum file not found at $CHECKSUM_FILE"
  exit 1
fi

echo "Verifying dataset in $DATA_DIR ..."
sha256sum -c "$CHECKSUM_FILE"
echo "Verification complete."
```

### `check_manifest.py` — Validate dataset manifest

```python
#!/usr/bin/env python3
import json
import sys
from pathlib import Path

MANIFEST = Path("/signsage-data/metadata/dataset_manifest.json")

def main():
    if not MANIFEST.exists():
        print(f"ERROR: Manifest not found at {MANIFEST}")
        sys.exit(1)

    with open(MANIFEST) as f:
        data = json.load(f)

    required = ["dataset_name", "version", "checksum_sha256"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"ERROR: Missing required manifest fields: {missing}")
        sys.exit(1)

    print(f"Manifest OK: {data['dataset_name']} v{data['version']}")
    print(f"Checksum: {data['checksum_sha256']}")

if __name__ == "__main__":
    main()
```

**Do not commit these scripts or their output to Git.** Keep them on the old
laptop or a personal dev machine only.

---

## Summary

| Component | Location | Offline Behavior |
|-----------|----------|-----------------|
| Full dataset | `/signsage-data/isign_2024/` | `MODEL_UNAVAILABLE` if inaccessible |
| Checksums | `/signsage-data/checksums/` | Verified on-access; fails → unavailable |
| Materialized subsets | `/signsage-data/signed_subsets/` | Kaggle can materialize independently |
| Training artifacts | `/signsage-artifacts/` | Separate from data; promote gate independent |
| Model checkpoints | `/signsage-artifacts/checkpoints/` | Never promote 352868924 |
| Evaluation reports | `/signsage-artifacts/evaluations/` | Versioned alongside checkpoints |
| Tailscale/SSH | `signsage-laptop.local` | Graceful degradation; no exposure |

**Key invariant**: The system never fabricates translation results. If data or
models are unavailable, the user is honestly informed.