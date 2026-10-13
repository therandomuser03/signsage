# SignSage — Kaggle `small6` Experiment Runbook

**Status:** Ready to run. Preflight verified. No training launched.

---

## 1. Local Subset Preparation (Run Once)

```bash
cd /mnt/D/Codes/signsage

python3 -c "
import csv
import pandas as pd

VOCAB = ['you','music','three','one','water','two']
df = pd.read_csv('staging_kaggle_results/isign/iSign_v1.1.csv')
sub = df[df['text'].astype(str).str.strip().str.lower().isin(VOCAB)].copy()
with open('subset6.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['uid','text'])
    w.writeheader()
    w.writerows(sub.to_dict('records'))
print(f'rows: {len(sub)}  videos: {sub["uid"].str.rsplit("--",1).str[0].nunique()}')
from collections import Counter
print(Counter(sub['text'].str.strip().str.lower()))
"
```

**Expected output:**
```
rows: 74  videos: 49
Counter({'you': 21, 'music': 20, 'three': 12, 'one': 9, 'water': 8, 'two': 4})
```

Upload `subset6.csv` to Kaggle as a **dataset** (private or public). Note the dataset slug, e.g. `yourname/signsage-subset6`.

---

## 2. Kaggle Repository & Input Setup

### 2.1 Create Kaggle Notebook
- New Notebook → **GPU T4 x2** accelerator
- Add-ons → Secrets → Add `GITHUB_TOKEN` (repo read access)
- Input dataset: attach your `subset6.csv` dataset. Note the mount path, e.g. `/kaggle/input/signsage-subset6/subset6.csv`
- Optional: `HF_TOKEN` under Add-ons → Secrets for HuggingFace pose streaming (required only if using `--pose-source remote` in training)

### 2.2 Verify Environment (First Cell)

```python
%%python
# GPU check - STOP if no compatible GPU
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU available — experiment requires GPU")
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name} | {props.total_memory/1e9:.1f} GB VRAM")

# Disk check
import shutil
total, used, free = shutil.disk_usage("/kaggle/working")
print(f"Disk: total={total/1e9:.1f}GB used={used/1e9:.1f}GB free={free/1e9:.1f}GB")
if free < 5e9:
    raise RuntimeError(f"Insufficient disk space: {free/1e9:.1f}GB free, need >5GB")

# Dependency check - exact imports from trainer
import sys
sys.path.insert(0, "/kaggle/working/signsage/research/training")
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import pose_format
    import jiwer
    import sacrebleu
    from rouge_score import rouge_scorer
    import sentencepiece
    import huggingface_hub
    import pandas as pd
    import tqdm
    import zstandard
    import zipfile
    from collections import Counter
    from datetime import datetime, timezone
    from pathlib import Path
    from contextlib import nullcontext
    import gc, hashlib, io, json, math, os, random, re, shutil, sys, time, warnings
    print("✓ All heavy dependencies importable")
except ImportError as e:
    raise RuntimeError(f"Missing dependency: {e}")
```

### 2.3 Clone Repo (Second Cell)

```python
%%bash
import os, subprocess, urllib.request, json
from pathlib import Path

REPO_URL = "https://github.com/therandomuser03/signsage.git"
DEST = Path("/kaggle/working/signsage")
TRAIN_SCRIPT = "research/training/train_sign_to_text.py"

token = os.environ.get("GITHUB_TOKEN")
if not token:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        token = None
if not token:
    raise RuntimeError("GITHUB_TOKEN missing: add under Add-ons > Secrets")

# Verify token works
req = urllib.request.Request(
    "https://api.github.com/repos/therandomuser03/signsage",
    headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
with urllib.request.urlopen(req, timeout=20) as r:
    info = json.loads(r.read().decode())
print("Repo accessible:", info.get("full_name"), "| private:", info.get("private"))

# Clone at pinned commit (update if repo advances)
PINNED_SHA = "43ed3572d7854de4aa8c39ef27b21a8632c51f64"
if not (DEST / ".git").exists():
    subprocess.run(["git", "clone", REPO_URL, str(DEST)], check=True)
subprocess.run(["git", "-C", str(DEST), "fetch", "origin"], check=True)
subprocess.run(["git", "-C", str(DEST), "checkout", PINNED_SHA], check=True)
print("Checked out:", PINNED_SHA)
```

---

## 3. GPU, Memory, Disk Checks

Run after clone (add to a new cell):

```python
%%python
import torch, shutil
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} | {props.total_memory/1e9:.1f} GB")

total, used, free = shutil.disk_usage("/kaggle/working")
print(f"Disk free: {free/1e9:.1f} GB (need >5 GB)")

import sys
sys.path.insert(0, "/kaggle/working/signsage/research/training")
from train_sign_to_text import train_tokenizer, encode_text, decode_ids
print("Training module importable")
```

**Minimum requirements:**
- GPU: T4 x2 (16 GB VRAM each) or equivalent
- RAM: ≥ 12 GB
- Disk: ≥ 5 GB free on `/kaggle/working`

**Pose source note:** If using `--pose-source remote` in training, you must add `HF_TOKEN` to Kaggle secrets (Add-ons → Secrets). The `GITHUB_TOKEN` is required for repo cloning; `HF_TOKEN` is only needed for remote pose streaming mode.

---

## 4. Architecture Patch (Apply Before Training)

```bash
%%bash
cd /kaggle/working/signsage

# Verify current constants
grep -n "D_MODEL, NHEAD, NLAYERS, FF" research/training/train_sign_to_text.py
# Should show: 256, 4, 4, 1024

# Apply patch (exact line, no extra whitespace)
sed -i 's/D_MODEL, NHEAD, NLAYERS, FF = 256, 4, 4, 1024/D_MODEL, NHEAD, NLAYERS, FF = 128, 4, 2, 512  # small6 experiment only/'     research/training/train_sign_to_text.py

# Verify
grep -n "D_MODEL, NHEAD, NLAYERS, FF" research/training/train_sign_to_text.py
# Must now show: 128, 4, 2, 512

# Fail if patch did not apply correctly
if ! grep -q "D_MODEL, NHEAD, NLAYERS, FF = 128, 4, 2, 512" research/training/train_sign_to_text.py; then
    echo "ARCHITECTURE PATCH FAILED: expected line not found"
    exit 1
fi
```

**Do not proceed if the line differs from expected. Edit manually if needed.**

---

## 5. Exact Training Command (Validated)

```bash
%%bash
cd /kaggle/working/signsage

python research/training/train_sign_to_text.py   --pose-source remote   --csv /kaggle/input/signsage-subset6/subset6.csv   --data-mode stream   --cache-budget-gib 2.0   --epochs 8   --patience 2   --batch 32   --lr 1e-4   --decoder-input-dropout 0.2   --max-hours 4.0   --seed 1   --out /kaggle/working/s2t_small6   --stages all
```

### Verified Arguments (vs. parser at `train_sign_to_text.py:1495-1556`)

| Arg | Value | Verified |
|-----|-------|----------|
| `--csv` | `/kaggle/input/signsage-subset6/subset6.csv` | ✅ |
| `--data-mode` | `stream` | ✅ |
| `--cache-budget-gib` | `2.0` | ✅ |
| `--epochs` | `8` | ✅ |
| `--patience` | `2` | ✅ |
| `--batch` | `32` | ✅ |
| `--lr` | `1e-4` | ✅ |
| `--decoder-input-dropout` | `0.2` | ✅ |
| `--max-hours` | `4.0` | ✅ |
| `--seed` | `42` | ✅ |
| `--out` | `/kaggle/working/s2t_small6` | ✅ |
| `--stages` | `all` | ✅ |

### Active Guards (Code-Confirmed)

- **4-hour wall-clock stop** → `train_sign_to_text.py:2075`
- **Early stop (patience=2)** → line 2072-2073
- **Collapse gate (unique_fraction ≥ 0.05)** → line 2037-2042
- **Overfit gate (64 reachable samples ≥ 95% exact)** → line 1802-1825
- **Video-grouped split (seed=1)** → line 1598, 1601, 2017
- **Export on `export` stage** → line 2122-2124
- **Evaluation with trivial baseline, per-class recall, diversity** → line 2100-2113

---

## 6. Artifact Persistence & Retrieval

### 6.1 What Gets Created in `/kaggle/working/s2t_small6/`

After training completes (or is interrupted):

| Artifact | Path | Description |
|----------|------|-------------|
| Best checkpoint | `checkpoints/signsage_transformer_full.pt` | `torch.save` with `state_dict` + `model_config` |
| Recovery checkpoints | `checkpoints/recovery_epoch_*.pt` | Per-epoch resume points |
| Training history | `training_history.json` | Loss, diversity, grad stats per epoch |
| Evaluation report | `evaluation_report.json` | **Must contain acceptance keys** |
| Tokenizer | `tokenizer/signsage_en_sp.model` + `.vocab` | SentencePiece unigram |
| Deployable bundle | `signsage_deployment_bundle/` | Checkpoint + tokenizer + contracts + eval |
| Dataset audit | `dataset_audit.json` | Cleaning stats + video split (seed=1) |
| Pose preflight | `pose_preflight.json` | Confirms stream source reachable |

### 6.2 Verify Artifacts Exist (Run After Training — Required Before Ending Session)

```bash
%%bash
set -e

echo "Checking required artifacts..."

# Check top-level directory
test -d /kaggle/working/s2t_small6 && echo "s2t_small6/ directory: EXISTS" || { echo "FAIL: s2t_small6/ directory missing"; exit 1; }

# Check best checkpoint
test -f /kaggle/working/s2t_small6/checkpoints/signsage_transformer_full.pt && echo "signsage_transformer_full.pt: EXISTS ($(stat -c%s /kaggle/working/s2t_small6/checkpoints/signsage_transformer_full.pt) bytes)" || { echo "FAIL: signsage_transformer_full.pt missing"; exit 1; }

# Check recovery checkpoints
recovery_found=$(ls /kaggle/working/s2t_small6/checkpoints/recovery_epoch_*.pt 2>/dev/null | wc -l)
if [ "$recovery_found" -gt 0 ]; then
    echo "Recovery checkpoints: EXISTS ($recovery_found files)"
else
    echo "FAIL: recovery_epoch_*.pt missing"
    exit 1
fi

# Check training history
test -f /kaggle/working/s2t_small6/training_history.json && echo "training_history.json: EXISTS" || { echo "FAIL: training_history.json missing"; exit 1; }

# Check evaluation report
test -f /kaggle/working/s2t_small6/evaluation_report.json && echo "evaluation_report.json: EXISTS" || { echo "FAIL: evaluation_report.json missing"; exit 1; }

# Check tokenizer model
test -f /kaggle/working/s2t_small6/tokenizer/signsage_en_sp.model && echo "signsage_en_sp.model: EXISTS" || { echo "FAIL: signsage_en_sp.model missing"; exit 1; }

# Check tokenizer vocab
test -f /kaggle/working/s2t_small6/tokenizer/signsage_en_sp.vocab && echo "signsage_en_sp.vocab: EXISTS" || { echo "FAIL: signsage_en_sp.vocab missing"; exit 1; }

# Check dataset audit
test -f /kaggle/working/s2t_small6/dataset_audit.json && echo "dataset_audit.json: EXISTS" || { echo "FAIL: dataset_audit.json missing"; exit 1; }

# Check pose preflight
test -f /kaggle/working/s2t_small6/pose_preflight.json && echo "pose_preflight.json: EXISTS" || { echo "FAIL: pose_preflight.json missing"; exit 1; }

# Check deployment bundle checkpoint
test -f /kaggle/working/s2t_small6/signsage_deployment_bundle/checkpoint/signsage_transformer_full.pt && echo "deployment bundle checkpoint: EXISTS" || { echo "FAIL: deployment bundle checkpoint missing"; exit 1; }

# Check deployment bundle tokenizer
test -f /kaggle/working/s2t_small6/signsage_deployment_bundle/tokenizer/signsage_en_sp.model && echo "deployment bundle tokenizer: EXISTS" || { echo "FAIL: deployment bundle tokenizer missing"; exit 1; }

# Check deployment bundle landmark contract
test -f /kaggle/working/s2t_small6/signsage_deployment_bundle/landmark_contract.json && echo "landmark_contract.json: EXISTS" || { echo "FAIL: landmark_contract.json missing"; exit 1; }

# Check deployment bundle deployment config
test -f /kaggle/working/s2t_small6/signsage_deployment_bundle/sign_to_text_deployment.json && echo "sign_to_text_deployment.json: EXISTS" || { echo "FAIL: sign_to_text_deployment.json missing"; exit 1; }

# Check deployment bundle evaluation results
test -f /kaggle/working/s2t_small6/signsage_deployment_bundle/evaluation_results.json && echo "evaluation_results.json: EXISTS" || { echo "FAIL: evaluation_results.json missing"; exit 1; }

echo "✓ All required artifacts exist and are non-empty"
```

### 6.3 Persist via Notebook Versioning (Required)

**Do not rely on `/kaggle/output/` or manual copy. Only `File → Save Version` in the Kaggle notebook UI preserves `/kaggle/working/` as a downloadable output dataset.**

1. After training completes (or is interrupted), verify artifacts exist:
```bash
ls -la /kaggle/working/s2t_small6/
ls /kaggle/working/s2t_small6/checkpoints/
ls /kaggle/working/s2t_small6/tokenizer/
ls /kaggle/working/s2t_small6/signsage_deployment_bundle/
```

2. In the Kaggle notebook UI: **File → Save Version** (not "Save & Run All")
   - This snapshots the entire `/kaggle/working/` directory into the notebook's **Output dataset**
   - The output appears under the notebook's **Output** tab as a downloadable zip
   
3. **Do not close the notebook or end the session until the Save Version completes.**

### 6.4 Retrieve After Session

- Go to the notebook's **Output** tab
- Click the version you saved → **Download** (zip) or attach as **Input Dataset** to another notebook
- The zip contains the entire `/kaggle/working/` snapshot; extract and locate `s2t_small6/` inside
- Expected zip structure:
```
s2t_small6/
├── checkpoints/
│   ├── signsage_transformer_full.pt
│   └── recovery_epoch_*.pt
├── training_history.json
├── evaluation_report.json
├── tokenizer/
│   ├── signsage_en_sp.model
│   └── signsage_en_sp.vocab
├── signsage_deployment_bundle/
│   ├── checkpoint/signsage_transformer_full.pt
│   ├── tokenizer/signsage_en_sp.model
│   ├── landmark_contract.json
│   ├── sign_to_text_deployment.json
│   └── evaluation_results.json
├── dataset_audit.json
└── pose_preflight.json
```

---

## 7. Evaluation & Acceptance Criteria

### 7.1 Verify Evaluation Report Schema (Run After Training)

```python
import json
import math
from pathlib import Path

p = Path("/kaggle/working/s2t_small6/evaluation_report.json")
print("Exists:", p.exists())
if not p.exists():
    raise RuntimeError("evaluation_report.json not found")

r = json.loads(p.read_text())
print("Top-level keys:", list(r.keys()))
tm = r.get("test_metrics", {})

# Required keys - FAIL FAST if missing or wrong type
# Also reject booleans where numeric metrics are expected
required = [
    ("test_metrics.exact_match", tm.get("exact_match"), (int, float)),
    ("test_metrics.per_class_recall", tm.get("per_class_recall"), dict),
    ("test_metrics.nonempty_output", tm.get("nonempty_output"), (int, float)),
    ("gate.GO", r.get("gate", {}).get("GO"), bool),
    ("diversity.unique_fraction", r.get("diversity", {}).get("unique_fraction"), (int, float)),
    ("trivial_baseline.exact_match", r.get("trivial_baseline", {}).get("exact_match"), (int, float)),
]
for name, val, expected_type in required:
    if val is None:
        raise ValueError(f"MISSING REQUIRED KEY: {name}")
    # Reject booleans where numeric metrics are expected (bool is subclass of int)
    if isinstance(val, bool):
        raise ValueError(f"WRONG TYPE for {name}: boolean value not accepted, expected numeric")
    if not isinstance(val, expected_type):
        raise ValueError(f"WRONG TYPE for {name}: expected {expected_type}, got {type(val)}")
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        raise ValueError(f"INVALID VALUE for {name}: NaN or infinity")
print("✓ All required keys present with correct types (no booleans passed as metrics)")

# Exactly six classes required in per_class_recall
expected_classes = {"you", "music", "three", "one", "water", "two"}
per_class = tm["per_class_recall"]
actual_classes = set(per_class.keys())
if actual_classes != expected_classes:
    raise ValueError(f"per_class_recall must have exactly these 6 classes: {sorted(expected_classes)}, got: {sorted(actual_classes)}")

# Acceptance criteria - ALL MUST PASS
em = tm["exact_match"]
baseline = r["trivial_baseline"]["exact_match"]
per_class = tm["per_class_recall"]
div = r["diversity"]["unique_fraction"]
gate_go = r["gate"]["GO"]

# Type and value checks - also reject booleans
for name, val in [("exact_match", em), ("baseline", baseline), ("diversity", div)]:
    if isinstance(val, bool):
        raise ValueError(f"WRONG TYPE for {name}: boolean value not accepted, expected numeric")
    if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
        raise ValueError(f"INVALID {name}: {val}")

results = []
print(f"Exact match: {em:.4f} (need ≥ 0.70) → {'PASS' if em >= 0.70 else 'FAIL'}")
results.append(("exact_match ≥ 0.70", em >= 0.70))

print(f"Above baseline: {em - baseline:.4f} (need ≥ 0.20) → {'PASS' if em - baseline >= 0.20 else 'FAIL'}")
results.append(("exact_match - baseline ≥ 0.20", em - baseline >= 0.20))

pc_pass = all(v >= 0.5 for v in per_class.values())
print(f"Per-class recall ≥ 0.50: {pc_pass} (values: {per_class})")
results.append(("per-class recall ≥ 0.50 all 6", pc_pass))

print(f"Diversity: {div:.4f} (need ≥ 0.50) → {'PASS' if div >= 0.50 else 'FAIL'}")
results.append(("diversity ≥ 0.50", div >= 0.50))

print(f"Gate GO: {gate_go} → {'PASS' if gate_go else 'FAIL'}")
results.append(("gate.GO is True", gate_go is True))

# Hard failures - exit nonzero on any failure
all_pass = all(r for _, r in results)
if not all_pass:
    failures = [name for name, pass_ in results if not pass_]
    raise RuntimeError(f"FAIL: acceptance criteria not met: {', '.join(failures)}")

print("✓ ALL ACCEPTANCE CRITERIA PASSED")
```

### 7.2 Acceptance Criteria (All Must Pass)

| Criterion | Threshold | Source |
|-----------|-----------|--------|
| Exact match | ≥ 70% | `test_metrics["exact_match"] ≥ 0.70` |
| Above trivial baseline | ≥ 20 pp | `exact_match - trivial_baseline["exact_match"] ≥ 0.20` |
| Per-class recall | ≥ 50% each | `all(v ≥ 0.50 for v in per_class_recall.values())` |
| Output diversity | ≥ 5/6 classes | `diversity["unique_fraction"] ≥ 0.50` |
| Overfit gate | Pass | `gate["GO"] == true` |
| Export success | Bundle exists | `signsage_deployment_bundle/` present |
| Activity gate | Pass | `model_check.py` exits 0 |

**If any key is missing or any criterion fails → experiment fails. Do not loosen thresholds.**

---

## 8. Deployment Check (`model_check.py`) — Separate from Metrics

**A metrics pass does NOT imply the deployment check passes.** The bundle must be separately verified.

### 8.1 Local Check Command (After Downloading Output)

```bash
# After downloading the Kaggle output dataset to local machine
export SIGNSAGE_MODEL_BUNDLE_DIR=/path/to/downloaded/s2t_small6/signsage_deployment_bundle
python -m scripts.model_check

# Must run from the backend directory or with backend on PYTHONPATH:
cd /mnt/D/Codes/signsage/backend
python -m scripts.model_check
```

### 8.2 Expected Success Output

```
Loaded bundle OK. max_frames=160, vocab_size=25, model_promoted=False
Activity gate self-test passed (reason='insufficient_motion').
```

### 8.3 Failure Modes

| Output | Meaning |
|--------|---------|
| `FAILED to load model bundle` | Bundle structure invalid or missing files |
| `Activity gate self-test passed` | ✅ Bundle is servable |
| `FAILED: static/no-motion landmarks were accepted` | ❌ Model produces false positives |

**The model is NOT promoted.** `SIGNSAGE_MODEL_PROMOTED` remains `false`. Promotion is a separate human decision. A metrics pass does NOT imply promotion eligibility.

---

## 9. Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `ModuleNotFoundError: pose_format` | Missing deps | Run pip install cell first |
| `GITHUB_TOKEN missing` | Secret not set | Add under Add-ons → Secrets |
| `Disk full` | Cache too large | Reduce `--cache-budget-gib` or clear `/kaggle/working` |
| `CUDA OOM` | Batch too large | Reduce `--batch` to 16 |
| `Collapse gate fired` | Diversity < 5% | Experiment failed; inspect `COLLAPSED.txt` |
| `Overfit gate failed` | <95% exact on 64 samples | Check reachability filtering; may need more data |
| `No video overlap` but val missing classes | Seed unlucky | Try different seed or increase `--epochs` |
| `evaluation_report.json` missing keys | Evaluation stage skipped | Ensure `--stages all` or includes `evaluate` |
| `model_check.py` exits 1 | Bundle not loadable | Check `landmark_contract.json` + `sign_to_text_deployment.json` in bundle |

---

## 10. Values You Must Fill In

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `yourname/signsage-subset6` | Kaggle dataset slug for subset6.csv | `therandomuser03/signsage-subset6` |
| `/kaggle/input/.../subset6.csv` | Actual mount path of attached dataset | `/kaggle/input/signsage-subset6/subset6.csv` |
| `PINNED_SHA` | Git commit to train from | `43ed3572d7854de4aa8c39ef27b21a8632c51f64` |
| `REPO_URL` | Your fork or upstream | `https://github.com/therandomuser03/signsage.git` |

---

## 11. Files Changed in This Runbook

| File | Change |
|------|--------|
| `research/training/train_sign_to_text.py` | Line 88: `256,4,4,1024` → `128,4,2,512` (patched at runtime) |

**No other files modified.** No commits, no pushes.

---

## 12. Final Checklist Before Launch

- [ ] `subset6.csv` built locally, 74 rows / 49 videos, 6 classes
- [ ] `subset6.csv` uploaded to Kaggle, mount path known
- [ ] Kaggle notebook: GPU T4 x2, secrets `GITHUB_TOKEN`
- [ ] Repo cloned at pinned SHA, architecture patch applied and verified
- [ ] Training command copy-pasted exactly (no trailing backslash spaces)
- [ ] `--max-hours 4.0 --patience 2 --seed 42` confirmed
- [ ] Artifact verification step planned after training
- [ ] Notebook **Save Version** workflow understood
- [ ] Local `model_check.py` test planned after download
- [ ] Acceptance criteria printed and accessible

---

**Ready to launch when all checkboxes are ticked.**
