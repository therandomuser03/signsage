# SignSage — FDMSE-ISL 200-Word Training on Google Colab

Train a practical **200-word isolated Indian Sign Language (ISL) recognition baseline** on Colab's free GPU using the **FDMSE-ISL** dataset, entirely inside the Colab runtime. **Nothing is downloaded to your laptop.**

> **Honesty box.** The archive's internal layout is not published in the paper and is **not verified here** — it is engineered to *discover and report* the layout, and to **stop loudly** if the signer-independent split is violated, if fewer than 200 classes are eligible, if disk runs out, or if extraction/training crashes. It never fabricates results. It does not promote anything into SignSage's production pipeline.

---

## What you need first

1. A free [Google Colab](https://colab.research.google.com/) account.
2. **Approved access to FDMSE-ISL** from the portal: <https://cs.rkmvu.ac.in/~isl/isl-dataset/>. You receive a temporary access key by email after administrator approval. The key is **not** in this repository and must never be committed.
3. Free runtime disk: the notebook measures `/content`, `/tmp` and local scratch at startup and picks the fastest location that fits (**archive size + ~11 GB**: extraction subset, feature cache, slack). If nothing fits, it stops **before** downloading.
4. Research use only. Cite the dataset paper (below). Do not redistribute the archive.

## Step-by-step

### 1. Open the notebook in Colab

Colab → **File → Upload notebook** → choose `SignSage_FDMSE_ISL_200_Word_Training.ipynb`. (Or push it to your own GitHub and use **GitHub → Open in Colab**, or open it from Drive.)

### 2. Select a GPU runtime

**Runtime → Change runtime type → Hardware accelerator: GPU.** Colab free tier commonly assigns an **NVIDIA T4 (16 GB)**; any assigned CUDA GPU works unchanged. CPU-only also runs, but training will be very slow — don't.

### 3. Provide the access key (never typed into a cell)

Colab → **key icon** (left sidebar) → **Add new secret** → name **`FDMSE_ISL_KEY`**, value = your key → enable **Notebook access** (gear next to the secret). The notebook reads it via `google.colab.userdata`.

If the secret is unavailable you get a hidden prompt instead (input not echoed). Environment variable `FDMSE_ISL_KEY` also works (handy for local runs). **Never paste the key into a cell.**

### 4. (Optional) Enable Drive persistence in Cell 0

Only if you want checkpoints/bundle synced to Google Drive automatically (e.g., before an interruption-prone session):

```python
USE_DRIVE  = True
DRIVE_ROOT = Path("/content/drive/MyDrive/SignSage_FDMSE_training")
```

Otherwise leave `USE_DRIVE = False` — the notebook works entirely in `/content`. **Never** point Drive paths at the archive/subset/feature cache; heavy data stays local.

### 5. Run it

**Runtime → Run all.** Sections run in order:

| Section | What happens | Typical time (Colab T4, free tier) |
|---|---|---|
| 1. Environment | GPU/VRAM/RAM/disk/network checks, minimal pip installs | ~1–2 min |
| 2. Secure acquisition | HEAD probe (size, Range support), disk check, **resumable** download over HTTPS with `.part` file (206-resume; 200-after-Range → restart; 416 → verify-and-finalize; progress without exposing the key). No plain HTTP. | ~0.5–2 h (portal-dependent) |
| 3. Dataset inspection | zip listing, split/signer/word/view discovery, report | seconds |
| 4. Vocabulary selection | eligibility scoring → top 200 by coverage; saves `vocabulary.json` | seconds |
| 5. Extraction | explicit disk check on the selected subset **before** extracting; only selected classes extracted (~3–5 GB) | minutes |
| 6. Preprocessing | **benchmark first** (10 videos, measured), then MediaPipe Holistic → `sign_features_v3` → resumable feature cache | **~1–4 h CPU** for ~4k clips |
| 7–9. Dataset/model/training | `sign_features_v3` (60×122); 2-layer BiGRU (~0.5 M params), AMP, cosine LR, early stopping on val macro-F1, crash-safe checkpoints | ~10–40 min QUICK |
| 10. Evaluation | val for model selection; test **once**; confusion matrix, macro-F1, per-class report | minutes |
| 11. Export | `fdmse_200word_bundle.zip` written (to Drive too, if enabled) | seconds |

Landmark extraction is the bottleneck. The benchmark cell prints a measured projection for the full run; if it exceeds ~5 h, reduce `N_LANDMARK_FRAMES` (48→32) in Cell 0 and keep `MODE = "QUICK"`. **No fixed end-to-end time is promised** — free Colab tiers are not guaranteed windows, and the portal's bandwidth is shared.

### 6. Retrieve the exported model

`/content/fdmse_200word_bundle.zip` — use the **Files** panel (folder icon) → right-click → **Download**. Contents: `model.pt` (raw state dict), `meta.json` (backend-compatible), `label_mapping.json`, `vocabulary.json`, `preprocessing.json`, `metrics.json`, `README.txt`. If `USE_DRIVE=True`, the same zip lands in `SignSage_FDMSE_training/` on your Drive automatically.

## Resume after Colab disconnects

Everything important is resumable:

- The archive stays at `<scratch>/FDMSE-ISL.zip` (scratch is measured at startup — typically `/tmp`). **The 30 GB download never restarts.**
- Extraction skips files already extracted; feature caching skips files already processed (`sha1` of the relative path).
- Training resumes from `checkpoints/ckpt_latest.pt` (epoch + optimizer + scheduler + scaler + history) when `RESUME = True` (default). Re-run **Runtime → Run all** — completed steps skip themselves.
- With `USE_DRIVE=True`, the newest `ckpt_latest.pt`/`ckpt_best.pt`/`history.json` are also copied into Drive **after every improvement**, and are copied back on the next run if `/content` was wiped.
- With `USE_DRIVE=True`, the extracted-feature **shard archive** (`fdmse_features/shards/*.npz`) and its `manifest.csv` are synced to Drive at the end of Section 6, and **restored at startup of the next session** so landmark extraction does not need to re-run.
- Without Drive: anything not yet downloaded from `/content` is gone when the runtime dies. Download the bundle early and often.

## Integrating into SignSage (optional, experimental)

The bundle is **not** loaded by production code. An experimental loader is included at `backend/app/ml/fdmse_word_adapter.py` (mirrors the `word_classifier_adapter` shape; imports from the SignSage backend, not from Colab).

1. Download the zip from step 6 and unzip to e.g. `backend/word_model_fdmse_200/` (do **not** overwrite `backend/word_model/`).
2. Smoke-test **outside** production:

```python
import sys; sys.path.insert(0, "backend")
from app.ml.fdmse_word_adapter import FDMSEWordAdapter
ad = FDMSEWordAdapter.load("backend/word_model_fdmse_200")
print(ad.labels, ad.model_version, ad.confidence_threshold)
res = ad.classify_features(feat.npy)   # feat: (60, 122) float32, sign_features_v3 output
print(res)
```

3. Wire **nothing** into `/ws/translate`, `word_translation.py`, `/api/v1/capabilities`, or the frontend until evaluation gates pass. Promotion requires the gates in `docs/ml/MODEL_STATUS.md`. Until then, predictions are advisory only.

| Component | Status after this work |
|---|---|
| SignSage word_v3 model (9 words, one signer) | untouched; still the experimental word recognizer |
| FDMSE-ISL 200-word bundle | new **experimental artifact only** |
| Production pipeline, auth, history, translation | untouched |

## If auto-detection reports a layout it cannot parse

`Cell 5` prints its candidate scores per path depth. If the layout is unusual (e.g., metadata CSV with a separate split file, or signs in per-word folders without a split directory), set overrides at the top of the cell:

```python
FORCE_SPLIT_INDEX  = <depth of the split component>
FORCE_SIGNER_INDEX = <depth of the signer component>
FORCE_WORD_INDEX   = <depth of the word component>   # overrides word auto-detect
FORCE_VIEW         = "frontal"                        # if camera view can't be established
```

then re-run from Cell 5. The parsed table is printed and saved to `checkpoints/parsed_index.csv` — verify it before trusting the vocabulary selection.

## Troubleshooting

| Symptom | Action |
|---|---|
| `portal returned HTTP 403/401` | Access key wrong/expired, or secret not granted notebook access. Re-check **key icon → Notebook access**. |
| `NOT ENOUGH DISK` | The notebook stops *before* downloading. Free space (deleting other files from `/content`), or re-run on a roomier runtime. It never falls back to partial downloads. |
| 206 response mismatch | The partial file is preserved; the notebook aborts instead of appending. Inspect rate-limited/responses from the portal; delete the `.part` to restart from zero. |
| HTTP 416 | The partial is accepted **only after a fresh `HEAD` confirms the remote size matches**; otherwise the run aborts. |
| Camera view not established | Set `FORCE_VIEW=<token>` in Cell 0 to the true view; the session otherwise stops rather than mixing views. |
| `Extraction failure rate > 5%` | Inspect `checkpoints/extraction_failures.json`; usually path/parser mismatch, not corrupt video. Fix `FORCE_*` and re-run. |
| `signer overlap check: FAIL` | Partition split shares signers — either mis-parse or a broken archive. Inspect `parsed_index_full.csv`, fix `FORCE_SIGNER_INDEX`, re-run. This check cannot be bypassed. |
| Training crashes mid-run | `RESUME=True`; re-run — continues from last epoch. |
| `val_macro_f1` flat at chance | Check `vocabulary_report.txt`, confirm ≥1 video per val class, and that features aren't all-zero (Cell 11 fails loud in that case). |
| Files panel is empty after disconnect | `/content` is ephemeral. Set `USE_DRIVE=True` before long runs. |

## Citation

```bibtex
@article{PatraSuvajitPAA25,
  title={Hierarchical Windowed Graph Attention Transformer Encoder and a Large Scale Dataset for Indian Sign Language Recognition},
  author={Suvajit Patra and Arkadip Maitra and Megha Tiwari and K. Kumaran and Swathy Prabhu and Swami Punyeshwarananda and Soumitra Samanta},
  journal={Pattern Analysis and Applications},
  volume={28}, number={3}, pages={148}, year={2025}
}
```

## Verified vs. not verified

- ✅ Notebook JSON structure, cell order, imports, and all code cells parse (`ast`-validated).
- ✅ End-to-end machinery smoke-tested on a **synthetic** (non-FDMSE) 176-video corpus: download handling (mocked), archive check, discovery, split verification, vocabulary selection, selective extraction, benchmark, caching, dataset, overfit check, 1-epoch training, checkpointing, evaluation, export.
- ✅ Resume (HTTP 200 restart, HTTP 206 verified append, invalid/mismatched Content-Range aborts safely, HTTP 416 verify-and-finalize), disk preflight, camera-view filtering, signer-overlap abort and split-verification failure paths — all tested with mocked responses; checkpoint interrupt/resume tested end-to-end.
- ❌ **Nothing** about real FDMSE-ISL outcomes (layout, class coverage, achievable accuracy, runtime) is verified — the dataset has not been downloaded here. First real run happens in Colab; expect to spend one interactive session reading the Section 3 report before trusting anything downstream.
- ❌ No accuracy claim is made or implied. The published figure (93.86 % top-1, HWGAT) is the authors' graph-attention model, not this BiGRU baseline.
