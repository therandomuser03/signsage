# SignSage Runtime Artifact Storage Audit

**Status:** Verified storage plan + exact production-inference file list.  
**Approved for:** storage migration (separate approval).  
**Constraints honored:** Nothing uploaded to Hugging Face; no production adapter modified; no weights replaced; no files deleted; `render.yaml` and env vars unchanged; `SIGNSAGE_MODEL_PROMOTED=false` unchanged; nothing committed or pushed.

---

## 1. Exact runtime file dependency list

These are the only files the production backend can load for Sign→Text inference. Every path is relative to `SIGNSAGE_MODEL_BUNDLE_DIR` (Render: `/models/signsage_deployment_bundle`).

| File | Loaded when | Loader | Role |
|------|-------------|--------|------|
| `landmark_contract.json` | Startup-first-inference (bundle load) | `DeploymentContract.from_bundle` → `LandmarkContract.load` (`app/ml/contracts/landmark_contract.py:44`) | Authoritative 543-landmark contract: component order/ranges, `raw_feature_dim=1629`, `normalization_version=pose_normalize_v1`, `max_frames`, `final_model_input_dim=3258`. Cross-validated against the deployment manifest at load time (`inference_contract.py:27`). |
| `sign_to_text_deployment.json` | Startup-first-inference (bundle load) | `DeploymentContract.from_bundle` → `ModelContract.load` (`app/ml/contracts/model_contract.py:47`) | Deployment manifest: `feature_dim`, `model_input_dim`, `max_frames`, `max_tokens`, `vocab_size`, `special_token_ids` (PAD0/UNK1/BOS2/EOS3), `decoding`, `tokenizer_model`, `checkpoint` paths. |
| `checkpoint/<name>` (from manifest `checkpoint` key) | Lazy, on first `transcribe()` | `SignToTextAdapter.load` (`app/ml/adapters/sign_to_text_adapter.py:102`) → `torch.load(...)` | PyTorch checkpoint. Must contain `state_dict` and `model_config`; optional `vocab`, `tokenizer_fingerprint`. Reconstructs `SignTransformer` (`app/ml/models/sign_to_text.py:57`) and validates `input_dim`/`raw_feature_dim` against the contract. |
| `tokenizer/<name>` (from manifest `tokenizer_model` key) | Lazy, on first `transcribe()` | `SignToTextAdapter.load` → `sentencepiece.SentencePieceProcessor` | SentencePiece unigram model. Validated: special-token IDs must match manifest `special_token_ids`; optional `tokenizer_fingerprint` must match. |

**No other file is loaded by production inference code.** The tokenizer `.vocab` and `train_text.txt` are not read at runtime by the adapter (only the `.model` SentencePiece file is).

### What loads at application startup vs. lazily

- **At startup (`backend/app/main.py:23`):** `get_sign_to_text_pipeline()` runs only when `SIGNSAGE_MODEL_PROMOTED=true` OR `SIGNSAGE_ALLOW_UNPROMOTED_DEV=true`. Render sets `SIGNSAGE_MODEL_PROMOTED=false` and does not set the dev flag, so **no model file is loaded at startup on Render**. `/api/v1/health` honestly reports `model_bundle_found=false`, `model_loaded=false` (`app/api/routes/health.py:19`).
- **Lazy (first inference):** the four bundle files above, loaded by `SignToTextAdapter.load()` on first `transcribe()` call via the registry (`app/ml/models/registry.py:33`).

### Files required by the current `SignToTextAdapter`

`landmark_contract.json`, `sign_to_text_deployment.json`, one checkpoint `.pt`, one SentencePiece tokenizer `.model`. Confirmed by reading `sign_to_text_adapter.py:102-172` and `model_storage.py:22-40`.

### Files required by Model A (existing SignSage iSign T5-small-motion / run 352868924)

The adapter's contract expects: SP-16k SentencePiece tokenizer (`PAD0/UNK1/BOS2/EOS3`), 543-landmark `pose_normalize_v1` input, `input_dim=3258` (`raw_feature_dim=1629 × 2`), `SignTransformer` architecture (d_model/nhead/layers/ff from `model_config`). The verified checkpoint `verified_checkpoint/signsage_transformer_full.pt` loads through `SignToTextAdapter` (confirmed by `scripts/model_check.py` and `research/manifests/models/signsage-s2t_v352868924.yaml`), but it is **NO-GO** (WER 1.0046, EM 0.0, diversity 0.99%, trained on synthetic iSign text→pose data, not real ISL video). It is **not promoted** (`SIGNSAGE_MODEL_PROMOTED=false`) and must not be served.

A complete, promoted Model A deployment bundle (checkpoint + tokenizer + both JSON contracts) does **not** currently exist on disk in a form the backend references. The smoke-test bundle under `research/training/artifacts/smoke_test/signsage_deployment_bundle/` has the correct structure but is synthetic (`scope=smoke_synthetic`/pilot, WER 1.166, `GO=false`) and cannot be served as promoted.

### Files required by Model B (ISLiM / Sanket)

**Neither is required, integrated, or loadable by the current production adapter.**

- **ISLiM** (`research/pretrained_models/islim/checkpoints/ISLiM_4Loss_best_for_10kSpamoOF.pt`, ~860 MB): 94 MediaPipe Holistic landmarks × 4 dims, max_frames=128, Conv1D + 4-layer Transformer encoder + T5-small decoder, 16 query tokens. Incompatible with the 543-landmark SignSage adapter contract (`docs/architecture/DATA_AND_MODEL.md:88`, `research/pretrained_models/islim/MODEL_AUDIT.md:50-61`). Creating a compatible adapter would require a landmark re-extraction/remapping layer (543→94 or re-extract MediaPipe Holistic keypoints with visibility), a new adapter contract, and the T5-small HF tokenizer — **do not modify the production adapter**.
- **Sanket** (`research/pretrained_models/sanket/checkpoints/model_best.pth`, 1.4 GB): dual-stream TimeSformer video encoder + pose encoder (195/258 features) + T5-base decoder. Cannot run through the pose-only adapter; requires a separate video-and-pose inference pipeline (`research/pretrained_models/sanket/MODEL_AUDIT.md:30-42`). Dataset license (INCLUDE) unverified.

### Tokenizer, vocabulary, configuration, landmark contract

- Tokenizer: `tokenizer/signsage_en_sp.model` (SentencePiece) — required. `.vocab` is informational only, not read at runtime.
- Configuration: `sign_to_text_deployment.json` (manifest) — required.
- Landmark contract: `landmark_contract.json` — required.
- Vocabulary: no separate runtime vocabulary file; the SentencePiece model is the vocabulary.

### Dataset archives / training files accessed by production code

**None.** Grep of the entire backend (`backend/`) for `results-v4`, `results.zip`, `Indian sign`, `RealSign`, `staging_kaggle_results`, `research/pretrained_models`, `verified_checkpoint`, `data/manifests`, `iSign_v1.1`, `pose_cache`, `train_text`, `.npz`, `.csv` found **zero runtime references**. `data/manifests/subset6.csv` (1.7 KB) is referenced only by docs/training scripts, not backend runtime.

### Backend can start and serve non-inference endpoints without a model bundle

**Yes.** `render.yaml` startCommand runs `fetch_bundle.sh && uvicorn`. `fetch_bundle.sh` exits 0 when `SIGNSAGE_BUNDLE_URL` is unset or the bundle is already present; it never blocks startup. `/api/v1/health` and `/api/v1/capabilities` serve and report `model_bundle_found=false`. Translation endpoints (REST `/api/v1/translation/sign-to-text` and WebSocket `/ws/translate`) refuse with `MODEL_UNAVAILABLE` via the promotion gate (`registry.py:47`) and bundle-resolution errors. The checkpoint-free heuristic fallback (5 static hand shapes, `app/ml/inference/heuristic_signs.py`) needs no checkpoint and is reported honestly as `heuristic-fallback`.

---

## 2. Complete large-file inventory

Files >1 MB in the directories the task asked about (`models/`, `verified_checkpoint/`, `outputs/`, `research/`, `data/`, `staging_kaggle_results/`), plus root dataset zips and the `mobile/` research assets. Dependency/runtime files under `.venv/`, `frontend/node_modules/`, and Android Gradle build intermediates are excluded (they are not model/dataset artifacts).

Sizes in bytes; `~` = human-readable.

| # | Relative path | Size (bytes) | ~Size | Type | Association | Referenced by production code? | Required for | Safe outside GitHub? |
|---|---------------|-------------|-------|------|-------------|--------------------------------|--------------|----------------------|
| 1 | `research/results/previous_runs/results.zip` | 12,372,042,501 | ~11.5 GB | zip archive | Kaggle run 352868924 outputs (51 files) | No | Research/archive | Yes |
| 2 | `results-v4.zip` | 11,998,359,671 | ~11.2 GB | zip archive | Dataset/results archive | No | Research/archive | Yes |
| 3 | `staging_kaggle_results/s2t_v2/pose_cache/shard-00000.zip` | 4,010,935,738 | ~3.7 GB | zip | Pose cache shard | No | Training/eval pose cache | Yes |
| 4 | `staging_kaggle_results/s2t_v2/pose_cache/shard-00002.zip` | 3,800,526,178 | ~3.5 GB | zip | Pose cache shard | No | Training/eval pose cache | Yes |
| 5 | `staging_kaggle_results/s2t_v2/pose_cache/shard-00001.zip` | 3,534,079,622 | ~3.3 GB | zip | Pose cache shard | No | Training/eval pose cache | Yes |
| 6 | `staging_kaggle_results/s2t_v2/pose_cache/shard-00003.zip` | 3,092,384,110 | ~2.9 GB | zip | Pose cache shard | No | Training/eval pose cache | Yes |
| 7 | `research/pretrained_models/sanket/checkpoints/model_best.pth` | 1,438,145,048 | ~1.4 GB | pth | Sanket checkpoint (Model B candidate) | No | Research/eval | Yes |
| 8 | `research/pretrained_models/islim/checkpoints/ISLiM_4Loss_best_for_10kSpamoOF.pt` | 899,970,408 | ~859 MB | pt | ISLiM checkpoint | No | Research/eval | Yes |
| 9 | `research/pretrained_models/islim/checkpoints/ISLiM_full_posestitch_pre_processed.pt` | 899,966,348 | ~859 MB | pt | ISLiM checkpoint | No | Research/eval | Yes |
| 10 | `staging_kaggle_results/_diag/data.npz` | 734,003,530 | ~700 MB | npz | Pose cache | No | Training/eval | Yes |
| 11 | `RealSign-Indian-Sign-Language-Dataset-main.zip` | 653,370,264 | ~623 MB | zip | RealSign dataset | No | Research/archive | Yes |
| 12 | `Indian sign Language-Real-life Words.zip` | 514,941,675 | ~491 MB | zip | ISL words dataset | No | Research/archive | Yes |
| 13 | `research/pretrained_models/islim/mobile/models/model_state_dict.pt` | 300,002,192 | ~286 MB | pt | ISLiM full model state dict | No | Mobile research | Yes |
| 14 | `research/pretrained_models/islim/mobile/models/t5_decoder_first_step.pt` | 166,792,063 | ~159 MB | pt | ISLiM T5 decoder | No | Mobile research | Yes |
| 15 | `research/pretrained_models/islim/mobile/models/t5_decoder_step.pt` | 166,789,645 | ~159 MB | pt | ISLiM T5 decoder | No | Mobile research | Yes |
| 16 | `research/pretrained_models/islim/mobile/models/t5_encoder.pt` | 141,504,428 | ~135 MB | pt | ISLiM T5 encoder | No | Mobile research | Yes |
| 17 | `verified_checkpoint/kaggle_checkpoint/s2t_small6/checkpoints/signsage_transformer_full.pt` | 203,649,677 | ~194 MB | pt | Model A variant (s2t_small6) | No | Research/eval | Yes |
| 18 | `verified_checkpoint/kaggle_checkpoint/s2t_small6/checkpoints/signsage_transformer_recovery.pt` | 203,649,677 | ~194 MB | pt | Model A recovery | No | Research/eval | Yes |
| 19 | `outputs/checkpoint_reload_verified/signsage_transformer_full_16k.pt` | 203,649,677 | ~194 MB | pt | Model A 16k variant | No | Research/eval | Yes |
| 20 | `verified_checkpoint/signsage_transformer_full.pt` | 105,292,571 | ~100 MB | pt | Model A (352868924), loads via adapter | No | Research/eval (NO-GO) | Yes |
| 21 | `research/training/artifacts/smoke_test/checkpoints/signsage_transformer_full.pt` | 105,292,571 | ~100 MB | pt | Smoke-test Model A | No | Research/smoke | Yes |
| 22 | `research/training/artifacts/smoke_test/checkpoints/signsage_transformer_recovery.pt` | 105,294,735 | ~100 MB | pt | Smoke-test recovery | No | Research/smoke | Yes |
| 23 | `outputs/checkpoint_reload_verified/signsage_transformer_full.pt` | 105,287,771 | ~100 MB | pt | Model A reload-verified copy | No | Research/eval | Yes |
| 24 | `research/training/artifacts/smoke_test/signsage_deployment_bundle/checkpoint/signsage_transformer_full.pt` | 35,268,763 | ~34 MB | pt | Smoke-test bundle checkpoint (subset) | No | Research/smoke | Yes |
| 25 | `research/pretrained_models/islim/mobile/models/t5_lm_head.pt` | 65,802,265 | ~63 MB | pt | ISLiM T5 lm_head | No | Mobile research | Yes |
| 26 | `research/pretrained_models/islim/mobile/models/t5_shared_embed.pt` | 65,802,283 | ~63 MB | pt | ISLiM shared embed | No | Mobile research | Yes |
| 27 | `research/pretrained_models/islim/mobile/models/t5_decoder_first_step.ptl` | 166,567,061 | ~159 MB | ptl | ISLiM traced (duplicate of #14) | No | Mobile research | Yes |
| 28 | `research/pretrained_models/islim/mobile/models/t5_decoder_step.ptl` | 153,977,975 | ~147 MB | ptl | ISLiM traced (duplicate of #15) | No | Mobile research | Yes |
| 29 | `research/pretrained_models/islim/mobile/models/t5_encoder.ptl` | 75,564,757 | ~72 MB | ptl | ISLiM traced (dup) | No | Mobile research | Yes |
| 30 | `research/pretrained_models/islim/mobile/models/t5_lm_head.ptl` | 65,802,133 | ~63 MB | ptl | ISLiM traced (dup) | No | Mobile research | Yes |
| 31 | `research/pretrained_models/islim/mobile/models/t5_shared_embed.ptl` | 65,801,654 | ~63 MB | ptl | ISLiM traced (dup) | No | Mobile research | Yes |
| 32 | `research/pretrained_models/islim/mobile/models/encoder_traced.pt` | 57,993,186 | ~55 MB | pt | ISLiM pose encoder | No | Mobile research | Yes |
| 33 | `research/pretrained_models/islim/mobile/android_test/app/src/main/assets/test_data/keypoints_module.pt` | 10,976,573 | ~10 MB | pt | ISLiM test data | No | Mobile test | Yes |
| 34 | `models/iSign_t5_small_motion/t5-small-motion/pose_encoder.pt` | 1,144,463 | ~1.1 MB | pt | ISLiM pose encoder (external eval path) | Only by `scripts/evaluate_external_isign.py` + smoke test | External eval | Yes |
| 35 | `staging_kaggle_results/isign/iSign_v1.1.csv` | 9,808,778 | ~9.4 MB | csv | iSign dataset (convenience cache) | No | Research/reference | Yes |
| 36 | `staging_kaggle_results/s2t_v2/tokenizer/train_text.txt` | 6,085,144 | ~5.8 MB | txt | Training text | No | Training | Yes |
| 37 | `outputs/translation/tokenizer/signsage_en_sp.model` | 538,842 | ~526 KB | pt/.model | SentencePiece tokenizer | No (copy) | Research | Yes |
| 38 | `outputs/translation/tokenizer/signsage_en_sp.vocab` | 306,259 | ~299 KB | vocab | SentencePiece vocab | No (not read at runtime) | Research | Yes |
| 39 | `staging_kaggle_results/_diag/signsage_en_sp.model` | 538,842 | ~526 KB | .model | Tokenizer (dup of #37) | No | Research | Yes |
| 40 | `staging_kaggle_results/s2t_v2/tokenizer/signsage_en_sp.model` | 538,834 | ~526 KB | .model | Tokenizer (dup of #37) | No | Research | Yes |
| 41 | `research/image_classification/artifacts/alphabet/model.pt` | 8,793,189 | ~8.4 MB | pt | Static alphabet CNN | Only via `image_alphabet_bundle_dir` env (unset by default) | Image classify endpoint | Yes |
| 42 | `research/image_classification/artifacts/words/model.pt` | 8,787,109 | ~8.4 MB | pt | Static word CNN | Only via `image_word_bundle_dir` env (unset by default) | Image classify endpoint | Yes |
| 43 | `research/training/artifacts/smoke_test/signsage_deployment_bundle/tokenizer/signsage_en_sp.model` | 247,690 | ~242 KB | .model | Smoke-test tokenizer | No | Research/smoke | Yes |

**SHA-256 values** (computed, byte-exact): see per-file hashes below. Key hashes:
- `verified_checkpoint/signsage_transformer_full.pt` = `92b437196545027ed6aa8eb6e34e52b8998d0d671f64923371ea14dd59327f36` (identical to `research/training/artifacts/smoke_test/checkpoints/signsage_transformer_full.pt`)
- `verified_checkpoint/kaggle_checkpoint/s2t_small6/checkpoints/signsage_transformer_full.pt` = `8043db7d6dab6c7d3f562f00a0e0793766c66784538eebf1f1867c7882a2d9de` (identical to `.../signsage_transformer_recovery.pt` and `outputs/checkpoint_reload_verified/signsage_transformer_full_16k.pt`)
- `outputs/checkpoint_reload_verified/signsage_transformer_full.pt` = `809be1b2388294761fd47820cc8e97dbf1ef4799c8767b8864f745f0ba2d5b6d` (different 105 MB file)
- `research/pretrained_models/islim/checkpoints/ISLiM_4Loss_best_for_10kSpamoOF.pt` = `cf97d0ac9abcdfced0e128e98eede967ddf2479b2590eb0d1b891f96f176fc54`
- `research/pretrained_models/islim/checkpoints/ISLiM_full_posestitch_pre_processed.pt` = `ffe4efbb5262abdd6221e4c8d818c92fc1db7fa7ea955ae0afa827251dd9140c`
- `research/pretrained_models/sanket/checkpoints/model_best.pth` = `03cda82d26ab7e6d0adc8718d41d072e5181aef380a33dca17f08d3bd7dae729`
- `research/results/previous_runs/results.zip` = `3e3f950ccaa6fc2e502e91c50c1931d9a7efe920d7fb8249d0875949f3bf70a9`
- `results-v4.zip` = `d44b8b91f7a69e0d31675f39766c41b8b841a0ed920f0bbc01e4b1fcedc1fcf4`
- `RealSign-...zip` = `5f53cd71553010a5bc598ce66fc45067329fa37bcddcc20ce617d4228c97b432`
- `Indian sign Language-Real-life Words.zip` = `7b4367cd6e7e4a949e9950ad3b92f870faf180ee7965e22e99d634271c0d18dc`
- Pose-cache shards #3-6: `5b94fc4658de118a651ddd22abfba273527596d0b52a253d318081fb3b5c23e4`, `5658906b823085940db7a3472c5ce7e7cb3792d4cf6973b18dd2d9f7778683fd`, `5ba0e395962e43e7902dfdc7f9e89e1191a0b5b8cdeed9e721e00e412333bcd5`, `d521aae599a7e3ba13b995cf9000d68c4ae512e7acbaaca10e21b4a0341e9664`
- `staging_kaggle_results/_diag/data.npz` = `3e23065622db03fe8c6659f72221948e626da6a6a832d9f5f26717111c12283a`; `val.npz` = `b407f28a959a93b7bee47ec9e656bcbdda7b3bccef619996d6bd9bcda0ad4bc8`
- `staging_kaggle_results/isign/iSign_v1.1.csv` = `6299ed581eb16c2850ca1502057b85d1990b6539c6787dc81564048d3c695943`
- Tokenizer model copies (#37-40) = `e2964684c936ea4986f0144562ae39329c887dcb16fd1a02728a4af97a0c1421` (identical bytes)

**Duplicates confirmed by identical SHA-256 + size:**
- `verified_checkpoint/signsage_transformer_full.pt` == `research/training/artifacts/smoke_test/checkpoints/signsage_transformer_full.pt` (92b4371...)
- kaggle_small6 `full.pt` == `recovery.pt` == `outputs/checkpoint_reload_verified/signsage_transformer_full_16k.pt` (8043db7...)
- ISLiM `*.pt` under `mobile/models/` == same-named `*.ptl`/Android-asset copies (same size; `.ptl` bytes differ slightly from `.pt` but represent the same traced exports)
- Tokenizer `.model` copies (#37-40) are byte-identical.

**Not production-referenced:** Every file above is research/training/evaluation/mobile. Zero backend runtime code references any of them (verified by grep). The image-classification `model.pt` files (#41-42) are loadable only if `SIGNSAGE_IMAGE_ALPHABET_BUNDLE_DIR`/`SIGNSAGE_IMAGE_WORD_BUNDLE_DIR` are set — both unset in `render.yaml` and default `None` in config, so the image-classification REST endpoint reports the bundle as missing until configured.

---

## 3. Production vs research artifact classification

| Category | Artifacts | Storage recommendation | Rationale |
|----------|-----------|------------------------|-----------|
| **A. Required production inference artifact** | None currently deployable. The contract requires `checkpoint/`, `tokenizer/`, `landmark_contract.json`, `sign_to_text_deployment.json` at `SIGNSAGE_MODEL_BUNDLE_DIR`, but no complete, promoted bundle exists that production references. | Private Hugging Face Hub Model repo (versioned ZIP) → Render persistent disk at runtime via `fetch_bundle.sh`. Not GitHub. | Production must not serve the NO-GO checkpoint. Bundle is the deployable unit; HF is the versioned store; Render disk is a runtime copy. |
| **B. Required production config/tokenizer/preprocessing asset** | `landmark_contract.json`, `sign_to_text_deployment.json`, `tokenizer/signsage_en_sp.model` — required **only together with a valid checkpoint**. | Same as A (inside the versioned bundle ZIP). Not GitHub for weights; contract JSON could live in Git but travels with the bundle. | Adapter validates all four at load (`sign_to_text_adapter.py:102`; `model_storage.py:35`). |
| **C. Research checkpoint** | Model A (`verified_checkpoint/*.pt`, `outputs/checkpoint_reload_verified/*.pt`, `research/training/artifacts/smoke_test/checkpoints/*.pt`, kaggle_small6 copies), ISLiM (`islim/checkpoints/*.pt`), Sanket (`sanket/checkpoints/model_best.pth`), `models/iSign_t5_small_motion/.../pose_encoder.pt` | Local research storage; private HF Hub **only after licensing verified** (ISLiM/Sanket rights unresolved — see §6). Not GitHub (`*.pt`/`*.pth` gitignored). | Not promoted; not served. Keep for research/repro. |
| **D. Training dataset or pose cache** | `staging_kaggle_results/s2t_v2/pose_cache/*` (~14.4 GB), `_diag/data.npz` + `val.npz`, root dataset zips (`results-v4.zip`, `RealSign-*.zip`, `Indian sign Language-...zip`), `iSign_v1.1.csv`, `train_text.txt`, `results.zip` | Authorized external dataset storage / local research. **Not Render** (2 GB disk insufficient; not required by production code). | Not referenced by backend runtime. iSign is CC BY-NC-SA 4.0 gated; INCLUDE license unverified — do not rehost or put on Render. |
| **E. Evaluation artifact** | `outputs/*` reports (`FINAL_AUDIT_REPORT_VERIFIED.json`, `reference_prediction_table.md`, `unresolved_risks.md`, etc.), `research/evaluation/*`, smoke-test `evaluation_report.json`/`gate_report.json` | Local research storage; small JSON/Markdown reports may be committed to GitHub, large ones excluded. | Evidence only, not served. |
| **F. Mobile-specific model or asset** | `research/pretrained_models/islim/mobile/*` (TorchScript `.ptl`, Android assets, `t5_config.json`), Android Gradle build intermediates (~800 MB native libs) | Local research storage. Not GitHub (build intermediates gitignored; `.ptl` large binaries not tracked). | Mobile-only; not required for server inference (`HF_STORAGE_POLICY.md:66`). |
| **G. Obsolete or duplicate artifact** | `outputs/checkpoint_reload_verified/signsage_transformer_full.pt` (distinct 105 MB copy), `kaggle_small6/recovery.pt` duplicates, `mobile/models/*.pt` duplicates of `research/.../mobile/models/*.pt`, tokenizer `.model` copies, smoke-test `tokenizer/` dir (duplicate of bundle tokenizer), `research/training/artifacts/smoke_test/checkpoints/recovery.pt` | Leave in place (do not delete per constraints); exclude from Git; do not migrate to Render/HF. | Duplicates of research artifacts; no production role. |

---

## 4. Dataset storage recommendations

- **Do not put complete training datasets on Render.** Production code accesses no dataset archive, pose cache, or CSV at runtime. The 2 GB Render disk cannot hold the pose cache (~14.4 GB) or result zips (~23 GB combined), and there is no operational reason to.
- **iSign** (`iSign_v1.1.csv`, `.pose` archive referenced at 158+ GB, CC BY-NC-SA 4.0, gated HF `Exploration-Lab/iSign`): keep in authorized external/local research storage only. Do not commit; do not rehost; do not download unless a bounded prototype is separately approved.
- **INCLUDE** (Sanket dataset, ~4,284 videos, 250 categories): license unverified. Do not store on Render or HF until rights verified.
- **Pose cache** (`staging_kaggle_results/s2t_v2/pose_cache/`, ~14.4 GB across 4 shards; `_diag/data.npz`+`val.npz`): pre-extracted continuous pose data; contains almost none of the 6-word pilot vocab. Keep in local/external research storage. Not Render.
- **Dataset zips at repo root** (`results-v4.zip`, `RealSign-*.zip`, `Indian sign Language-...zip`, `Indian-Sign-Language-dataset-main.zip`): already gitignored; keep local/external only.

---

## 5. Model storage recommendations

- **Production deployment bundle:** Private Hugging Face Hub Model repository, versioned ZIP pinned to a commit SHA, downloaded by Render at startup via `fetch_bundle.sh` into `SIGNSAGE_MODEL_BUNDLE_DIR` on the persistent disk. Immutable + commit-pinned URL (`https://huggingface.co/USER/REPO/resolve/COMMIT_SHA/signsage_deployment_bundle-v1.zip?download=true`). `SIGNSAGE_HF_TOKEN` as Render secret for private repo. **Do not create the repo or upload anything** (approval pending).
- **Research checkpoints (ISLiM, Sanket, Model A variants):** local research storage; HF only after licensing verification. Not GitHub.
- **Do not bake weights into the Docker image** (Dockerfile already confirms: weights not baked; `PIP_NO_CACHE_DIR`, CPU torch).
- **GitHub:** only code, configs, and small contract JSONs. All `*.pt`/`*.pth`/`.ptl`/dataset zips are gitignored.

---

## 6. Current Render architecture

- `render.yaml`: Docker web service `signsage-backend`, health check `/api/v1/health`, 2 GB persistent disk `signsage-models` mounted at `/models`, `startCommand: bash scripts/fetch_bundle.sh && exec uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Env: `SIGNSAGE_MODEL_BUNDLE_DIR=/models/signsage_deployment_bundle`, `SIGNSAGE_MODEL_PROMOTED=false`, `SIGNSAGE_DEVICE=cpu`, `SIGNSAGE_LOG_LEVEL=INFO`, CORS origins, `SIGNSAGE_BUNDLE_URL`/`SIGNSAGE_BUNDLE_SHA256`/`SIGNSAGE_HF_TOKEN` (dashboard secrets, `sync: false`), optional Supabase/Groq/Ollama.
- `backend/Dockerfile`: `python:3.11-slim`, CPU-only torch, `SIGNSAGE_MODEL_BUNDLE_DIR` and `SIGNSAGE_MODEL_PROMOTED=false` set at build, non-root user, no weights baked in.
- `backend/scripts/fetch_bundle.sh`: downloads `SIGNSAGE_BUNDLE_URL` ZIP to `SIGNSAGE_MODEL_BUNDLE_DIR`; verifies `SIGNSAGE_BUNDLE_SHA256` when set; extracts; validates required entries (`sign_to_text_deployment.json`, `landmark_contract.json`, `checkpoint/`, `tokenizer/`). Skips download when bundle already complete on disk (persistent Render disk reuse). Exits 0 (warns) when URL unset — server starts, health reports `model_bundle_found=false`. HF token auth supported (`SIGNSAGE_HF_TOKEN`/`HF_TOKEN`, Bearer, only to `huggingface.co`, stripped on cross-host redirect). Single-shot `urllib` download — **no resume/range support**; interrupted download = full re-download.
- Startup lifespan (`backend/app/main.py:23`): loads model only if `model_promoted` or `allow_unpromoted_dev`; failure logged, not fatal.
- Registry (`backend/app/ml/models/registry.py`): loads `SignToTextAdapter` (default `model_kind=sign_to_text`) or `IsolatedWordAdapter` (+ optional `ContinuousSignPipeline`) lazily, process-wide, with promotion gate.

---

## 7. Render disk capacity analysis

| Item | Size |
|------|------|
| Configured disk | 2 GB |
| One typical bundle (ZIP + extracted) | ~200–700 MB (checkpoint 35–204 MB + tokenizer ~0.5 MB + JSONs) |
| Two retained versions (active + rollback) | ~400 MB–1.4 GB |
| Three retained versions | ~600 MB–2.1 GB (exceeds 2 GB at the high end) |
| Pose cache / datasets | Not stored on Render (not required) |

**Verdict:** 2 GB is sufficient for a **single active bundle** (even the 204 MB checkpoint × ZIP + extracted ≈ 400 MB). It is **not sufficient to retain multiple rollback versions** of larger checkpoints alongside the active bundle. **Recommend increasing disk to 4 GB minimum** if Render retains old bundle directories during rollout/rollback (per `RENDER_MODEL_DEPLOYMENT.md:151-155`, do not overwrite in place; upload to a new path and switch `SIGNSAGE_MODEL_BUNDLE_DIR`). If only one active version is ever kept, 2 GB is acceptable but leaves little headroom.

---

## 8. Hugging Face deployment workflow

Per `docs/development/RENDER_MODEL_DEPLOYMENT.md` and `backend/scripts/fetch_bundle.sh`:

1. Expected bundle ZIP structure (archive root): `checkpoint/`, `tokenizer/`, `landmark_contract.json`, `sign_to_text_deployment.json` (optionally `evaluation_results.json`, `token_vocabulary.json`). Confirmed by `fetch_bundle.sh:24-29`, `model_storage.py:35`, and the smoke-test bundle.
2. Private HF auth: **supported** — `fetch_bundle.sh` sends `Authorization: Bearer <SIGNSAGE_HF_TOKEN>` only to `huggingface.co` and strips it on any cross-host redirect (`fetch_bundle.sh:78-106`). Public repo needs no token.
3. SHA-256 verification: supported when `SIGNSAGE_BUNDLE_SHA256` is set; **should be mandatory** for production (currently optional — empty string skips).
4. Resume/recover: **not supported** — single-shot download; no range/resume; interrupted download is removed and retried from scratch.
5. Complete existing bundle reuse: **supported** — `required_present()` skip-logic avoids re-download when the bundle is already on the persistent disk. No complete promoted bundle exists on disk today.
6. 2 GB disk: sufficient for one active bundle; increase to 4 GB if retaining rollback versions.
7. Disk for ZIP + extracted + retained versions: one bundle ≈ 2× checkpoint size; two versions ≈ 4×. See §7.
8. Recommendation: **immutable, versioned artifacts + commit-pinned download URLs**; never overwrite a bundle in place; switch `SIGNSAGE_MODEL_BUNDLE_DIR` to a new versioned directory per release; keep old directory until new release verified.

**Do not create the HF repository or upload anything** (separate approval).

---

## 9. Licensing blockers

| Artifact | License / rights status | Blocker? |
|----------|------------------------|----------|
| **iSign** (dataset + weights derived from it, incl. Model A 352868924) | CC BY-NC-SA 4.0, gated HF `Exploration-Lab/iSign`. Non-commercial, share-alike. Weights do **not** inherit the repo's MIT code license. Whether serving predictions on Render counts as commercial use is unresolved; separate agreement may be needed. | **Yes** — NC/SA commercial-use question unresolved; gated access requires `HF_TOKEN`. |
| **INCLUDE** (Sanket dataset) | License not stated in model card or HF repo; source Kaggle. **Unverified.** Sanket model card tags `MIT` for the model only; dataset rights unclear. | **Yes** — must verify independently before deployment/rehosting. |
| **ISLiM** checkpoints | Dataset iSign = research-only, not commercial. Checkpoint license not explicitly stated; assume research-only unless established. Redistribution rights unverified. | **Yes** — do not rehost/upload until verified. |
| **Sanket** checkpoint | Model card `MIT`; INCLUDE dataset license unverified. | **Yes** (dataset side). |
| **Private HF repo** | Does **not** resolve licensing restrictions. | N/A — policy: do not upload weights whose redistribution rights are not established (`HF_STORAGE_POLICY.md:12-14`, `DATA_AND_MODEL.md:114-123`). |

**License gate (must be recorded before any public/commercial checkpoint deployment):** (1) whether NC/SA permits the intended use or a separate agreement is needed; (2) whether serving predictions (incl. Render bundle download) counts as commercial use; (3) attribution/ShareAlike obligations for bundle + adaptations; (4) whether the dataset-derived tokenizer needs the same treatment. Local research/train/eval/dev inference is not blocked.

---

## 10. Required future implementation changes

1. **Export and validate a complete, promoted deployment bundle** with the exact structure in §8 (checkpoint + tokenizer + both JSON contracts), run `python -m scripts.model_check` against it, and only then consider flipping `SIGNSAGE_MODEL_PROMOTED` (separate human decision; currently `false`).
2. **Pin and verify `SIGNSAGE_BUNDLE_SHA256`** on Render (make checksum mandatory) and set `SIGNSAGE_BUNDLE_URL` to a commit-pinned private HF URL + `SIGNSAGE_HF_TOKEN` secret.
3. **Increase Render disk to 4 GB** if retaining rollback bundle versions; otherwise keep 2 GB with a single active version.
4. **Add resume/range support** to `fetch_bundle.sh` (optional, for large bundles on interrupted deploys) — currently single-shot.
5. **Do not integrate ISLiM or Sanket** into the production adapter: ISLiM needs a 94-landmark remapping layer and separate contract; Sanket needs a separate video+pose (TimeSformer + T5-base) pipeline. Any such work is a new adapter, not a change to `SignToTextAdapter`.
6. **Resolve licensing** for iSign (CC BY-NC-SA 4.0), INCLUDE, ISLiM, Sanket before any upload/rehost or production serving.
7. **Keep `SIGNSAGE_MODEL_PROMOTED=false`** until a future run passes the quality gate (WER/BLEU + diversity + trivial-baseline + exact_match + signer-independent evaluation) and receives explicit approval.

---

## Recommended storage structure

```
GitHub (code + small config only; no weights, no datasets)
├── backend/
│   ├── app/                          # FastAPI + ML adapters (contracts live here)
│   ├── scripts/fetch_bundle.sh       # bundle downloader (unchanged)
│   ├── scripts/model_check.py        # bundle loader self-test (unchanged)
│   └── Dockerfile                    # CPU torch, no baked weights
├── render.yaml                       # 2 GB disk at /models (unchanged; consider 4 GB for rollback)
├── docs/
│   ├── architecture/DATA_AND_MODEL.md
│   ├── development/RENDER_MODEL_DEPLOYMENT.md
│   └── reports/RUNTIME_ARTIFACT_STORAGE_AUDIT.md   # this report
└── (all *.pt/*.pth/*.ptl/dataset zips → gitignored)

Private Hugging Face Hub Model repo (versioned, commit-pinned; NOT created yet)
└── signsage_deployment_bundle-v1.zip   # root: checkpoint/, tokenizer/, landmark_contract.json, sign_to_text_deployment.json

Render persistent disk at /models (runtime copy only; re-downloaded by fetch_bundle.sh)
└── /models/signsage_deployment_bundle/
    ├── checkpoint/
    │   └── signsage_transformer_full.pt
    ├── tokenizer/
    │   └── signsage_en_sp.model
    ├── landmark_contract.json
    ├── sign_to_text_deployment.json
    └── (optional) evaluation_results.json

Authorized external / local research storage (NOT GitHub, NOT Render)
├── research/pretrained_models/            # ISLiM, Sanket checkpoints (rights unverified)
├── verified_checkpoint/                   # Model A copies (NO-GO)
├── outputs/                               # evaluation reports
├── staging_kaggle_results/                # pose cache (~14.4 GB), iSign CSV
├── research/results/previous_runs/results.zip
└── results-v4.zip, RealSign-*.zip, "Indian sign Language-...zip"  # dataset zips

Mobile assets (research only; not server inference)
└── research/pretrained_models/islim/mobile/   # TorchScript .ptl, Android assets
```

**Exact production-inference file list (relative to `SIGNSAGE_MODEL_BUNDLE_DIR`):**
1. `landmark_contract.json`
2. `sign_to_text_deployment.json`
3. `checkpoint/<checkpoint>`  (the `.pt` named by the manifest `checkpoint` key)
4. `tokenizer/<tokenizer>`   (the SentencePiece `.model` named by the manifest `tokenizer_model` key)

No dataset archive, training file, research checkpoint, or mobile asset is required for production inference. Nothing above has been uploaded, modified, deleted, committed, or pushed.
