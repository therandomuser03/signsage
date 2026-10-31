# Hugging Face Storage Policy — SignSage

Status: **Not integrated**, **Not promoted**, **Blocked** on licensing verification and adapter contract before any upload.

## Recommended future repositories
- `therandomuser03/signsage-islim` (ISLiM checkpoints + config + mobile inference docs)
- `therandomuser03/signsage-sanket` (Sanket checkpoint + config + tokenizer docs)

Neither repository has been created. No upload performed.

## Policy rules (before any upload)
1. Verify checkpoint redistribution rights (ISLiM: dataset `iSign` is research-only `CC BY-NC-SA 4.0`; checkpoint license unverified. Sanket: `MIT` model tag; `INCLUDE` dataset license unverified).
2. Verify dataset license permits redistribution of derived weights (iSign: no commercial rights established. INCLUDE: unverified).
3. Do not upload datasets (`iSign_v1.1.csv`, `.pose` archives, `INCLUDE` clips) to Hugging Face.
4. Use model repositories only for large weights (`.pt`, `.pth`). Use Git LFS only if needed for smaller tracked artifacts.
5. Pin a model revision or commit SHA. Record SHA-256 hash separately (`research/pretrained_models/islim/MODEL_AUDIT.md`; `sanket/MODEL_AUDIT.md`).
6. Download weights during deployment through a controlled artifact downloader (`backend/scripts/fetch_bundle.sh` or equivalent). Never commit weights to Git.
7. Never store Hugging Face access tokens (`HF_TOKEN`) in source code (`docs/development/DEVELOPMENT.md` §5; `.gitignore`: never commit `.env` or secrets).
8. Keep model configuration (`t5_config.json`, tokenizer files, adapter contracts) in GitHub (`docs/architecture/`, `backend/app/ml/adapters/`).
9. Do not promote any model (`SIGNSAGE_MODEL_PROMOTED=false`) merely because weights load. Promotion requires independent evaluation (`docs/architecture/DATA_AND_MODEL.md` §5), verified adapter contract (`docs/reference/end-to-end-contract.md`), and separate human-approved GO.

## Current model assets
- ISLiM: `research/pretrained_models/islim/checkpoints/` (`ISLiM_4Loss_best_for_10kSpamoOF.pt`: 899,970,408 bytes, SHA-256 `cf97d0ac...`; `ISLiM_full_posestitch_pre_processed.pt`: 899,966,348 bytes, SHA-256 `ffe4efbb...`). Mobile assets: `research/pretrained_models/islim/mobile/`.
- Sanket: `research/pretrained_models/sanket/checkpoints/model_best.pth`: 1,438,145,048 bytes, SHA-256 `03cda82d...`.
- Production model (`iSign_t5_small_motion/`) preserved separately: `models/iSign_t5_small_motion/`. Not uploaded.
- Verified experimental checkpoint (`verified_checkpoint/signsage_transformer_full.pt`): preserved; not promoted; source `results-v4.zip` (12 GB, untouched, preserved).

## Getting model files
Model weights and TorchScript exports are intentionally excluded from Git. A fresh clone will not contain them.

- ISLiM checkpoints: download from the official [Google Drive folder](https://drive.google.com/drive/folders/100H39_Pcpd6WeXmXG0-bEvif-HDHihzC?usp=sharing) and place them in `research/pretrained_models/islim/checkpoints/`. Redistribution rights remain unverified; do not rehost them.
- Sanket checkpoint: obtain `model_best.pth` from the [Hugging Face model repository](https://huggingface.co/ayush2635/sanket-isl-translator) and place it in `research/pretrained_models/sanket/checkpoints/`. Verify dataset rights before deployment or rehosting.
- ISLiM mobile exports: the original split archives were user-provided; no verified public download URL is recorded. Obtain them from the original provider and extract them under `research/pretrained_models/islim/mobile/`. Android tests expect their assets under `mobile/android_test/app/src/main/assets/`.

## Outstanding licensing issues
- ISLiM dataset (`iSign`) license: **Unverified** for weight redistribution; dataset restricted to research use. No upload permitted without author agreement.
- Sanket dataset (`INCLUDE`) license: **Unverified**. Must verify independently (`docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §2; `docs/architecture/DATA_AND_MODEL.md` §1).
- `SIGNSAGE_MODEL_PROMOTED` remains `false`. No promotion decision made. Any upload is for organizational/research purposes only and does not constitute promotion.
