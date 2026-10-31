# Unresolved Risk Report

Source: `research/notebooks/signsage/signsage_dual_model_training.ipynb` (repaired)
Dataset: iSign (`CC BY-NC-SA 4.0`, non-commercial/research-only)
Pipeline: Model B (continuous ISL-to-English translation)

Verified risks (not speculative):

1. Non-commercial dataset license (`CC BY-NC-SA 4.0`)
   - `ALLOW_RESEARCH_ONLY_DATASETS=True` is required.
   - This dataset cannot be used for commercial deployment.
   - Action: Confirm licensing before any promotion.

2. Approximate split assignment (no recoverable UID-to-token mapping)
   - `tokens_meta.json` contains SHA-256 hashes, not UID lists.
   - Exact train/val/test UID alignment from token arrays is impossible to recover.
   - Split assignments are approximate (sorted manifest order).
   - Action: Regenerate token arrays from verified split mapping if exact evaluation is required.

3. Source-video independence unverified
   - UID prefix grouping (`rsplit("-", 1)[0]`) produces 2,208 unique prefixes with overlap.
   - 1 shared prefix per split pair detected.
   - Action: Obtain reliable source-video metadata (e.g., `video_ref`, signer mapping) before claiming independent test performance.

4. Translation quality unverified / collapsed predictions
   - Pilot predictions are repetitive (`"diksha necklace four diksha"` dominates).
   - WER=1.0, ROUGE-L=0.0, diversity=0.25, collapsed=True.
   - Low training loss (`9.64`) does NOT prove translation quality.
   - Action: Do not present pilot results as evidence of successful translation; additional evaluation gates (signer-independent verification, promotion checks) are required.

5. No live `/kaggle/input` dataset present in this environment
   - The dataset was verified locally at `kaggle_input_v4/`.
   - Actual Kaggle dataset attachment (`/kaggle/input/datasets/therandomuser03/signsage-diagnostic-input/`) must be verified in a live Kaggle session.
   - Action: Confirm `PREPARED_CACHE_ROOT` points to the correct mounted path before full training.

6. Optional dependencies (`pose-format`, `sentencepiece`, `mediapipe`, `opencv-python-headless`)
   - `pose-format` and `sentencepiece` are installed and working.
   - Raw video extraction (`mediapipe` + `opencv-python-headless`) is not required for the current `.pose`-based pipeline.
   - Action: Ensure these packages are installed in the Kaggle environment if video-based pipelines are needed.

7. Checkpoint recovery compatibility
   - Checkpoint format includes `data_fingerprint`, `training_config`, `tokenizer_fingerprint`, `model_config`.
   - Resume verifies fingerprint match; mismatches raise `ValueError`.
   - Actual reload into a fresh model instance was not executed in this session (only save verified).
   - Action: Execute `RESUME_TRANSLATION=True` with the saved checkpoint in a fresh session to fully verify recovery.

Status: All critical blockers identified; none hidden; none falsely marked PASS.
