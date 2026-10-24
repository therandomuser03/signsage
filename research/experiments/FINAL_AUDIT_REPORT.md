# SignSage — Final bounded-diagnostic audit and readiness report
# Produced 2026-10-24. READ-ONLY audit; no archive/repo modifications beyond the focused fix.

A. ARCHIVE STATUS
----------------
Path: /mnt/D/Codes/signsage/results.zip
Size: 11,998,361,485 bytes (~11.2 GiB compressed)
Format: ZIP64; directory listing readable without full extraction (only ZIP r mode used).
Integrity: non-destructive CRC verified; nested pose-cache shards (shard-00000.zip ~3.74 GiB, shard-00001.zip ~3.30 GiB, shard-00002.zip ~3.54 GiB, shard-00003.zip ~2.88 GiB) passed CRC checks in the prior forensic audit.
Contents verified (read-only inspection):
  - s2t_small6/pose_cache/manifest.json (283,669 bytes): version=1, complete=True, n=7500
  - s2t_small6/tokenizer/signsage_en_sp.model (538,842 bytes) -> fingerprint 2d88a5de48318cb9
  - s2t_small6/tokenizer/tokens_meta.json (369 bytes): splits train/val/test fingerprinted; subset=2500 per split
  - s2t_small6/auto_subset.json (210 bytes): subset=2500, seed=42, projected_total_hours=4.99
  - s2t_small6/feasibility.json (1,240 bytes): verdict=PASS, data_mode=cache
  - s2t_small6/checkpoints/signsage_transformer_full.pt (203,649,677 bytes) and recovery.pt (same size, byte-identical to full): both contain the collapsed epoch-1 model (SHA 8043db7d6dab6c7d3f562f00a0e0793766c66784538eebf1f1867c7882a2d9de)
  - CSV: isign/iSign_v1.1.csv (9,808,778 bytes)
No extraction of the full archive was performed in this pass.

B. ARTIFACT INVENTORY (verified paths and sizes from archive inspection)
-----------------------------------------------------------------------
Verified present and readable (archive paths):
  s2t_small6/pose_cache/manifest.json          283,669  (complete, version 1, 7500 uids)
  s2t_small6/pose_cache/shard-00000.zip     4,010,935,738
  s2t_small6/pose_cache/shard-00001.zip     3,534,079,622
  s2t_small6/pose_cache/shard-00002.zip     3,800,526,178
  s2t_small6/pose_cache/shard-00003.zip     3,092,384,110
  s2t_small6/tokenizer/signsage_en_sp.model     538,842  (vocab 16000, fp 2d88a5de48318cb9)
  s2t_small6/tokenizer/signsage_en_sp.vocab     306,259
  s2t_small6/tokenizer/tokens.npz               240,948  (train/val/test ids, 2500 each)
  s2t_small6/tokenizer/tokens_meta.json             369  (split fingerprints verified)
  s2t_small6/tokenizer/train_text.txt           6,085,144
  s2t_small6/auto_subset.json                        210  (subset=2500, seed=42)
  s2t_small6/feasibility.json                        1,240  (PASS, 3.30h projected)
  s2t_small6/checkpoints/signsage_transformer_full.pt  203,649,677
  s2t_small6/checkpoints/signsage_transformer_recovery.pt 203,649,677  (identical to full; epoch-1 collapsed)
Not present / not verified:
  - No deployment-ready bundle (no deployment artifacts for this run).
  - No evaluation report, no export manifest, no fresh non-collapsed checkpoint.
  - No full dataset download performed (not needed for bounded diagnostic).
Checkpoint verdict: CHECKPOINT PRESENT, RESUMABILITY UNVERIFIED. The two .pt files are byte-identical and retain the collapsed epoch-1 state. There is no verified safe-resume proof (no epoch-2 recovery with intact RNG/GradScaler; the forensic snapshot is readable but not an exact-resume checkpoint). A fresh diagnostic must start from newly initialized weights, not from these files.

C. CHECKPOINT VERDICT
--------------------
Classification: CHECKPOINT PRESENT, RESUMABILITY UNVERIFIED.
Evidence:
  - Archive contains two byte-identical .pt files (full and recovery), both 203,649,677 bytes, SHA 8043db7d6dab6c7d3f562f00a0e0793766c66784538eebf1f1867c7882a2d9de.
  - Both contain epoch=1, collapsed state (no diversity-guard acceptance at epoch 1; guard activates from epoch 2).
  - The current working script writes atomic recovery checkpoints (tmp + fsync + replace) and a separate collapsed-epoch snapshot; it also writes initialization_config and training_args into the payload. The archive's older checkpoints lack these fields.
  - Safe structural load (torch.load with weights_only=True on a synthetic payload) succeeds; full resume requires weights_only=False and loads optimizer/scheduler/RNG, which is the intended path but has not been demonstrated to recover from the archive's epoch-1 collapsed state.
What remains unverified: whether resuming from the archive's epoch-1 checkpoint produces a non-collapsed epoch 2. The safe diagnostic path is to start fresh (new out directory, no --resume) rather than rely on this unverified resume.

D. CACHE / TOKENIZER VERDICT
----------------------------
Reusable: YES, with the existing verified assets.
Evidence:
  - Pose-cache manifest: version 1, complete=True, n=7500, seed=42, subset=2500 per split.
  - Token cache meta fingerprints: train/val/test all fingerprint 2d88a5de48318cb9; max_len=96; n=2500 per split.
  - Tokenizer fingerprint: 2d88a5de48318cb9 (sha256 of model bytes + vocab size + special IDs).
  - Cache identity check (manifest uid_to_shard vs expected subset uids) returns ([], []) — exact match.
  - All 7,500 token sequences verified against tokens.npz; no payload extraction performed.
  - Tokenizer vocab size 16000; special IDs (0=PAD, 1=UNK, 2=BOS, 3=EOS) verified.
Mismatches / missing metadata: none verified. The archive does not contain a deployment bundle or fresh non-collapsed checkpoint; those are separate stages.

E. TRAINING FAILURE (verified facts vs hypotheses)
-----------------------------------------------
Verified facts from logs/code/archive:
  - Previous run settings: --lr 1e-4, --epochs 8, --batch 32, --decoder-input-dropout 0.2, --subset 2500, --data-mode cache, --stages tokenizer,overfit,train.
  - Tokenizer and cached token IDs reused successfully; pose cache complete.
  - Overfit gate passed: constrained exact match 96.875%, unconstrained 95.3125% (these are memorization results, not generalization).
  - Real training collapsed at epoch 2: unique_fraction=0.0352 (< 0.05 floor); validation loss 21.4396; top output frequency 62.3%.
  - Gradient clipping activated on 100% of optimizer steps in both epochs.
  - The collapse guard stopped training before the epoch-2 checkpoint/recovery write; the archive's checkpoints are the collapsed epoch-1 state.
  - The archive's tokenizer, token cache fingerprints, sample splits, and pose-cache membership independently matched the CSV and metadata; individual .pose payloads have not been fully decoded and validated.
Likely explanations (explicitly labeled as hypotheses, not proven):
  1. Unstable initial decoder logits due to miscalibrated embedding initialization (supported by synthetic measurement: legacy init logit std 9.34 / CE 36.25 vs scaled 1.90 / CE 11.33 at the real 256-wide, 4-layer, 16000-token architecture; not proof of sole cause).
  2. Gradient clipping on 100% of steps preventing useful updates (observed; clipping norm is pre-clip, after AMP unscaling; Adam's approximate scale invariance does not eliminate the effect).
  3. Too few optimizer steps before the diversity check (supported by schedule: ~158 steps before epoch-2 check at subset 2500 / batch 32).
  4. Weak or incorrect pose-text alignment (not conclusively established; pose-rolled loader and unigram baseline are diagnostic controls, not proof).
Evidence still needed: a real bounded Kaggle run with the corrected initialization, fresh weights, and the corrected diagnostic command (below) to observe whether gradient clipping frequency and output diversity improve.

F. SHORTEST SAFE NEXT ACTION (primary path: bounded diagnostic, not resume)
--------------------------------------------------------------------------
Choose: bounded diagnostic (path 2) — NOT resume from the unverified collapsed checkpoint (path 1 is unsupported by evidence).
Exact corrected command / notebook action (supported by inspected code and artifacts):
  1. Commit and push the verified training script (see G for exact sequence).
  2. Update the notebook's PINNED_SHA and EXPECTED_TRAIN_SCRIPT_SHA256 to the new real commit/hash.
  3. Start a fresh Kaggle session; attach Version 4's saved output (containing s2t_small6/ and isign/iSign_v1.1.csv) as an input dataset.
  4. Do NOT run the full notebook (Run All). Skip the restore/provision/preflight/measure/certify/prepare/training/evaluate/export/persist cells.
  5. Execute the bounded diagnostic cell (provided in the corrected action below) with an external timeout; it uses --pose-source pointing to the first local shard file (not remote), --pose-cache pointing to the full existing cache directory, --data-mode cache, --stages overfit,train, fresh out directory, no --resume, and --embed-init scaled.
  6. Preserve the entire output directory (logs, checkpoints, history, gate reports) via Kaggle's Save Version before ending the session.
Why this is safe:
  - Uses the complete existing pose cache (read-only shards) without rebuilding.
  - Uses the verified tokenizer/token cache (fingerprint 2d88a5de48318cb9) without retraining.
  - Starts fresh weights (no --resume), so the collapsed epoch-1 checkpoint is not loaded.
  - Uses the corrected scaled initialization (verified by synthetic comparison).
  - Applies an external timeout (not relying solely on the soft --max-hours epoch-level check).
  - Does not claim evaluation/export success; only reports completed epochs, diversity, gradient norms, and checkpoint status.
What remains before a larger run: evidence from this bounded run showing reduced clipping frequency, improved diversity, and stable loss trends; then a separate full-dataset training stage with held-out evaluation and manual review.

G. BACKEND READINESS
-------------------
Required artifacts for backend inference (from the deployment bundle design):
  - Trained checkpoint (signsage_transformer_full.pt) with embedded model_config and tokenizer_fingerprint.
  - Tokenizer model (signsage_en_sp.model) with verified special IDs.
  - Landmark contract (landmark_contract.json) defining the 543-landmark ordering and feature-flattening.
  - Deployment manifest (sign_to_text_deployment.json) with architecture, decoding settings, and normalization version.
  - Inference package (inference/) with preprocessing, model, decoder, loader.
Current archive state: the archive contains the collapsed epoch-1 checkpoint but NO deployment bundle or evaluation artifacts. The working repository now contains the corrected training script, the new diagnostic runner (kaggle_diagnostic.py), the new focused test (test_kaggle_diagnostic.py), and the minimal notebook adjustments (mode guard, persistence explanation). The backend bundle must be produced by a successful training run followed by the export stage; it is not present now.
What remains before an inference smoke test: a verified non-collapsed checkpoint from a bounded or full run, plus the exported bundle produced by the export stage.

H. LIMITATIONS
--------------
Verified safely (CPU, no GPU, no download, no full training):
  - Archive integrity and directory listing (read-only ZIP inspection).
  - Tokenizer/token-cache/split fingerprint consistency against the CSV.
  - Pose-cache manifest identity and version match.
  - Script hash/current file identity; symlink target consistency.
  - Initialization mechanics (scaled vs legacy) and restricted checkpoint reload.
  - Isolated collapse-guard behavior (preserves previous checkpoints, writes readable snapshot, raises failure).
  - Parser compatibility of the proposed diagnostic command.
Not verified (requires real Kaggle GPU session):
  - Whether the corrected initialization actually reduces gradient clipping frequency in real training.
  - Whether output diversity improves above the 0.05 floor at epoch 2.
  - Whether the bounded diagnostic completes two full epochs within the 24-minute external timeout.
  - Whether resuming from a future non-collapsed recovery checkpoint works correctly (only structural reload verified).
  - Whether the full dataset (123,779 rows) trains without new failure modes.
  - Whether the backend bundle produced by a future export stage is fully compatible with the corrected initialization and new script version.
Unresolved hypotheses (not proven by this audit):
  - The scaled initialization is the sole or primary cause of the previous collapse.
  - The unigram baseline comparison (approximate due to batch-mean vs token-weighted averaging) proves pose use.
  - The pose-rolled loader's positive gap proves pose dependence (it is a control, not proof).
Not executed: full training, GPU initialization, dataset download, Kaggle session start, commit/push, notebook pin update, or backend promotion.

I. EVIDENCE REFERENCES (exact paths and lines)
---------------------------------------------
- Archive inspection: results.zip (11,998,361,485 bytes); manifest/version/n verified; tokenizer fingerprint verified; token sequences verified; no payload extraction performed.
- Script: research/experiments/train_sign_to_text.py (current working hash 213926c2599dc2bc319e012893b4463d301060c7efe19167f290420724c5ba5c; historical reference 1dbea95287eb6f1c3f7eea918c8f138673db3d740d66d7a186f3d2166b018732). Key locations: init_token_embedding (:889), build_model (:2037), resume/checkpoint load (:2247), collapse guard/snapshot (:2337), benchmark isolation (:939, :1901), token_weighted option (:1432), PoseRolledLoader (:1415), check_checkpoint_not_collapsed (:889).
- Backend model: backend/app/ml/models/sign_to_text.py (:85–105): independent output projection, no weight tying, scale=d_model**0.5, padding_idx preserved.
- Notebook adjustments: research/notebooks/signsage_kaggle.ipynb (new cells: diag-mode-config, persist-explanation; training cell fb3bffaf: mode guard inserted; clone cell: sequence comment added; final cell: mode reference updated). Notebook pin remains 4a91f029ecc7a95f4eb970eccfbb7abfccae2a1d; expected script hash remains 1531bb28ddc6e1ea79738582e08c2118999297172ce93dfada5f7faab0bdff3b (stale relative to working file).
- New files (uncommitted): research/experiments/kaggle_diagnostic.py, research/experiments/test_kaggle_diagnostic.py.
- Tests executed (CPU only, no optimizer steps): test_collapse_fix.py (12 passed, 1 deselected synthetic training), test_pipeline_hardening.py (atomic save, killed write, cache refusal, valid cases — all passed), test_notebook_final_structure.py::test_checkout_cell_pins_commit_and_script_content_and_matches_the_repo_script (passed — confirms old pin/hash pairing is intact; expected failure when comparing to new file hash), synthetic initialization comparison (passed), restricted checkpoint reload (passed), isolated guard execution (passed).
- No modifications to results.zip, no dataset download, no GPU initialization, no full training, no commit/push performed.

J. BLOCKERS (must be resolved before a real bounded Kaggle run)
---------------------------------------------------------------
1. The working script contains the uncommitted fix; the notebook's PINNED_SHA and EXPECTED_TRAIN_SCRIPT_SHA256 are stale. Before any real Kaggle run, commit and push the verified code, then update both constants to the real new commit/hash.
2. The proposed corrected command uses `--pose-source` pointing to a local shard file (not remote) to avoid HF authentication/network reads; confirm the mounted input directory layout matches this assumption (ordinary `s2t_small6/pose_cache/` + `isign/iSign_v1.1.csv`).
3. The external timeout mechanism (GNU timeout) must be available in the Kaggle environment; verify before launch.
4. The bounded diagnostic does not save evaluation metrics or a deployment bundle; those remain separate stages requiring sufficient GPU time and manual review.
5. The archive's collapsed checkpoint cannot be safely resumed; the diagnostic uses fresh weights. Any claim of resumed training requires a separate, verified non-collapsed recovery checkpoint.

K. EXACT CORRECTED KAGGLE ACTION (copyable; does not start full training)
-----------------------------------------------------------------------
Prerequisites (before running):
  - Commit/push verified code; obtain real HEAD SHA-256; update PINNED_SHA and EXPECTED_TRAIN_SCRIPT_SHA256 in the notebook clone/check cell.
  - Attach Version 4's saved output (containing s2t_small6/ and isign/iSign_v1.1.csv) as an input dataset.
  - Confirm GPU accelerator is enabled only when you intend to execute the diagnostic.

Single bounded diagnostic cell (run individually, NOT Run All):
  See research/experiments/kaggle_diagnostic.py. Key invocation:
    python3 -m signsage.research.experiments.kaggle_diagnostic \
        --input-root /kaggle/input \
        --out /kaggle/working/s2t_diag1 \
        --timeout-seconds 1440 \
        --expected-tokenizer-fingerprint 2d88a5de48318cb9
  This validates CSV/cache/tokenizer identity, uses the complete existing pose cache read-only, starts fresh weights (`--embed-init scaled`), runs only `--stages overfit,train`, applies an external 24-minute timeout, and writes logs/checkpoints/history/gate reports. It does not evaluate, export, or claim quality success.

Expected outputs (preserve before ending session):
  /kaggle/working/s2t_diag1/subprocess_stdout_stderr.log
  /kaggle/working/s2t_diag1/pose_preflight.json
  /kaggle/working/s2t_diag1/gate_report.json (if overfit completes)
  /kaggle/working/s2t_diag1/training_history.json (after completed epochs)
  /kaggle/working/s2t_diag1/checkpoints/signsage_transformer_recovery.pt (after accepted epoch)
  /kaggle/working/s2t_diag1/checkpoints/signsage_transformer_full.pt (if best improves)
  /kaggle/working/s2t_diag1/checkpoints/signsage_transformer_collapsed_epochNN.pt (if diversity guard fires)
  /kaggle/working/s2t_diag1/COLLAPSED.txt (if guard fires)
  /kaggle/working/s2t_diag1/diagnostic_config.json

L. FILES TO COMMIT / PUSH (exact sequence)
------------------------------------------
1. research/experiments/train_sign_to_text.py (verified fix; current working hash 213926c2599dc2bc319e012893b4463d301060c7efe19167f290420724c5ba5c)
2. research/experiments/test_collapse_fix.py (new regression tests)
3. research/experiments/kaggle_diagnostic.py (new bounded diagnostic runner)
4. research/experiments/test_kaggle_diagnostic.py (minimal non-training tests)
5. research/notebooks/signsage_kaggle.ipynb (only after step 1 is pushed; update PINNED_SHA and EXPECTED_TRAIN_SCRIPT_SHA256 to the real new commit/hash; add mode guard, persistence explanation, clone sequence comment)

Exact sequence (do NOT invent future SHAs):
  git add -- research/experiments/train_sign_to_text.py research/experiments/test_collapse_fix.py research/experiments/kaggle_diagnostic.py research/experiments/test_kaggle_diagnostic.py
  GIT_EDITOR=true git commit -m "fix: calibrate decoder embedding initialization; add bounded diagnostic runner and focused tests"
  git push origin master
  NEW_SHA=$(git rev-parse HEAD)
  NEW_SCRIPT_SHA=$(sha256sum research/training/train_sign_to_text.py | awk '{print $1}')
  # Then edit the notebook clone/check cell (index 3 / id 2773fd82) and set:
  #   PINNED_SHA = "${NEW_SHA}"
  #   EXPECTED_TRAIN_SCRIPT_SHA256 = "${NEW_SCRIPT_SHA}"
  # Then commit/push the notebook separately.

M. PRE-FLIGHT CHECKLIST (before any real bounded Kaggle run)
----------------------------------------------------------
- [ ] Verified code committed and pushed; real HEAD SHA obtained.
- [ ] Notebook PINNED_SHA and EXPECTED_TRAIN_SCRIPT_SHA256 updated to the real new values (not fabricated).
- [ ] Notebook clone/check cell verifies the exact commit and script hash; fails closed on mismatch.
- [ ] Notebook execution mode set to "diagnostic" (default); full mode only selected intentionally.
- [ ] Version 4 saved output attached as input; mounted CSV and cache paths verified by the diagnostic cell.
- [ ] Fresh output directory (`/kaggle/working/s2t_diag1`) does not exist; no old checkpoints copied.
- [ ] No `--resume` passed; no old collapsed checkpoint loaded.
- [ ] Tokenizer fingerprint (`2d88a5de48318cb9`) and split fingerprints (`seed=42`, `subset=2500`) verified.
- [ ] External timeout mechanism (`timeout` binary) available; working disk sufficient for checkpoints.
- [ ] GPU enabled only when intended; CPU-only checks completed.
- [ ] After run: preserve `s2t_diag1/` via Save Version; inspect `training_history.json`, `gate_report.json`, gradient norms, `unique_fraction`, and top-output frequency before any larger run.
- [ ] Do NOT describe a passing synthetic smoke test, overfit gate, or bounded diagnostic as evidence of generalization or production readiness.

N. STOP CONDITION
-----------------
This is the final focused pass. No additional broad audits, no automatic repair loops, no full training, no GPU-heavy execution, no dataset download, no commit/push performed by this agent, and no production promotion. The repository changes are minimal and auditable; the user must commit/push and then execute the bounded diagnostic manually.
