# SignSage Phase 1–2 Verification — 2026-10-30

## Phase 1 — Signer metadata (read-only)

- CSV inspected (`kaggle_input_v4/isign/iSign_v1.1.csv`): exactly `uid,text`. No signer column.
- Audit (`staging_kaggle_results/s2t_v2/dataset_audit.json`): `signer_column: null`; note says "signer-independence assumed from video-level split only; confirm on the training host."
- Official web sources checked read-only:
  - HF dataset card (`Exploration-Lab/iSign`): CSV format `uid,text`; split by `video_id`; no signer IDs documented.
  - Project site (`exploration-lab.github.io/iSign/`): validation by certified ISL signers on 593 pairs; dataset released as video+translation, no signer identifier.
  - arXiv abstract (`2407.05404`): no signer-ID release mentioned.
- **Conclusions (distinguished, not conflated):**
  1. No signer column in inspected CSV — confirmed.
  2. No signer metadata in inspected files — confirmed (audit, CSV, tokenizer meta, website, dataset card).
  3. Evidence that signer metadata does not exist anywhere — NOT established. Full paper / pose archive / gated dataset not downloaded; authors may hold signer IDs internally. Do NOT claim signer independence or signer-ID absence globally.
- Remaining uncertainty: signer-independent evaluation cannot be established from public CSV; must remain documented risk (per readiness report §7).

## Phase 2 — Pilot preflight (verified, not guessed)

- Subset: `kaggle_input_v4/s2t_small6/` exists; `auto_subset.json` = 2500/split; `feasibility.json` verdict PASS; `pose_cache/manifest.json` complete, n=7500; tokenizer fingerprint `2d88a5de48318cb9` verified.
- Pairing: deterministic by stable UID; CSV split seed 42 matches subset seed 42; pose_cache `uid_to_shard` maps to tokenizer `tokens_meta.json` splits.
- Targets: `tokenizer/train_text.txt` shows multi-word sentences (e.g., "one day, akbar drew a line on the floor and ordered."); dataset audit mean 10.86 words, median 11, p90 18; 2.1% one-word rows (not isolated-word-only). Suitable for sentence-level validation.
- Pose arrays: loaded `.pose` files with `pose_format` — float32, finite, multi-frame (93 frames in sampled UID `ee235596b76f-2`), compatible with pipeline contract.
- Split consistency: audit split (train 98,485 / val 13,014 / test 12,280) + subset 2500/split = 7500 total; manifest n=7500. Consistent.
- Trainer / diagnostic: `research/experiments/train_sign_to_text.py` and `kaggle_diagnostic.py` present; `args_output.txt` confirms CLI args.
- CLI support verified: `--embed-init scaled`, `--stages overfit,train`, `--resume` false (diagnostic asserts), `--out` fresh dir, `--timeout-seconds` 1440 = 24 min; `--max-hours 0.4` = 24 min graceful stop.
- Side-effect guard: diagnostic stages `overfit,train` only; no evaluate/export in `run_args`; writes to unique `out`; does not touch `results.zip`, backend model, or `SIGNSAGE_MODEL_PROMOTED` (remains `false`).
- Disk/deps: 78G free; venv `backend/.venv` activated (torch 2.14.0+cu130, sentencepiece ok); pose_cache local read-only.
- Blocker status: NONE critical. Subset supervision is sentence-level; pairing verified; all safety checks pass.

## Command prepared (not yet executed at time of writing)

```bash
mkdir -p results
OUT="results/diag_small6_$(date +%s)"
source backend/.venv/bin/activate
timeout 1440 python research/experiments/kaggle_diagnostic.py \
  --input-root ./kaggle_input_v4 --out "$OUT" \
  --timeout-seconds 1440 --expected-tokenizer-fingerprint 2d88a5de48318cb9
```

~24 min hard timeout; exit 124 = timeout (not pass); outputs preserved under `$OUT`.
