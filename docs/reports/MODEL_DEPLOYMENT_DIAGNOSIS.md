# SignSage — Model & Deployment-Gate Diagnosis

Date: 2026-10-12. Method: code paths read directly, settings executed,
archive metadata listed, checkpoint bytes inspected read-only in `/tmp`
(removed afterwards). `results.zip` was never modified. No training, no
promotion change, no commits.

## 1. Current status: why live sign-to-text is unavailable

Every sign-to-text request is refused **before any model code runs** because
`SIGNSAGE_MODEL_PROMOTED` is `false` (code default and local env agree).
The gate is holding correctly: no existing artifact satisfies the promotion
criteria, the configured bundle directory does not exist in this checkout,
and the only loadable checkpoint is a collapsed, unexported training snapshot
the serving loader would reject anyway. Three independent layers agree;
fixing any one would still leave the other two refusing.

## 2. Execution trace: frontend → decoded output

1. Frontend opens `GET ws://…/ws/translate`, sends `landmarks.frame`
   (`[x,y,z]` × 543) batches (`frontend/.../websocket/client.ts`,
   `TranslatePage.tsx`).
2. `backend/app/api/websocket/translation.py::translate_ws` accepts, then
   calls `build_translation_service(settings)`
   (`backend/app/core/dependencies.py`).
3. `build_translation_service` calls `get_sign_to_text_pipeline(settings,
   allow_unpromoted=settings.allow_unpromoted_dev)`
   (`backend/app/ml/models/registry.py`).
4. **Refusal happens here:** `if not settings.model_promoted and not
   allow_unpromoted: raise ModelNotPromotedError(...)`. The WS handler catches
   `Exception`, logs `[MODEL] translation unavailable`, sends
   `{"type":"error","code":"MODEL_UNAVAILABLE",…}` and closes normally.
   REST `POST /api/v1/translation/sign-to-text` fails the same way with
   503 `MODEL_NOT_PROMOTED`. Capabilities reports
   `sign_to_text_available:false, reason:model_not_promoted`.
5. Never reached: `resolve_bundle_dir` (requires `checkpoint/`, `tokenizer/`,
   `landmark_contract.json`, `sign_to_text_deployment.json`;
   `backend/app/infrastructure/storage/model_storage.py`) →
   `SignToTextAdapter.load` (checkpoint↔contract↔tokenizer cross-checks;
   `backend/app/ml/adapters/sign_to_text_adapter.py`) → `transcribe`
   (`(T,543,3)` → `prepare_live_landmarks` → `greedy_decode_ids` →
   `decode_token_ids` → acceptance gate).

## 3. Artifact inventory (all read-only; archive listing + `unzip -p`)

- `results.zip` (415 entries): only SignTransformer weights are
  `s2t_v2/checkpoints/signsage_transformer_full.pt` + `..._recovery.pt`
  (~203 MB each, plus identical prior-run copies). No 352868924 `.pt` and no
  `signsage_deployment_bundle.zip` are in this archive.
- `s2t_v2/COLLAPSED.txt`: `epoch 2: unique_fraction=0.0195 < 0.05;
  val_loss=22.8017; top='the she more assumed mouse disgrace and' (70.3%)` —
  the diversity-collapse guard fired at epoch 2; training stopped.
- `s2t_v2/gate_report.json`: a 64-sample overfit gate (`passed:true`,
  exact 0.953) — a *training* gate, not the deployment promotion gate; the
  same run still collapsed one epoch later.
- `s2t_v2/tokenizer/`: `signsage_en_sp.model` (539 KB) + `.vocab`,
  `tokens_meta.json`, `tokens.npz` — tokenizer pieces only, no
  `landmark_contract.json` / `sign_to_text_deployment.json`, so no loadable
  deployment bundle exists for s2t_v2.
- `research/manifests/models/signsage-s2t_v352868924.yaml` + the in-zip
  `signsage/research/evaluation/reports/352868924/evaluation_results.json`
  (352868924's own record): WER 1.0046, BLEU 0.2429 (worse than its trivial
  baseline BLEU 0.3964), ROUGE-L 0.089, EM 0.0, diversity 0.99%,
  top-prediction 44.84% → `transformer_beats_trivial:true` BUT
  `output_diversity_ok:false` → `phase_0_status: NO-GO`.
- Promotion rule (`backend/models/README.md`): promote only on
  `transformer_beats_trivial AND output_diversity_ok`. 352868924 fails the
  second conjunct; s2t_v2 never produced an evaluation at all.
- Configured bundle dir (`../research/artifacts/352868924/
  signsage_deployment_bundle`, `research/artifacts/` is gitignored):
  **does not exist** in this checkout.

## 4. Root cause

The failing condition is `not settings.model_promoted and not
allow_unpromoted` in `get_sign_to_text_pipeline`
(`backend/app/ml/models/registry.py`), with both values `False` (code
default in `backend/app/core/config.py: model_promoted: bool = False`,
`allow_unpromoted_dev: bool = False`; live settings printed 2026-10-12
confirm `False`/`False`). Live evidence: `GET /health` →
`{model_promoted:false, model_bundle_found:false, model_loaded:false}`;
`POST /translation/sign-to-text` → 503 `MODEL_NOT_PROMOTED`; WS first frame
→ `MODEL_UNAVAILABLE` then clean close. The gate is **correctly preventing**
an unverified model from being served — 352868924 is its own author's NO-GO
and s2t_v2 is a guard-stopped collapse.

## 5. Smoke-test results (bounded, `/tmp` only, cleaned up)

- Extracted `s2t_v2/.../signsage_transformer_full.pt` + `signsage_en_sp.model`
  to `/tmp` (nothing else; archive untouched).
- `torch.load` (trusted source: the user's own training output): payload keys
  `[bad, best, epoch, history, optimizer_state_dict, rng, scheduler_state_dict,
  state_dict, val_loss]`; `epoch: 1`; `model_config: {}` (empty);
  no `vocab`; no `tokenizer_fingerprint`; 135 tensors / 17,001,216 params;
  shapes show `input_proj … (384, 3258)`, `tok_embed (16000, 256)`,
  256-dim attention — the right model *family* (3258-dim input matches the
  serving contract's `model_input_dim`).
- **Load via serving path: FAILS deterministically at validation, before any
  weight math** — `SignToTextAdapter.load` raises `ModelLoadError("Checkpoint
  is missing 'state_dict'/'model_config'")` on the empty `model_config`, and
  on this env's torch the raw `torch.load` itself first requires
  `weights_only=False` (numpy globals; the adapter passes no such flag).
  This is a raw trainer snapshot (optimizer/rng/scheduler state included),
  never an `export_bundle` product.
- No decode was attempted: without an exported config, special-token binding,
  and contract, running the weights would mean inventing the serving
  parameters — a fabricated test, not a smoke test. Expected output if forced:
  epoch-1 collapse gibberish consistent with `COLLAPSED.txt` (70.3% top).
- **Could not be performed:** real-input transcription through the serving
  stack (no loadable bundle exists); 352868924-bundle load (its `.pt` is not
  in this checkout's artifacts).

## 6. Minimal fix

**No safe fix is currently supported — deliberately no code changed.**
Setting `SIGNSAGE_MODEL_PROMOTED=true` would be unjustified (no run passes
both promotion conjuncts) and still insufficient (bundle dir absent; the only
extractable checkpoint is unloadable by the serving validator). The smallest
*legitimate* path to a working demo remains the documented one: a training
run that beats-trivial **and** passes diversity → `export_bundle` →
`model_check.py` → point `SIGNSAGE_MODEL_BUNDLE_DIR` at it → promote. That is
future training work, out of scope for this task by explicit constraint.

## 7. Verification (all executed 2026-10-12)

- `pytest tests/integration/test_websocket_auth_and_limits.py tests/contract -q`
  → **37 passed** (gate/contract behavior, incl. unpromoted-model handling).
- Live TestClient probe: health 200 unpromoted/unfound/unloaded; capabilities
  `available:false/model_not_promoted`; REST 503 `MODEL_NOT_PROMOTED`; WS first
  frame `MODEL_UNAVAILABLE`. Matches §4 exactly.
- Checkpoint byte inspection as in §5; `/tmp/s2t_smoke` removed afterwards;
  `git status` shows no artifact or code modifications from this task.

## 8. Remaining blockers for a genuine sign-to-English demo

1. A checkpoint that passes **both** promotion conjuncts (none exists).
2. An exported deployment bundle (contracts + tokenizer + evaluation) for it.
3. A materialized, loadable bundle path wired to `SIGNSAGE_MODEL_BUNDLE_DIR`.
4. Only then: promotion flag + `model_check.py` + live end-to-end verification.
