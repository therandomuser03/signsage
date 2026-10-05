# SignSage Model Readiness (read-only)

Question answered: can any current model actually serve through the current backend? Answer: no — and that is the gate working as designed.

## 1. Checkpoint inventory

| Checkpoint | Metrics (measured, not claimed) | Promotion flag | Backend load | Verdict |
|------------|---------------------------------|----------------|--------------|---------|
| 352868924 `signsage_transformer_full.pt` (+`.latest`, `recovery`, distinct CRCs) | WER 1.0046, BLEU 0.24, ROUGE-L 0.089, EM 0.0, diversity 0.99%, top-pred 44.8% (`research/manifests/models/signsage-s2t_v352868924.yaml`, `research/evaluation/reports/352868924/`) | `promoted_to_backend:false`, `status:verified_no_go` | Loads (`bundle_loads:true`), deterministically reproduces collapse ("the police said that") | BROKEN as translator / WORKING as negative reference. Must stay unpromoted. |
| SignTransformer V2 (planned, `research/training/READINESS.md`: pre-norm, d256/4L, 16k SP, 12ep) | None — training not started | n/a | n/a | EXPERIMENTAL |
| T5-small POC (`research/t5_pose_poc/`, `google-t5/t5-small`, 61,605,888 params) | Synthetic-gate only: EM 0.0 → 0.09375 vs 0.5 threshold → `WIRING-FAIL` (`t5_poc_gate/gate_report.json`), correctly blocking any 5k run. No real-gate, no 5k, no `best.pt`. | n/a (research-only, never serving per its README) | Cannot load via `SignToTextAdapter` (Transformer arch + 32k vocab vs 16k SP head) | EXPERIMENTAL, currently failing closed as designed |

## 2. Tokenizer compatibility (architectural, not a bug)

- Serving track: SentencePiece unigram 16k, `PAD0/UNK1/BOS2/EOS3`, trained on training targets (`research/training/train_sign_to_text.py:321,342,701`). Artifact preserved (`research/experiments/sign-to-text/352868924/.../signsage_en_sp.model`). Backend `decode_token_ids` + adapter load enforce exact special-id match. COMPATIBLE.
- T5 track: `google-t5/t5-small` 32k (`pad0/eos1/unk2`), required to keep pretrained language knowledge (`research/t5_pose_poc/README.md:14-20`). Checkpoint records `tokenizer: google-t5/t5-small`. INCOMPATIBLE with the serving track by design. A T5 model would need its own adapter, contract, and export format — none exist, none are planned while SignTransformer is the production owner.

## 3. Evaluation gates that must stay

-Kept from the NO-GO post-mortem: WER/BLEU-1..4/ROUGE-L/EM, trivial (mode-copy) baseline with `beats_trivial_bleu4`, diversity (`unique_fraction`, `collapsed<0.05` STOP), sensitivity (swap/zero/shuffle → `pose_conditioned`), qualitative examples, export/load verification (`scripts/model_check.py` zero-motion-must-reject, `backend/models/README` bar: `transformer_beats_trivial AND diversity_ok`).
- T5 track mirrors them in `research/t5_pose_poc/evaluate.py:53-147` plus a binding 64-sample overfit gate (EM ≥ 0.95 or hard STOP before any 5k).
- Rule: loss decrease, fluent text, model load, or overfit-gate pass alone never constitute success.

## 4. What "ready" would require (no fabrication)

A promotable model needs, in order: (a) real trained weights from either track beating the trivial baseline with diversity_ok on held-out data; (b) `export_bundle` producing `checkpoint/ + tokenizer/ + landmark_contract.json + sign_to_text_deployment.json`; (c) `scripts/model_check.py` pass; (d) `BUNDLE_DIR` pointed + `PROMOTED=true` + adapter integration tests green. None of (a)–(d) exist today. The honest backend state is `MODEL_UNAVAILABLE`, and the frontend already renders it.
