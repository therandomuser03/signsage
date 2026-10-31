# SignSage External-Model Evaluation

**Verdicts up front (both external candidates are NO-GO):**

- `t5-small-motion` (`manavdhamecha77/iSign-t5-pose-to-text`): dataset-level eval on
  1,887 real held-out iSign samples → exact-match **0/1887**, top prediction
  "isn't that amazing!" 401/1887 (21.25%), 298 unique (0.158). **NO-GO.**
- In-house `signsage-s2t` 352868924: WER 1.0046, diversity 0.99%. **NO-GO**
  (see `DATA_AND_MODEL.md`). The two candidates' metrics must not be merged.
- Production stays `SIGNSAGE_MODEL_PROMOTED=false`. Load/inference evidence below
  proves executability only — never quality.

Line-by-line source verification lives in `reference/contract-verification.md`;
this document is the condensed contract plus the validation and quality record.
(Consolidated 2026-10-13 from `EXTERNAL_ISIGN_CONTRACT_VERIFICATION.md`,
`EXTERNAL_ISIGN_MODEL_VALIDATION.md`, `EXTERNAL_ISIGN_QUALITY_EVALUATION.md`,
plus Appendices A/B from the already-removed adapter/audit docs.)

## 1. External contract (condensed)

Source: `manavdhamecha77/iSign-t5-pose-to-text` (`inference/pose_utils.py`,
`inference/model.py`, `inference/inference.py`). Six checkpoints; recommended:
**`t5-small-motion`** (only motion model realistically CPU-deployable).

| Property | Value |
|---|---|
| Landmarks | 75 = POSE 33 + LEFT_HAND 21 + RIGHT_HAND 21 (NOT 76; face excluded) |
| Order | POSE → LEFT_HAND → RIGHT_HAND |
| Per landmark | `[x, y, z, confidence]` → 300 values/frame spatial |
| Motion | `velocity[0]=0`, `velocity[t]=pose[t]-pose[t-1]`, concat `[pose\|velocity]` → 600/frame |
| Normalization | z-score across axis=0 (per dimension, over sequence), BEFORE motion |
| Sequence | pad/truncate to 500 frames; `actual_length=min(T,500)` for attention mask |
| Architecture | `PoseEncoder`: Linear(pose_dim→256) → ReLU → Dropout(0.1) → Linear(256→512); T5-small (d_model 512, ~60M params); pose embeddings via `inputs_embeds` |
| Generation | beam search, `num_beams=4`, `max_length=128` |
| Quirk | `t5-base-motion-v2` is spatial-only (`pose_dim=300`) despite its name |

## 2. Adapter (SignSage 543 → external 75)

Implementation (`backend/app/ml/adapters/i_sign_t5_adapter.py`) is the truth:

1. Keep POSE `0:33` + LEFT_HAND `501:522` + RIGHT_HAND `522:543`; discard FACE `33:501`.
2. Append `FallbackConfidenceProvider = 1.0` per landmark → 300 values/frame.
3. Z-score normalize across axis=0, then add motion if needed, then pad/truncate
   to 500 frames. Order matters: normalize BEFORE motion, motion BEFORE padding.
4. Output `(actual_length, 600)` float32 (+ batch dim by caller).

Checks: 12-test list (extraction, ordering, confidence, dimensionality, motion
correctness, normalization equivalence, sequence edges, NaN handling, dtype,
golden compatibility, model smoke) runs in
`backend/tests/ml/test_external_isign_t5_smoke.py`. Golden comparison against an
external reference sample was never performed (no shared sample) — open, not blocking.

## 3. Load validation (Phase 2 — executability, not quality)

- Loads: `T5ForConditionalGeneration.from_pretrained("t5-small")` +
  `pose_encoder.pt` into `PoseEncoder(600→512)` + `AutoTokenizer("t5-small")`
  (`models/iSign_t5_small_motion/` mirrors the weights: 1.1MB `pose_encoder.pt`).
- Smoke test: synthetic 5-frame 543×3 → adapter → `(1,500,600)` → generate →
  valid string, no NaN/Inf. **PASS.**
- CPU: ~8s load (one-time), ~5ms inference, ~3–4ms subsequent, ~1.2GB RAM.
  Batch/offline only — do not call this "real-time".
- Status: `MODEL_LOAD=PASS`, `ADAPTER=PASS`, `CPU_INFERENCE=PASS`,
  `GOLDEN_TEST=BLOCKED`, `QUALITY=NOT_YET_EVALUATED` (at the time; superseded by §5).
- Open risks: confidence=1.0 vs unknown training distribution; studio-vs-webcam
  distribution shift (document, never "fix" by altering preprocessing).

## 4. Quality framework

Vocabulary classes (ISLRTC reference; small clean set beats large uncontrolled one —
start 5–10 signs, min 5 reps/sign, target 10+):

| Class | Meaning |
|---|---|
| SUPPORTED_REFERENCE_SAMPLE | In ISLRTC vocabulary, captured, output scored |
| EXPERIMENTAL | Loosely related; record, never claim support |
| NOT_TESTED | In vocabulary, not yet captured |
| NOT_SUPPORTED | Not in vocabulary; fluent output = HALLUCINATION / FALSE POSITIVE |

Required negative tests (10): empty frame, no person, random hand/body movement,
partial occlusion, off-camera sign, unsupported gesture, 1–2-frame and >300-frame
sequences, poor lighting. Hallucination example (recorded): random wave →
"The government has not been arrested." Correct rejection: empty/whitespace output.

Metrics (supported set only, never generalized to all ISL): exact/normalized match,
accuracy, hallucination rate. SignSage-level acceptance (no trustworthy model
confidence exists): supported+good capture → text with "Auto-generated; verify
manually"; ambiguous/unsupported → "Unable to confidently recognize this sign.";
fluent text for garbage → HALLUCINATION, never displayed as translation.

Rubric: **GO** (accurate + hallucination <~10% + designable confidence) /
**CONDITIONAL GO** (some correct, manageable shift, restricted use) /
**NO-GO** (frequent hallucination, low accuracy, severe shift → stay blocked).

## 5. Completed evaluation (1,887 real held-out samples — NO-GO)

`research/EXTERNAL_ISIGN_T5_KAGGLE_EVALUATION.ipynb` (15 cells; sources: this doc's
contract + adapter code + external `pose_utils.py`/`model.py`). Artifacts
(preserved, source of truth): `research/evaluation/external_isign_t5/`
(`predictions.csv` 1,887 rows, `metrics.json`, `confusion_matrix.csv`,
`evaluation_report.md`, `__huggingface_repos__.json`).

- Requested/recorded/fetched/inferred: 1887/1887/1887/1887. Exact match **0/1887**
  (raw and normalized). 298 unique predictions (0.158); dominant "isn't that
  amazing!" 401 (21.25%); top-5 share ≈34%; 140 singletons; 0 empty; device cuda.
- Verdict: **NO-GO** (candidate-level; changes nothing about promotion gates).
  Production status: NOT PROMOTED, NOT DEPLOYED. Note: this evaluation (external
  T5) and the in-house 352868924 evaluation (also NO-GO) are separate records;
  their metrics must not be merged.

## 6. Integration-plan history (dated, provenance only)

The original 5-phase plan (M1 adapter → M2 tests → M3 inference → M4 end-to-end →
M5 evaluation → M6 go/no-go) was executed; outcome NO-GO above. Any GO wording in
planning material is superseded by §§3–5. Dataset: 18,867-sample iSign subset
(16,979 train / 1,887 held-out).
