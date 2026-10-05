# External iSign Quality Evaluation

## 1. Objective

Evaluate whether the `t5-small-motion` checkpoint from `manavdhamecha77/iSign-t5-pose-to-text`
produces sufficiently reliable translations from real SignSage MediaPipe captures to justify
integration into the application. This evaluation uses a controlled vocabulary based on the
ISLRTC reference material and a small, repeatable capture protocol.

**The model MUST remain blocked for production promotion throughout this evaluation:**
`SIGNSAGE_MODEL_PROMOTED=false`

## 2. Model Under Test

- **Checkpoint**: `t5-small-motion` from `manavdhamecha77/iSign-t5-pose-to-text` (Hugging Face)
- **Model architecture**: PoseEncoder (Linear 600→256→ReLU→Dropout 0.1→Linear 256→512) + T5-small decoder
- **Input contract**: (N, 600) = [pose(300) | velocity(300)], 75 landmarks × [x, y, z, confidence=1.0]
  - Z-score normalization across axis=0, before motion addition
  - Motion: velocity[t] = pose[t] - pose[t-1], velocity[0] = 0
  - Sequence padded/truncated to 500 frames
- **Preprocessing**: SignSage iSign adapter (`backend/app/ml/adapters/i_sign_t5_adapter.py`)
  - Filters 543 landmarks → 75 (POSE + LEFT_HAND + RIGHT_HAND)
  - Appends confidence=1.0 per landmark
  - Z-score normalization across sequence axis
  - Adds velocity features if motion model
  - Pads/truncates to 500 frames
- **Tokenizer**: `t5-small` (English)
- **Generation**: beam search, num_beams=4, early_stopping=True, max_length=128
- **Hardware**: CPU only (verified ~5ms inference)

**DO NOT MODIFY:**
- Checkpoint weights or architecture
- Adapter preprocessing contract
- Tokenizer or vocabulary
- Generation configuration

## 3. Evaluation Vocabulary

The evaluation vocabulary is based on the ISLRTC (Indian Sign Language Research and Training Centre)
reference material already documented in the project. The vocabulary is intentionally limited.

### Classifications

| Classification | Definition |
|---|---|
| **SUPPORTED_REFERENCE_SAMPLE** | Sign is in the ISLRTC reference vocabulary; captured on camera; model output evaluated against expected label |
| **EXPERIMENTAL** | Sign is loosely related but not explicitly in the reference vocabulary; output recorded but not claimed as supported |
| **NOT_TESTED** | Sign is in the reference vocabulary but has not yet been captured/evaluated |
| **NOT_SUPPORTED** | Sign is NOT in the ISLRTC reference vocabulary; model generating fluent text for unsupported input is treated as HALLUCINATION / FALSE POSITIVE |

### Reference Vocabulary (Development Pending Capture)

This section will be populated as signs are captured. Do not add entries until captured via the
protocol in Section 4. The initial focus is on common, clearly identifiable signs that can be
performed within a standard webcam frame using the 75-landmark subset.

**Do not invent vocabulary that is not supported by the reference material.**

## 4. Capture Protocol

A simple, repeatable capture protocol for each selected sign:

1. Start camera.
2. Wait until MediaPipe detects the signer (person in frame).
3. Perform the sign naturally — begin and end within the camera view.
4. Capture the complete sequence from start to finish.
5. Stop capture (stop sending frames; do not continue indefinitely).
6. Send the landmark sequence to the adapter.
7. Run `t5-small-motion` inference.
8. Record the generated output and metadata.

### Repetitions

- **Minimum**: 5 repetitions per sign
- **Target**: 10+ repetitions per sign if practical
- Fewer signs with more repetitions is preferred over many signs with few repetitions

### Per-Sign Capture Details

For each sign, record:
- **sample_id**: unique identifier (e.g., `sign_hello`, `sign_thankyou`)
- **expected_text**: the ISLRTC reference label (e.g., `"Hello"`, `"Thank you"`)
- **num_frames**: number of frames captured (should be ≤ 500 after adapter padding)
- **adapter_input_shape**: shape after adapter preprocessing (expected: (500, 600))
- **model_checkpoint**: `t5-small-motion`
- **generated_output**: text produced by the model
- **inference_latency**: time from adapter output to generated text (seconds)
- **classification**: SUPPORTED_REFERENCE_SAMPLE, EXPERIMENTAL, NOT_TESTED, NOT_SUPPORTED
- **hallucination**: True if model generates fluent text for unsupported/noise input
- **notes**: any observations (lighting, framing, hand visibility, etc.)

## 4.1 Temporal Window Recommendation

Investigate whether capture should have a defined start/stop window rather than continuously
sending arbitrary camera frames. The model may behave differently with:

- Short sequences (5-20 frames)
- Natural-length sequences (full sign, typically 50-150 frames)
- Sequences padded heavily to 500 frames
- Sequences truncated to 500 frames

**Do not change the external preprocessing contract.** This is evaluation only.

**Document the finding** in the final report: whether a defined capture window improves
consistency and reduces padding/truncation artifacts.

## 5. Positive Samples (Supported Reference Signs)

For each supported sign, perform 5+ repetitions and record results.

### Expected format per sign:

| Field | Description |
|---|---|
| sample_id | e.g., `sign_hello` |
| expected_text | ISLRTC reference label, e.g., `"Hello"` |
| repetitions | number of times the sign was performed (5+) |
| generated_output_i | model output for repetition i |
| exact_match_i | True if generated == expected (case-insensitive) |
| normalized_match_i | True if generated normalizes to expected |
| latency_i | inference time for repetition i |
| notes_i | observations (framing, lighting, etc.) |

### Example (populated after evaluation):

| sample_id | expected_text | repetitions | generated_output | exact_match | normalized_match | latency | notes |
|---|---|---|---|---|---|---|---|
| sign_hello | Hello | 5 | "Hello" | ✅ | ✅ | 4.2ms | Good lighting, centered |
| sign_hello | Hello | 5 | "Hi" | ❌ | ❌ | 3.8ms | — |
| sign_hello | Hello | 5 | "Hey" | ❌ | ❌ | 4.1ms | — |

### Accuracy calculation (supported references only):

```
accuracy = correct_exact_matches / total_evaluated_samples
```

Report this only for the explicitly defined evaluation set. **Do not generalize it to all ISL.**

## 6. Negative / Unsupported Tests

**This is CRITICAL** because SignSage must never hallucinate a translation.

For each negative test, record what the model generates. If the model generates fluent but
unrelated text, mark: **HALLUCINATION / FALSE POSITIVE** — this is a failure.

### Required negative tests:

| Test # | Description | Expected model behavior | Classification |
|---|---|---|---|
| 1 | Empty camera frame (no person) | Model output recorded | NOT_SUPPORTED / HALLUCINATION |
| 2 | No person detected | Model output recorded | NOT_SUPPORTED / HALLUCINATION |
| 3 | Random hand movement (no structured sign) | Model output recorded | NOT_SUPPORTED / HALLUCINATION |
| 4 | Random body movement | Model output recorded | NOT_SUPPORTED / HALLUCINATION |
| 5 | Partial hand visibility (one hand occluded) | Model output recorded | EXPERIMENTAL / NOT_SUPPORTED |
| 6 | Sign performed outside camera view | Model output recorded | NOT_SUPPORTED |
| 7 | Unsupported gesture (not in vocabulary) | Model output recorded | NOT_SUPPORTED / HALLUCINATION |
| 8 | Very short sequence (1-2 frames) | Model output recorded | EXPERIMENTAL |
| 9 | Very long sequence (>300 frames, heavily padded) | Model output recorded | EXPERIMENTAL |
| 10 | Poor lighting / degraded capture (if practical) | Model output recorded | EXPERIMENTAL |

### Hallucination definition:

If the model generates a fluent English sentence for random movement, no person, or an
unsupported gesture, this is a **HALLUCINATION / FALSE POSITIVE**. The system must not
treat such output as a successful prediction.

**Example of hallucination:**

- Input: Random hand wave (no person structure)
- Generated: `"The government has not been arrested."`
- Result: **INCORRECT / HALLUCINATION** — not a successful translation

**Example of correct rejection:**

- Input: Random hand wave
- Generated: `""` (empty string) or `" "` (whitespace)
- Result: **Correct** — model did not produce a misleading translation

## 7. Quantitative Results

For samples with known expected text, calculate where meaningful:

| Metric | Definition | When to Report |
|---|---|---|
| Exact Match | generated_text (case-insensitive) == expected_text | For all supported reference samples |
| Normalized Match | Normalized version of generated matches expected | Where appropriate (e.g., punctuation, whitespace differences ignored) |
| Accuracy | correct_exact_matches / total_evaluated_samples | Only for the explicitly defined evaluation set |
| Hallucination Rate | % of negative/unsupported tests that produce fluent but unrelated text | Always report for negative tests |

**BUT only report accuracy for the explicitly defined evaluation set. Do not generalize it to all ISL.**

### Per-sign results table example:

| sign_id | expected | repetitions | correct | exact_match_rate | hallucination_rate | avg_latency |
|---|---|---|---|---|---|---|
| sign_hello | Hello | 5 | 3 | 60% | 0% (negative tests only) | 4.1ms |
| sign_thankyou | Thank you | 5 | 2 | 40% | 1/3 (random movement) | 3.9ms |

## 8. False Positives / Hallucinations

Document all instances where the model produces fluent but unrelated text.

### Classification:

| Outcome | Classification | Action |
|---|---|---|
| Model generates expected text | SUPPORTED_REFERENCE_SAMPLE | Record as correct |
| Model generates different but related text | EXPERIMENTAL | Record; do not claim support |
| Model generates fluent text for unsupported input | HALLUCINATION / FALSE POSITIVE | **Failure** — must be explicitly reported |
| Model generates empty/whitespace output | Correct rejection | Record as valid behavior |
| Model crashes/error | ERROR | Record; does not count as translation |

### SignSage-level acceptance mechanism:

If there is no defensible confidence mechanism yet (the model itself may generate text even
when input is garbage), the system must prefer:

> "Unable to confidently recognize this sign."

over an unrelated fluent sentence.

**Never fabricate a translation for: UNCERTAIN or UNSUPPORTED.**

## 9. Confidence / Uncertainty

### External model confidence

The `t5-small-motion` model itself does not expose meaningful confidence/log-probability
signals that should be treated as trustworthy. Generated text does not equal correct
recognition.

### SignSage-level acceptance

Since the model may generate fluent text for garbage input, a conservative SignSage-level
mechanism is required:

| Condition | System Response |
|---|---|
| Input is a supported sign with good capture | Display generated text with disclaimer: "Auto-generated; verify manually" |
| Input is ambiguous or capture quality is poor | Display: "Unable to confidently recognize this sign." |
| Input is unsupported (not in vocabulary, no person, random movement) | Display: "Unable to confidently recognize this sign." (never hallucinate) |
| Model produces fluent text for random movement | **HALLUCINATION** — system must NOT display the generated text as a translation |

### Documented findings

Investigate whether the model exposes any meaningful confidence/log-probability signal.
If it does not, document that. Then design the conservative SignSage-level acceptance
mechanism based only on measurable evidence available from the pipeline.

## 10. Temporal Window Investigation

Measure whether the model behaves differently with:

- Short sequences (5-20 frames)
- Natural-length sequences (full sign, typically 50-150 frames captured, padded to 500)
- Sequences padded heavily to 500 frames
- Sequences truncated to 500 frames

**Do not change the external preprocessing contract.** This is evaluation only.

### Findings to document:

- Does the model produce consistent output for the same sign captured at different frame counts?
- Does heavy padding (e.g., 5 frames padded to 500) degrade quality compared to natural-length sequences?
- Should capture have a defined start/stop window rather than continuously sending arbitrary frames?
- What is the optimal sequence length range for this model?

## 11. Performance

Measure using the actual evaluation pipeline:

| Metric | Target / Observation |
|---|---|
| Preprocessing time (adapter + normalization) | ~1-2ms for 5-frame sequence |
| Model inference time (pose encoder + T5 generate) | ~5ms (verified in Phase 2) |
| Decoding time | Included in inference above |
| Total end-to-end translation time | ~5-7ms per sequence (CPU) |
| Model loading time | ~8s (one-time) |
| Peak RAM | ~1.2GB (T5-small + pose encoder) |

**Do not call the application "real-time" unless the complete pipeline supports that claim.**
CPU-inference latency is suitable for offline or batch processing, not real-time interaction.

## 12. Distribution Shift

Explicitly document the difference between:

- **External model training data**: iSign dataset, studio conditions, controlled capture, likely specific camera distances, framings, and lighting
- **SignSage webcam capture**: uncontrolled home/office environments, varying camera distances, varying lighting, varying signer position, possible missing hands, varying body pose

### Investigated factors:

| Factor | External Training | SignSage Capture | Potential Impact |
|---|---|---|---|
| Camera distance | Controlled | Variable | May affect landmark scale |
| Framing | Studio-wide | Face/hand within webcam frame | May affect visibility |
| Lighting | Controlled | Variable (home/office) | May affect MediaPipe confidence |
| Signer position | Fixed | Variable | May affect landmark positions |
| Landmark stability | High (studio) | Medium (webcam movement) | May affect motion features |
| Missing hands | Rare | Common (one hand occluded) | Affects 75-landmark subset |
| Body pose variation | Fixed | Variable | Affects POSE landmarks |
| Sequence length | Variable, padded to 500 | Variable, padded to 500 | Model expects 500-frame max |
| Confidence values | Unknown (possibly varied) | Hardcoded 1.0 | Distribution mismatch |

### Mitigation (do NOT change preprocessing contract):

- Document the shift; do not claim it's negligible
- Evaluate quality on captured data; if degradation is observed, note it
- Do not attempt to "fix" the distribution shift by modifying the adapter's preprocessing
  (the external contract must remain mathematically equivalent)
- If distribution shift severely impacts quality, recommend returning to model development

## 13. Limitations

| Limitation | Severity | Notes |
|---|---|---|
| Confidence distribution | Medium | SignSage hardcodes 1.0; external training data confidence unknown |
| Distribution shift | High | Uncontrolled webcam capture vs studio training data |
| Model produces fluent text for garbage input | High | Hallucination/false positive risk — critical failure mode |
| Small evaluation vocabulary | Medium | Model does NOT support all of Indian Sign Language |
| CPU-only inference | Low | ~5ms per sequence; not suitable for real-time without optimization |
| No external validation dataset | Medium | Evaluation depends on captured SignSage data only |
| Temporal window undefined | Medium | Capture protocol may affect model behavior |
| Normalization timing verified but not evaluated on real data | Low | Z-score before motion confirmed in code; real-world effect unknown |

## 14. Recommendation

The final status must be one of:

| Status | When to Use |
|---|---|
| **GO** | Model produces accurate translations for supported signs on real capture; hallucination rate is acceptably low (<~10% for negative tests); confidence mechanism can be designed. **Proceed to controlled production integration.** |
| **CONDITIONAL GO** | Model produces some correct translations but hallucination rate is concerning; distribution shift is documented but manageable with capture constraints; quality can improve with more data or capture optimization. **Proceed with restrictions** (e.g., require good lighting, centered framing, both hands visible). |
| **NO-GO** | Model produces frequent hallucinations for unsupported input; accuracy on supported signs is too low; distribution shift is too severe; no reliable confidence mechanism. **Keep model blocked; return to model development.** |

### This evaluation's recommendation:

[TO BE FILLED AFTER EVALUATION RUN]

**Do NOT set SIGNSAGE_MODEL_PROMOTED=true** until an explicit quality evaluation confirms GO or CONDITIONAL GO status.

## 15. Files Changed

| File | Action |
|---|---|
| `docs/EXTERNAL_ISIGN_EVALUATION_SET.md` | Created — vocabulary classification and reference material |
| `docs/EXTERNAL_ISIGN_QUALITY_EVALUATION.md` | Created — full quality evaluation framework |
| `backend/scripts/evaluate_external_isign.py` | Created — evaluation harness |
| `backend/tests/ml/test_external_isign_t5_smoke.py` | Existing — smoke test (Phase 2) |
| `backend/app/ml/adapters/i_sign_t5_adapter.py` | Verified unchanged — matches external contract |

---

**Evaluation started**: [DATE]

**Model under test**: `t5-small-motion` from `manavdhamecha77/iSign-t5-pose-to-text`

**Safety gate**: `SIGNSAGE_MODEL_PROMOTED=false` (must remain throughout)

**Stop condition**: After evaluation report is complete. Do not promote, deploy, integrate into frontend, retrain, or fine-tune.