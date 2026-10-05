# External iSign Evaluation Set

## Objective

Define a small, controlled vocabulary of signs for evaluating the t5-small-motion model
through the SignSage iSign adapter. This set is based on the ISLRTC reference material
documented in the project and is intentionally limited — the model does NOT support all
of Indian Sign Language.

## Vocabulary Classification

Each sample is classified as one of:

### SUPPORTED_REFERENCE_SAMPLE

- The sign/phrase is in the ISLRTC reference vocabulary
- The sign can be reasonably performed within the camera frame
- The adapter preprocessing produces a valid (500, 600) tensor for motion model
- Model inference is expected to generate text related to the sign

### EXPERIMENTAL

- The sign/phrase is loosely related to the reference vocabulary but not explicitly listed
- The sign may require modifications to framing, distance, or lighting
- Model output is uncertain — should be treated as experimental until evaluated
- Do NOT claim this vocabulary is supported; record output and notes

### NOT_TESTED

- The sign is in the reference vocabulary but has not yet been captured/evaluated
- Insufficient data or poor capture quality
- Will be re-evaluated in a future run

### NOT_SUPPORTED

- The sign/phrase is NOT in the ISLRTC reference vocabulary
- The sign requires capabilities the model was not trained for (e.g., complex sentences,
  specific grammatical structures beyond single signs)
- Model generating fluent but unrelated text should be marked HALLUCINATION / FALSE POSITIVE
- Do NOT attempt to interpret model output as translation for NOT_SUPPORTED items

## Reference Material

This evaluation set draws from the ISLRTC (Indian Sign Language Research and Training Centre)
reference material already documented in the SignSage project. The vocabulary is limited to:

- Common, clearly identifiable signs (single words/concepts)
- Signs that can be performed within a standard webcam frame
- Signs using the 75-landmark subset (POSE + LEFT_HAND + RIGHT_HAND only)
- No complex grammatical constructions

**Do not invent vocabulary that is not supported by the reference material.**

## Sample Size Recommendation

- Minimum: 5 repetitions per sign
- Target: 10+ repetitions per sign if practical
- Signs: Start with 5-10 signs for an initial evaluation
- Expand gradually — a small, clean evaluation set is more useful than a large uncontrolled one

## Classification Summary Table

| Classification | Definition | Action |
|----------------|-----------|--------|
| SUPPORTED_REFERENCE_SAMPLE | ISLRTC vocabulary sign, captured on camera | Evaluate model output, compute accuracy |
| EXPERIMENTAL | Loosely related, not in reference vocabulary | Record output, do not claim support |
| NOT_TESTED | In reference vocabulary but not yet captured | Re-evaluate later |
| NOT_SUPPORTED | Not in reference vocabulary | Treat model output as potential hallucination |

## Vocabulary in Development (Pending Capture)

This section will be populated as signs are captured and evaluated. Do not add entries
until they have been captured via the evaluation protocol described in
`docs/EXTERNAL_ISIGN_QUALITY_EVALUATION.md`.