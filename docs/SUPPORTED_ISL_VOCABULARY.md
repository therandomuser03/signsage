# SignSage Supported ISL Vocabulary (evidence-bound)

Rule: a sign/phrase is SUPPORTED only with all three — present in training
data, present in evaluation, and successfully recognized in end-to-end
testing. Anything less is classified below. No mappings are invented here.

## Verdict up front

**SUPPORTED: none. The supported recognition vocabulary is the empty set.**
SignSage recognizes zero ISL signs today. The backend correctly answers
`MODEL_UNAVAILABLE`, and the UI must say so (it does). Nothing below changes
that; it explains exactly why each candidate set does NOT qualify.

## Evidence sources consulted

| Source | What it is | Recognition evidence? |
|--------|-----------|----------------------|
| iSign v1.1 (HF gated + old laptop; NOT in repo) | Continuous ISL videos with English translation targets; V1 tokenizer trained on 99,699 target sentences (`research/experiments/.../train_text.txt`, narrative-domain English, e.g. Akbar tales) | Domain only — no per-sign label inventory available locally |
| 352868924 eval (`research/evaluation/reports/352868924/`) | WER 1.0046, EM **0.0**, diversity 0.99%, top prediction 44.8% (`"the police said that"` on random input), `div_ok: False` → verified NO-GO | Negative: zero successful recognitions |
| T5 POC (`research/t5_pose_poc/`) | No weights exist; synthetic gate failed closed as designed | None |
| `Let's Learn some Basic Sign.pdf` (ISLRTC, 15pp) | Image/scanned reference document; only title text + the string "SEEYOUTOMORROW" are machine-extractable — its sign contents cannot be enumerated from the file and were NOT enumerated | None — reference material, not recognition evidence, not a claim source |
| `sign_vocabulary.json` + avatar manifest (14 IDs) | EN→ISL **generation** vocabulary (avatar animations), contract-tested | None for recognition — wrong direction (see below) |

## SUPPORTED

(none)

## EXPERIMENTAL (trained on, never recognized)

- The iSign-derived continuous-translation domain: dynamic (video-sequence)
  signs with sentence-level English targets. Static vs dynamic per-sign
  breakdown is unknown locally (no label inventory on this machine).
- 352868924 was trained on this domain and evaluated: exact match 0.0 across
  the test set — i.e., **no phrase in the training domain was successfully
  recognized**, which is why it is NO-GO and unpromoted.

## NOT TESTED

- Every iSign label: present in neither a local inventory nor any passing
  evaluation.
- The full ISL language: no model, no dataset split, and no evaluation
  anywhere in this project demonstrates it.

## NOT SUPPORTED (explicit exclusions — do not promote these)

1. **All of ISL as a language.** No evidence supports open-vocabulary claims.
2. **The 14 avatar gloss IDs** (`HELLO, THANK_YOU, SORRY, PLEASE, YES, NO,
   HELP, WATER, FOOD, NAME, YOU, ME, WHERE, HOW`): these are animation IDs
   for English→ISL avatar output. Listing them as "supported signs" would
   falsely imply the camera recognizes them. It does not.
3. **ISLRTC PDF contents** (including any alphabet/handshape pages and any
   sign readable from its images, e.g. the extracted string
   "SEEYOUTOMORROW"): a learning reference, not a capability. Its signs were
   never training labels, never evaluated, never recognized here. Do not add
   labels because they appear in the PDF.
4. **The ISL alphabet / fingerspelling**: no alphabet recognizer exists in
   this repository.

## Static vs dynamic (where known)

- Recognition inputs are always dynamic sequences (`(T,543,3)`, T≥2
  enforced); there is no isolated-static-sign classifier anywhere in the
  serving path, so even "static" signs have no supported route.
- iSign training targets are sentence-level translations of continuous
  signing (dynamic by nature).

## UI / README honesty requirement

- `README.md` already states research checkpoints are unreliable for
  open-vocabulary translation in either direction — keep it.
- `TranslatePage.tsx` makes no vocabulary-scope claim (verified: no
  "supported signs" language) — if scope text is ever added, it must read
  to the effect of: "No signs supported yet — model unavailable." Any list
  of signs, any "try signing X" prompt, or any alphabet chart in the
  recognition flow would be a fabrication until the table above gains a
  SUPPORTED row with evaluation evidence.
