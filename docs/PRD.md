# SignSage — Product Requirements Document (PRD)

| | |
|---|---|
| **Status** | Living document. Consolidated 2026-11-18 from `MASTER`, `product/*`, `STATUS_AND_ROADMAP` and the project owner's current direction. |
| **Model gate** | `SIGNSAGE_MODEL_PROMOTED=false` |
| **Authority** | This file states **intended** scope and requirements. It never proves that something is implemented — code and tests do ([AGENTS.md](../AGENTS.md)). |

## 1. Summary

SignSage is a web application for two-way communication between Indian Sign Language (ISL) users and English speakers:

- **ISL → English:** the browser reads the camera, extracts body/hand/face landmarks locally, and the backend recognises signs and composes English.
- **English → ISL:** typed or spoken English is mapped to ISL glosses and signed by a 3D avatar.

It began as an M.Tech research prototype extending an earlier ASL project (V1). The project direction as of November 2026 is to make it a **real, usable product with real users**, with recognition trained properly on a large ISL dataset — without ever dropping the honesty rules in §6.

## 2. Problem and users

**Problem.** Deaf and hard-of-hearing people and non-signers cannot communicate without a skilled interpreter, and interpreters are scarce. India has on the order of 98 lakh deaf or hearing-impaired people (Census of India, as cited in ISLRTC material), while ISL is far less served by datasets and tools than ASL.

| User | Needs |
|---|---|
| **ISL signer** | Be understood in English without an interpreter; see clearly when the system is unsure |
| **Hearing non-signer** (family, shop or service staff) | Read the signer's words; type or speak back and see them signed |
| **ISL learner** | Look up a sign; see a letter or word recognised |
| **Evaluator / researcher** | Reproducible procedures, measured results, honest failure analysis |

## 3. Goals and non-goals

### Goals (current direction)

| # | Goal |
|---|---|
| G1 | Make SignSage usable by real people, not only demonstrable. |
| G2 | Complete the recognition backend properly: train on the **whole** FDMSE-ISL dataset (2,002 words), evaluate on its official signer-independent split, and report exactly how much of the data was used. |
| G3 | Add a **fingerspelling fallback**: when a word cannot be recognised confidently, offer letter-by-letter (A–Z) recognition instead of guessing. |
| G4 | Improve Text → Sign: accurate hand movement and a vocabulary far beyond the current 14 signs, with honest labels until a fluent signer has reviewed the motion. |
| G5 | Give visible credit: the V1 contributors, the dataset authors, and every third-party component, in the repo and in the product. |
| G6 | Support the thesis: real-user evidence, reproducible procedures, measured results, failure analysis. |

### Non-goals

- A certified interpreter or any medical, legal, emergency or other high-stakes use.
- Uploading or storing raw camera video from the live camera path (the optional manual *upload a video* flow processes only the file the user picks).
- Claiming open-vocabulary sentence-level translation without evidence.
- Two-device relay, a native mobile app and a photorealistic avatar (deferred).

## 4. Feature scope and status

Status words are defined in §11. "Gate" is what must be true before the claim can be made. Statuses reflect the repository documentation as of 2026-11-17 — **verify against code and tests before repeating them.**

| Feature | Status | Gate / evidence |
|---|---|---|
| Camera → MediaPipe → landmarks → WebSocket | Implemented | Contract tests; real-camera session still to be verified on target hardware |
| Static alphabet classifier (A–Z) | Verified | 87.1% held-out accuracy, macro-F1 0.849; weak classes W, Z, S |
| Isolated word recognition `word_v3` | Experimental | 9 words, one signer; below promotion bar |
| Full-vocabulary recognition (FDMSE-ISL) | Experimental | Offline baseline trained: 82.1% top-1 / 95.1% top-5 on 8 unseen signers (studio video). Not integrated into the live app; webcam accuracy unmeasured; promotion gates still open |
| Fingerspelling fallback | Planned | FDMSE-ISL has all 26 letters but they average only 64% recall as words-among-2,002; restrict output to the 26 letters in spelling mode and measure, or train a letter-only model (see §9) |
| Sentence composition | Implemented | Rules, then constrained LLM; every signed word appears exactly once and no content word can be invented |
| English → gloss | Implemented | Deterministic lookup (14 IDs) plus optional LLM mapper, vocabulary-filtered |
| Gloss → 3D avatar motion | Experimental | 3 recorded-data clips, 4 hand-authored approximations, others show "No clip yet"; not validated by a signer |
| Text-to-speech | Implemented | Server provider when configured, otherwise browser voices |
| Speech-to-text input | Implemented | Browser Web Speech or server STT |
| Accounts, history | Implemented | Mocked integration passes; live multi-account verification still conditional |
| Open-vocabulary sentence-level Sign → Text | **Not available** | Previous transformer collapsed (WER ≈ 1.0); see [MODELS_AND_DATA](MODELS_AND_DATA.md) |

## 5. Requirements

### Functional

| ID | Requirement |
|---|---|
| FR1 | Live Interpreter shows recognised words with confidence and the top alternatives, and a distinct state for waiting, recognising, composing, composed and not confidently recognised. |
| FR2 | Below the confidence threshold the system refuses or offers fingerspelling; it never shows a fluent guess. |
| FR3 | Composed sentences use only the signed words (plus a small grammar-word allowlist). Unverifiable output falls back to a plain word join. |
| FR4 | Text → Sign lists unknown words as "dropped, not guessed" and shows "No clip yet" for glosses without motion. |
| FR5 | Auto-speak reads completed interpretations only, once each; never intermediate predictions. |
| FR6 | Sign-in is optional. Signed-in users get history they can search and delete; sign-out clears protected UI state. |
| FR7 | `/health` and `/capabilities` report model state truthfully; refusal is an explicit, documented state (`MODEL_UNAVAILABLE`, 503 `MODEL_NOT_PROMOTED`). |
| FR8 | A credits/about surface lists the V1 contributors, dataset authors with citations, and third-party licenses. |
| FR9 | Camera denial, poor tracking, hand out of frame, socket failure and an unavailable model each produce a clear message. |

### Non-functional

| ID | Requirement |
|---|---|
| NFR1 | **Privacy:** only landmarks leave the browser on the live camera path; the optional manual video upload is disclosed and its files are not retained; per-user isolation of history. See [SECURITY.md](../SECURITY.md). |
| NFR2 | **Honesty:** UI and docs distinguish verified, experimental, unsupported and unavailable (§6). |
| NFR3 | **Reliability:** bounded WebSocket reconnect (5 attempts with backoff); no queued stale frames. |
| NFR4 | **Performance:** landmark detection throttled to a bounded rate; end-to-end latency to be measured on the target device before any real-time claim. |
| NFR5 | **Reproducibility:** every reported metric traces to a dataset version, split, seed and source revision. |
| NFR6 | **Licensing:** dataset and weight terms respected and credited ([CREDITS.md](../CREDITS.md)). |
| NFR7 | **Accessibility:** large readable captions, theme support, keyboard operation — concrete targets still to be defined. |

## 6. Honesty and claims policy

Do not document or display what the project *should* do as though it already does.

`can execute ≠ produces output ≠ produces useful output ≠ generalises ≠ production-ready`

- A checkpoint that loads and runs proves nothing about quality.
- Never turn a metric from one task into a claim about another (isolated-sign accuracy, continuous gloss WER and sentence BLEU are different things).
- Say "The WebSocket path is implemented and tested, but model output is experimental" — not "The translator works".
- A wrong confident output is a **failure**; explicit uncertainty is a **correct** outcome.

## 7. Success criteria

**Promoting a recognition model** requires all of: documented held-out evaluation beating a trivial baseline, output diversity (no collapse), per-class results, signer-independent evaluation (or an explicitly accepted, written limitation), calibrated confidence floor, a passing wrong-input-must-reject test, bundle export and load check, a license review, and an explicit human release decision. The isolated-word bar used so far: **80% accuracy, 80% macro recall, 60% per-class recall.** Details: [MODELS_AND_DATA](MODELS_AND_DATA.md).

**Thesis success** = reproducible procedures + an integrated application + measured results + failure analysis + a defensible validation story. A working demo alone is not evidence of quality.

**Real-user metrics** (task success, usability score, error and refusal rates) are to be defined with the project guide before the first user study.

## 8. Demonstration scope

Demonstrate only what current evidence supports.

1. Overview of the architecture and the honesty gate.
2. Static alphabet recognition with a known test image (confidence and accept/reject visible).
3. English → ISL with supported words; show dropped-word handling and the "not validated ISL" label.
4. `/capabilities` and the explicit model-unavailable state, if useful.

Before presenting: start both servers, check `/api/v1/health` and `/api/v1/capabilities`, test the exact browser, camera and network, and keep clearly labelled recordings as fallback.

Do **not** present as completed: continuous sentence translation, broad-vocabulary or signer-independent recognition, experimental word recognition as production-ready, the 20-class static word classifier as reliable, or synthetic smoke-test output as a real translation.

## 9. Roadmap

Order matters; each step has a gate.

| Phase | Work | Gate |
|---|---|---|
| **Now** | ~~Run the resumable FDMSE-ISL landmark extraction over all 2,002 words~~ (done Nov 2026: 40,033 videos, 0 failed); check face-detection quality; train and evaluate | Official split respected; counts of used videos reported |
| | ~~Evaluate on the signer-independent test split, then retrain on all signers for the deployment model~~ (done: 82.1% top-1) | Test metric reported from the first run only |
| | Bundle exported and threshold calibrated on validation (done); remaining: backend adapter and tests, live-webcam calibration | §7 |
| | Text → Sign: fix hand-motion quality; widen vocabulary using recorded data | Signer review before any "validated" label |
| **Next** | Fingerspelling fallback: FDMSE-ISL has all 26 letters (mean recall 64.4% among all 2,002 classes, confused with words such as "Add" and "November"); evaluate a letters-only output, compare with the A–Z image classifier (87.1%) | Held-out letter accuracy reported |
| | Check that landmarks produced in the browser match those used in training (extractor parity, including the pose-derived face points and the video aspect ratio of `sign_features_v4`) | Numeric comparison on the same clips |
| | Credits page and frontend polish; live multi-account auth and RLS test | Live test log |
| | Latency on the target device | Measured, with hardware stated |
| **Later** | Fluent-signer review of avatar motion; larger vocabularies; native mobile; two-device relay | Fresh review for each |

**Open decisions**
- Dataset terms: permission covers the college project; confirm **in writing** that a free, non-commercial public thesis app and the trained weights are covered before opening to the public.
- Where landmark extraction for inference should run, and whether it matches the training extractor.
- Which text-to-speech provider is the default (documents disagree; check the TTS service).

## 10. Risks

| Risk | Mitigation |
|---|---|
| Studio-recorded training data differs from webcam capture (distribution skew) | Parity check; collect webcam evaluation clips; calibrate threshold |
| Isolated words are not sentences; sentence segmentation is hard | Pause detection plus word composition; keep claims at word level |
| Model collapse or fluent-wrong output | Diversity gate; wrong-input-must-reject test; confidence floor |
| Dataset or weight license limits | Written permission; credits; license gate before public release |
| Free-tier hosting limits (PyTorch, WebSocket) | Capacity test; paid tier if needed |
| Avatar motion mistaken for correct ISL | Persistent "not validated" labels until signer review |

## 11. Status vocabulary

| Term | Meaning |
|---|---|
| **Implemented** | Code exists and a check passes |
| **Integrated** | Components talk through the documented contract |
| **Verified** | A documented check or held-out evaluation passed |
| **Research result** | Measured, scoped to its dataset and task |
| **Experimental** | Available but unvalidated |
| **In progress / Planned** | Work under way / agreed, not started |
| **Production-ready** | Never claimed without full evidence |
