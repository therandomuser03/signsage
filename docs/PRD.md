# SignSage — Product Requirements

## What this is

A research prototype (M.Tech thesis project) for continuous, bidirectional translation
between Indian Sign Language (ISL) and English text/speech, built around a single core
research question: can a model trained on the iSign benchmark produce useful continuous
ISL <-> English translation, evaluated with standard MT metrics, without a human
interpreter present.

This is **not** a startup pitch and not a production accessibility product. Treat every
claim below as scoped to "good enough to evaluate and defend as a thesis," not "good
enough to deploy to Deaf users in a hospital or courtroom."

## Problem

~63 million people in India have significant auditory impairment (Sharma et al., IEEE
ICCCIS 2022). Existing academic work (see literature survey, 20 papers) is dominated by
isolated-sign recognition; continuous, sentence-level, bidirectional ISL translation
with non-manual marker (face/gaze/head) modeling is a genuine, still-open gap.

## Who else is in this space (know this before anyone asks)

- **DeepVisionTech.AI / Let'sTalkSign** — explicitly bidirectional sign<->speech,
  6 years operating, patented, deployed in classrooms/hospitals/workplaces.
- **SignISL**, **Signer.AI**, **SignAssistive** — text/speech -> ISL avatar products,
  Indian market, various deployment/traction claims (treat marketing claims like
  "<200ms" and "deployed at Indian Railways" with skepticism — these are early-stage
  companies and such copy is easy to overstate).

None of these publish WER/BLEU/ROUGE-L or evaluate on a public benchmark. **That is the
actual differentiator of this project**: rigorous, reproducible evaluation on iSign, not
being first to build a bidirectional ISL demo (we are not).

## Goals (SMART objectives, from the proposal)

1. Spatial-temporal Transformer-based model for continuous ISL-to-text translation.
2. Evaluate end-to-end translation latency on defined target hardware.
3. Implement and evaluate bidirectional ISL <-> natural-language communication.
4. Develop and evaluate a Text-to-Pose pipeline for continuous ISL output from natural
   language.

## In scope

- Responsive web app (desktop/laptop/mobile browsers)
- Continuous ISL -> language translation (sentence-level, not isolated-word)
- Spatial-temporal modeling of hands, face, and body (non-manual markers)
- Context-aware sentence generation
- Text -> ISL gloss -> continuous pose generation
- Bidirectional communication with dynamic sign visualization (skeleton/stick-figure
  first; avatar polish is not a goal — see Non-goals)

## Out of scope

- Photorealistic avatar generation
- Very large open-vocabulary ISL beyond benchmark scope
- Regional ISL dialect coverage
- Large-scale new volunteer data collection
- Production-grade mobile deployment
- **Any high-stakes or critical-context use** (legal proceedings, medical diagnosis,
  emergency services) — Deaf-community research (Krausneker & Schugerl 2022; Northeastern
  University research, Imai & Alikhani) is explicit that avatar/AI sign tech is not a
  substitute for certified human interpreters in these contexts, and this project makes
  no claim to be.
- Two-device / remote-pairing communication (room codes, WebSocket relay between
  separate clients) — **deferred until the single-device MVP proves useful**; see
  ARCHITECTURE.md for reasoning.

## Success criteria

This project succeeds if, in order:

1. `notebooks/signsage.ipynb` shows the Sign->Text model beating a trivial
   "predict the most common sentence" baseline on WER/BLEU/ROUGE-L on a held-out iSign
   split, and the Text->Pose model producing a DTW distance competitive with a
   retrieval baseline. Both comparisons come with stated caveats that must travel with the
   numbers: metrics cover only sentences of <=38 words, the retrieval baseline is
   capped at 3000 training rows (`RETRIEVAL_MAX_SAMPLES`), and the first Kaggle run is
   bounded to 2000/300/300 rows, which the notebook's own report labels a `GO CANDIDATE`
   rather than a validated result (see TESTING.md, PHASES.md).
2. A single-device demo shows the full loop working live: camera -> landmarks -> text,
   and text -> pose -> skeleton animation.
3. Latency and translation quality are measured and reported honestly (including
   failure modes), against the baselines listed in the original proposal
   (iSign benchmark, Khartheesvar et al. 2023, the earlier one-directional prototype,
   Santhanam et al. 2025).

It does **not** need a polished avatar, multi-device support, or production deployment
to be a successful thesis outcome.

## Non-negotiables

- Landmarks only, never raw video, ever leaves the user's device (privacy-by-construction,
  not a bolt-on — see SECURITY.md).
- No claim of solving non-manual-marker modeling in general — only a specific, falsifiable
  ablation (see PHASES.md / ARCHITECTURE.md).
