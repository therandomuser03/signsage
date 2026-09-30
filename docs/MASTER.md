# SignSage — Master Project Guide

**Status:** Current orientation document  
**Last reviewed:** 2026-09-29

## 1. Purpose

This file is the shortest route to understanding SignSage.

It does not replace specialized documentation. It tells a developer, evaluator, or future maintainer which document answers which question and separates:

- research evidence,
- implemented software,
- experimental components,
- planned work,
- deferred work.

## 2. One-sentence definition

SignSage is an M.Tech research prototype for bidirectional ISL ↔ English communication using browser-side landmark extraction, a FastAPI/WebSocket backend, Sign→Text inference, and English→ISL gloss/pose generation.

## 3. Reality check

### Implemented / integrated

- FastAPI backend
- WebSocket translation contract
- Diagnostics REST endpoints
- Browser MediaPipe Holistic integration
- Landmark transport
- Authentication integration
- History API/data isolation
- English→gloss endpoint
- Text→pose/skeleton rendering infrastructure
- Model capability reporting
- Automated backend/frontend/contract verification described by the project docs

### Experimental

- Research Sign→Text checkpoint
- Research Text→Pose generator
- General continuous ISL translation quality
- Generated-pose semantic fidelity
- Broad signer generalization

### Deferred

- Two-device relay
- Production-grade mobile deployment
- Photorealistic avatar
- Open-vocabulary ISL coverage
- Regional dialect coverage
- High-stakes deployment
- Large-scale new volunteer collection

## 4. Most important research finding

The completed Phase 0 experiments do **not** establish a useful Sign→Text model.

The documented larger run reports:

- trivial baseline WER ≈ 0.999
- GRU WER ≈ 0.991
- Transformer WER ≈ 1.367
- Transformer output diversity ≈ 21% in the earlier evaluation snapshot
- Text→Pose motion energy ≈ 6.7% of ground truth

Later evaluation notes report that filtering the junk-text issue changed the baseline and reversed the earlier GO-CANDIDATE interpretation. Therefore the old `GO-CANDIDATE` wording must not be treated as model readiness.

## 5. Architecture at a glance

```text
Browser
 ├─ Camera
 ├─ MediaPipe Holistic
 ├─ Motion-pause / manual boundary
 └─ React UI
       │
       │ landmarks / text
       ▼
FastAPI
 ├─ WebSocket translation
 ├─ Auth
 ├─ History
 ├─ Capabilities / model status
 ├─ Sign→Text
 └─ Text→Gloss / Text→Pose
       │
       ├─ Supabase
       └─ Optional Groq / Ollama
```

## 6. Documentation authority

When documents conflict:

1. Actual code and tests describe implementation.
2. `API.md` describes the intended public communication contract.
3. `STATUS.md` describes current verification state.
4. `RESULTS.md` describes measured research outcomes.
5. `DECISIONS.md` explains why a technical choice was made.
6. `PRD.md` describes intended scope, not implementation.
7. Historical phase notes should not override current implementation evidence.

## 7. Golden rule

**Do not document what the project is supposed to do as though it already does it.**

For model-related claims, always distinguish:

```text
can execute
    ≠
produces output
    ≠
produces useful output
    ≠
generalizes
    ≠
is production-ready
```
