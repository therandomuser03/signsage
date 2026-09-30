# SignSage — Product & Research Requirements

**Status:** Research scope + application requirements

## 1. Definition

SignSage is an M.Tech research prototype exploring continuous, bidirectional translation between Indian Sign Language (ISL) and English text/speech.

It is not defined as a production accessibility service.

## 2. Research objectives

1. Investigate spatial-temporal modeling for continuous ISL→English translation.
2. Measure translation latency on defined hardware.
3. Implement a bidirectional ISL↔English communication workflow.
4. Investigate English→ISL gloss and continuous pose generation.

## 3. Application scope

### In scope

- Responsive web interface
- Shared-device communication
- Continuous ISL input
- English text output
- English text input
- ISL gloss generation
- Pose/skeleton visualization
- Authentication
- Translation history
- Capability/model-status reporting
- Privacy-aware landmark transport
- Explicit uncertainty/error surfaces

### Out of scope

- Photorealistic avatar generation
- Unrestricted open-vocabulary ISL
- Regional dialect coverage
- Production-grade mobile deployment
- Large-scale volunteer data collection
- Two-device/remote relay
- High-stakes legal, medical, or emergency use
- Claims of replacing certified human interpreters

## 4. Requirements

### R1 — Communication

The system should support both directions:

```text
ISL → English
English → ISL representation
```

### R2 — Privacy

The translation protocol should transmit numerical landmarks rather than raw camera video.

### R3 — Transparency

The UI must expose model availability and meaningful failure states rather than silently presenting uncertain output as correct.

### R4 — Authentication

Authenticated users may access account-scoped functionality such as history.

### R5 — History

Stored translation history must be isolated by authenticated user.

### R6 — Research integrity

Research metrics must be documented separately from product claims.

## 5. Success criteria

The project is successful as a thesis when it can demonstrate:

- reproducible data/training/evaluation procedures,
- a technically integrated application,
- measurable research results,
- explicit failure analysis,
- and a defensible explanation of what is and is not validated.

A demo working mechanically is not sufficient evidence of translation quality.

## 6. Current implementation caveat

The research Sign→Text models documented in Phase 0 currently show severe output collapse. Therefore the application should expose model status and avoid implying that general continuous ISL translation has been solved.
