# SignSage — AI Agent Guide

This document is a compact operational guide for coding/research agents.

## Agent workflow

```text
Understand task
    ↓
Read MASTER + STATUS
    ↓
Identify affected subsystem
    ↓
Read specialist documentation
    ↓
Inspect actual code
    ↓
Make smallest coherent change
    ↓
Run relevant checks
    ↓
Update documentation/status
```

## Never infer implementation from the PRD

The PRD describes desired behavior.

The source code + tests determine what exists.

## Never infer model quality from infrastructure

These are separate claims:

- model checkpoint loads,
- inference executes,
- output is syntactically valid,
- output is semantically correct,
- output generalizes,
- output is safe for real-world use.

Only the relevant evaluation evidence can support the latter claims.

## When changing the ML pipeline

Check:

- input dimensionality,
- component ordering,
- normalization,
- split strategy,
- decoder behavior,
- padding/masking,
- baseline implementation,
- output diversity,
- WER/BLEU/ROUGE-L,
- Text→Pose motion energy.

## When changing the frontend

Check:

- camera permission behavior,
- landmark shape,
- WebSocket message schema,
- model capability state,
- error states,
- auth state,
- responsive layout.

## When changing backend APIs

Update:

- route implementation,
- schemas,
- tests,
- `API.md`,
- frontend consumers,
- `STATUS.md` if externally visible behavior changes.

## When uncertain

Do not silently invent missing project facts.

State the uncertainty, inspect the repository, or ask the user for the missing source of truth.
