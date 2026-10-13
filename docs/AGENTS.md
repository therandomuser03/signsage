# AGENTS.md — SignSage

Instructions for AI coding agents working on the SignSage repository.

## 1. Read these first

Before changing code, read:

1. `MASTER.md` (orientation + map)
2. `STATUS_AND_ROADMAP.md` (current state)
3. `DEVELOPMENT.md` (interfaces, deploy, security, testing)
4. `DATA_AND_MODEL.md` (data, vocabulary, readiness)
5. `EXTERNAL_EVALUATION.md` (only for external-model work)
6. `reference/` pins (only for contract-value changes)

For model/research work, also read:

- `research/evaluation/reports/352868924/evaluation_results.json` (verified NO-GO evidence; the historical `research/outputs/phase0/RESULTS.md` path cited by older docs is TBD/missing in this checkout)
- `research/notebooks/signsage.ipynb`

## 2. The most important rule

**Do not confuse an implemented software pipeline with a validated translation model.**

The current application infrastructure is substantially implemented. The research Sign→Text model remains experimental and has documented output-collapse problems.

Never tell the user that the model is accurate, production-ready, or generally useful unless new evidence in the repository establishes that.

## 3. Current truth hierarchy

When documents disagree:

1. Actual source code and tests determine implementation.
2. `STATUS_AND_ROADMAP.md` determines current project state.
3. `DEVELOPMENT.md` + `reference/end-to-end-contract.md` determine communication contracts.
4. Historical evaluation records (e.g. `research/evaluation/reports/352868924/evaluation_results.json`; older docs cite a historical `RESULTS.md` path that is TBD/missing) determine measured research outcomes.
5. `MASTER.md` §7 explains architectural choices.
6. `MASTER.md` §5 describes intended scope.

Historical numbered snapshots such as `API(9).md` are not automatically authoritative.

## 4. Before editing

Determine whether the task is:

- research/training,
- backend,
- frontend,
- API/contract,
- database,
- security,
- deployment,
- documentation.

Read the relevant specialist document before changing the corresponding area.

## 5. Preserve existing architecture unless there is evidence to change it

Current architectural choices include:

- FastAPI backend,
- React/TypeScript/Vite frontend,
- browser-side MediaPipe landmark extraction,
- WebSocket translation transport,
- Supabase authentication/history,
- single-device communication,
- research/training separated from serving.

Do not replace these merely because another stack is fashionable.

If proposing a major change, record the reason in `MASTER.md` §7.

## 6. Research rules

For Sign→Text:

- preserve the documented input representation unless the experiment explicitly changes it,
- keep train/validation/test leakage controls,
- compare against a baseline,
- report WER/BLEU/ROUGE-L and output diversity,
- inspect qualitative outputs,
- investigate repetition/mode collapse.

For Text→Pose:

- report DTW/MSE together with motion energy,
- do not interpret low MSE as successful generation by itself.

## 7. Dataset rules

Do not download, commit, or expose the gated iSign dataset unnecessarily.

Do not invent column names or pose-file layouts. Check the actual dataset when implementing data code.

Use the documented `video_id` split rule unless a new experiment explicitly justifies another split.

## 8. Privacy/security rules

Never:

- commit API keys,
- expose server secrets in frontend code,
- add a raw-camera upload path without explicit architectural approval,
- store user history without enforcing account isolation.

The intended camera pipeline is:

```text
camera → browser MediaPipe → landmarks → backend
```

## 9. Testing rules

After a non-trivial change, run the smallest relevant checks.

Examples:

```text
backend change → backend tests
frontend change → typecheck/build/tests
API change → contract tests
research change → notebook/data/evaluation checks
security change → targeted authorization/privacy tests
```

Do not claim a test passed unless it was actually run.

## 10. Documentation rules

When behavior changes:

1. update code,
2. update tests,
3. update `STATUS_AND_ROADMAP.md`,
4. update the affected specialist document,
5. update `MASTER.md` §7 if the architectural decision changed.

Avoid creating numbered copies such as `API(11).md`.

## 11. Communication style

Be precise.

Prefer:

> “The WebSocket path is implemented and tested, but the model output remains experimental.”

Avoid:

> “The translator works.”

unless the statement is explicitly scoped to software execution rather than translation quality.

## 12. Agent workflow

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

Never infer implementation from the PRD (it describes desired behavior; code +
tests determine what exists). Never infer model quality from infrastructure:
checkpoint loads, inference executes, and syntactically valid output are separate
claims from semantically correct, generalizing, production-safe output — only the
relevant evaluation evidence supports the latter. When uncertain, state the
uncertainty instead of inventing project facts. (Consolidated from
`AI_AGENT_GUIDE.md`, 2026-10-12; that file is removed.)
