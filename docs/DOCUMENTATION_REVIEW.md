# SignSage — Documentation Review

**Review date:** 2026-09-29

## 1. Main issue found

The documentation set contains repeated historical snapshots.

Examples include:

- early documents saying no backend/frontend existed,
- later documents describing the implemented backend,
- later documents describing the integrated frontend,
- repeated numbered copies such as `API(9).md`, `SECURITY(10).md`, etc.

This makes it difficult to know which statement is current.

## 2. Recommended structure

Keep specialized documents separate.

Do **not** merge the entire documentation set into one file.

Recommended top-level set:

```text
docs/
├── MASTER.md
├── STATUS.md
├── PRD.md
├── ARCHITECTURE.md
├── API.md
├── DATA_MODEL.md
├── DESIGN.md
├── SECURITY.md
├── TESTING.md
├── PHASES.md
└── DECISIONS.md
```

## 3. Why separate files

Different questions need different authorities:

- PRD = what is intended
- Architecture = how it works
- API = how components communicate
- Security = what data/security rules apply
- Testing = how correctness is checked
- Phases = how development progressed
- Decisions = why choices were made
- Status = what is true now
- Master = navigation and high-level truth

## 4. Numbered duplicate files

Files such as:

```text
API(9).md
API(10).md
SECURITY(9).md
DESIGN(10).md
PHASES(20260929-160136).md
```

should not coexist as active source-of-truth documents.

They should be archived, renamed as historical snapshots, or removed from the working documentation tree.

## 5. Model wording

Avoid:

> GO-CANDIDATE = good model

Use:

> historical GO-CANDIDATE label from an earlier evaluation gate; later evaluation exposed severe collapse and the model is not promoted.

## 6. Current vs historical

Every frequently changing document should have a status line.

Recommended vocabulary:

- Current
- Historical
- Experimental
- Planned
- Deferred

## 7. Documentation rule

When implementation changes:

1. update code,
2. update tests,
3. update `STATUS.md`,
4. update the relevant specialist document,
5. add a decision entry if the architecture changed.

This prevents documentation drift.
