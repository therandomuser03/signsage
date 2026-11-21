# AGENTS.md — SignSage

Instructions for AI coding agents (and a useful checklist for humans). Keep it short; the details live in `docs/`.

## 1. Read first

| Task | Read |
|---|---|
| Any task | [docs/PRD.md](docs/PRD.md) §3–6 (scope, honesty policy) and this file |
| Backend, API, contracts | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Models, datasets, training | [docs/MODELS_AND_DATA.md](docs/MODELS_AND_DATA.md) |
| UI | [docs/DESIGN_SYSTEM.md](docs/DESIGN_SYSTEM.md) |
| Avatar motion | [docs/SIGN_ANIMATION.md](docs/SIGN_ANIMATION.md) |
| Deploy / env vars | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| Secrets, privacy | [SECURITY.md](SECURITY.md) |
| Style, tests | [docs/CODE_STYLE.md](docs/CODE_STYLE.md), [docs/TESTING.md](docs/TESTING.md) |

## 2. The most important rule

**Do not confuse an implemented pipeline with a validated model.**
`can execute ≠ produces output ≠ produces useful output ≠ generalises ≠ production-ready`.
Never say the model is accurate, production-ready or generally useful unless evidence in the repository establishes it. When unsure, say so — do not invent project facts.

Prefer: "The WebSocket path is implemented and tested, but model output is experimental."
Avoid: "The translator works."

## 3. Truth hierarchy (when sources disagree)

1. Source code and tests — what is implemented.
2. The pinned contracts in `ARCHITECTURE.md` §5 (and the schema files they cite).
3. `MODELS_AND_DATA.md` — measured results and model status.
4. `PRD.md` — intended scope; **never infer implementation from it**.
5. `CHANGELOG.md` and anything dated — history, not current instructions.

## 4. Non-negotiables

- Raw camera video never leaves the browser; do not add an upload path without explicit architectural approval.
- Never commit or print secrets: API keys, Supabase service-role keys, `HF_TOKEN`, the FDMSE-ISL access key. Redact them in logs and notebooks.
- Never commit datasets, videos, archives or weights.
- Do not weaken a gate: `SIGNSAGE_MODEL_PROMOTED` stays `false` without a documented promotion; never enable `SIGNSAGE_ALLOW_UNPROMOTED_DEV` on a public service.
- Never fabricate a translation, a metric, a test result or a vocabulary entry. Unknown words are dropped, never guessed.
- Respect dataset licences and credit sources ([CREDITS.md](CREDITS.md)). Do not redistribute FDMSE-ISL data or data-derived clips without written permission.
- Store user history only with per-user isolation enforced.

## 5. Preserve the architecture unless evidence says otherwise

FastAPI backend · React/TypeScript/Vite frontend · browser-side MediaPipe · WebSocket for the live loop · Supabase auth/history · single-device scope · research separate from serving. A major change needs a reason recorded in `ARCHITECTURE.md` §9.

## 6. Research rules

- Split by video, and by signer where metadata allows; keep leakage controls; compare with a trivial baseline; report diversity; inspect failures by hand.
- Report test metrics from a single evaluation; never tune on test.
- Text → Pose: report DTW/MSE **with** motion energy.
- Do not invent column names or file layouts — inspect the real data.
- Keep long jobs resumable and outputs atomic.

## 7. Workflow

```text
Understand the task → read the relevant doc → inspect the actual code
→ make the smallest coherent change → run the relevant checks → update docs
```

Run the smallest relevant checks (see [TESTING](docs/TESTING.md)); **never claim a check passed unless you ran it.**

## 8. Keeping docs true

When behaviour changes, update in the same change: code, tests, the affected doc, and `CHANGELOG.md`. Update `PRD.md` §4 if a feature's status changed and `ARCHITECTURE.md` §9 if a decision changed. Edit documents in place — no numbered copies (`API(11).md`), no new one-off report files; put history in the changelog.

## 9. Style of communication

Be precise and scoped. State what was verified, how, and what was not. Prefer short, concrete statements over broad claims.
