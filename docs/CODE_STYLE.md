# SignSage — Code Style

Conventions for backend, frontend and research code. **Observed** rules come from how the repository is already organised; **proposed** rules are the recommended defaults where no configuration was visible in the documentation — add the matching config file if it does not exist, then change "proposed" to "enforced".

## 1. General

- Smallest coherent change; preserve the existing architecture unless there is evidence to change it ([ARCHITECTURE §9](ARCHITECTURE.md#9-key-decisions)).
- Prefer clear names over comments; comment the *why*, especially for contracts and honesty rules.
- Never duplicate a contract value. Landmark counts, order, dimensions, tokenizer ids and error codes come from one constant, mirrored by a test ([ARCHITECTURE §5](ARCHITECTURE.md#5-contracts)).
- **Version anything that must match across training and serving** in its name (`pose_normalize_v1`, `sign_features_v3`). Never change behaviour under an existing version name.
- No secrets in code, tests, notebooks or logs. Redact keys in error messages.
- Never create numbered copies of files (`API(11).md`); edit in place.

## 2. Python (backend, research scripts)

Observed:
- Python `>= 3.11` (`pyproject.toml`); the recorded working environment used 3.13.
- FastAPI with Pydantic schemas in `app/schemas/`; the WebSocket schema is canonical.
- Layering: `api/` (thin) → `services/` (logic) → `ml/` adapters and `infrastructure/` (I/O). No model or database code in route handlers.
- Heavy imports (torch, mediapipe, opencv) are lazy where startup time matters.
- Settings come from `core/config.py` environment variables prefixed `SIGNSAGE_`; add a default and document it in [DEPLOYMENT](DEPLOYMENT.md).
- Errors surface as typed exceptions mapped to documented codes; honest refusal is a normal return path, not a crash.

Proposed:
- Formatter and linter: `ruff format` + `ruff check`; type hints on public functions; `mypy` or `pyright` on `app/`.
- Use `pathlib`, f-strings, `dataclasses` or Pydantic models for structured data.
- Research code: seed everything, log the source revision, write outputs atomically (temp file then rename) so an interruption never leaves a half-written artifact.

## 3. TypeScript / React (frontend)

Observed:
- TypeScript, React 19, Vite, Tailwind 4, shadcn/ui; folders `app/`, `features/<name>/`, `services/`, `components/`.
- Wire types live beside the transport (`services/websocket/protocol.ts`, `types/api.ts`) and mirror the backend schemas.
- `tsc -b` must pass. `npm run lint` carried about 39 pre-existing errors in a past audit — do not add new ones; fix them in a dedicated pass.
- Detection work is throttled; never run landmark extraction at full animation-frame rate.
- Cleanup effects: cancel timers, intervals, sockets and camera tracks on unmount or mode change.

Proposed:
- Prettier for formatting; ESLint with the React and TypeScript presets.
- Function components and hooks; keep a page component thin and move logic into hooks under the feature folder.
- User-visible strings that make a claim (status, accuracy, support) go through one place so [DESIGN_SYSTEM §7](DESIGN_SYSTEM.md#7-voice-and-copy) can be audited.

## 4. Notebooks

- A notebook is a runbook, not a library: settings in the first cell, long-running work **resumable**, outputs written atomically, secrets from Colab secrets or the environment only.
- Anything reusable moves into a module that has tests; keep the notebook thin.
- Clear outputs before committing unless they are the recorded evidence, and never leave keys or personal data in them.

## 5. Git and reviews

- Commit messages follow the pattern already in use: `type(scope): summary`, for example `feat(history): …`, `test(api): …`, `docs: …`, `fix: …`.
- One logical change per commit. Update tests and the affected documents in the same change ([AGENTS.md](../AGENTS.md)).
- A pull request states what was run (backend tests, `tsc -b`, build, E2E) and what was **not** — never claim a check passed that was not run.
