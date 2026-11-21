# Contributing to SignSage

Thank you for helping. SignSage supports people who rely on accurate communication, so one rule matters above all: **never present an unvalidated result as a translation.** Read [AGENTS.md](AGENTS.md) §2 and §4 first.

## Set up

```bash
# backend
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs

# frontend (another terminal)
cd frontend && npm install && cp .env.example .env.local
npm run dev                              # http://127.0.0.1:5173
```

Never replace an existing `.env` with a template, and never commit one. Supabase is optional; the app works signed out.

## Before you open a pull request

1. Keep the change small and focused.
2. Run the relevant checks ([docs/TESTING.md](docs/TESTING.md)): `pytest`, `npm test`, `npx tsc -b && npm run build`, and E2E if you touched the UI flow.
3. Update documentation in the same change ([AGENTS.md](AGENTS.md) §8), including [CHANGELOG.md](CHANGELOG.md).
4. State in the PR what you ran and what you did **not** run.
5. Follow [docs/CODE_STYLE.md](docs/CODE_STYLE.md); commit messages look like `feat(history): …`, `fix: …`, `docs: …`.

## What needs extra care

| Area | Rule |
|---|---|
| Landmark order, dimensions, normalisation, tokenizer | Change code, contract files and tests together ([ARCHITECTURE §11](docs/ARCHITECTURE.md#11-changing-a-contract)) |
| Models and metrics | Follow the gates in [MODELS_AND_DATA §6](docs/MODELS_AND_DATA.md#6-promotion-gates); never cite a metric from another task |
| UI states and copy | Follow [docs/DESIGN_SYSTEM.md](docs/DESIGN_SYSTEM.md); honesty labels are not optional |
| Datasets, weights, recordings | Never commit them; respect licences and credit sources |
| Security-relevant code | Use the checklist in [SECURITY.md](SECURITY.md) |

## Reporting problems

Bugs and ideas: open an issue with steps to reproduce, expected vs actual behaviour, and your environment. Security problems: see [SECURITY.md](SECURITY.md) — do not file them publicly.
