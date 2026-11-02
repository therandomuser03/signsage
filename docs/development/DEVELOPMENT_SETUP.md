# Development Setup

This is a concise local setup reference. For the full interface, security, testing, and deployment workflow, see [DEVELOPMENT.md](DEVELOPMENT.md). For Render model-bundle deployment, see [RENDER_MODEL_DEPLOYMENT.md](RENDER_MODEL_DEPLOYMENT.md).

## Backend

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # only if .env does not already exist
uvicorn app.main:app --reload
```

Backend default: `http://localhost:8000`  
OpenAPI: `http://localhost:8000/docs`

Do not overwrite an existing `.env`. Populate only the environment variables required for the feature being tested. Keep credentials out of source control and logs.

## Frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite default: `http://localhost:5173`

Build check:

```bash
npm run build
```

## Tests

Backend, from `backend/` with its virtual environment active:

```bash
pytest
```

Frontend, from `frontend/`:

```bash
npm test
npm run build
```

Run the relevant tests for the area changed. A passing unit or contract test does not establish model quality or a complete real-camera browser session.

## Important environment cautions

- The repository's `pyproject.toml` declares Python `>=3.11`; the exact working environment recorded in the development notes used Python 3.13. Confirm dependency compatibility before changing Python versions.
- Use the Node version compatible with the current frontend toolchain; the submitted setup audit recorded Node 26.5.0. This is an observed environment version, not a universal requirement.
- Never replace a working `.env` with a template.
- Do not enable `SIGNSAGE_ALLOW_UNPROMOTED_DEV` on production.
- Keep `SIGNSAGE_MODEL_PROMOTED=false` unless the documented model-promotion process has been completed and explicitly approved.
