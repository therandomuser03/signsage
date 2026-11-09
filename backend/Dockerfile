# SignSage backend (FastAPI + PyTorch CPU inference).
# Build context: the backend/ directory.   docker build -t signsage-backend .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# CPU-only torch first (the default wheel drags in ~2 GB of CUDA libraries),
# then the app. `pip install .` sees torch already satisfied.
COPY pyproject.toml ./
COPY app ./app
# Ship the demo word-recognition model (8 words + rest) inside the image so
# word mode works without staging models/ separately. The non-root user
# (signsage, uid 10001) must be able to read it.
COPY word_model ./word_model
COPY scripts ./scripts
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
 && pip install . \
 && chmod -R a+rX word_model

# Model weights are NOT baked into the image (they're ~100 MB and change per run).
# Mount or download the promoted deployment bundle here at deploy time.
ENV SIGNSAGE_MODEL_BUNDLE_DIR=/models/signsage_deployment_bundle \
    SIGNSAGE_MODEL_PROMOTED=false

RUN useradd --create-home --uid 10001 signsage
USER signsage

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/v1/health' % os.environ.get('PORT','8000'), timeout=4)"

# Render/Fly/Heroku inject $PORT; default to 8000 locally.
# --proxy-headers + --forwarded-allow-ips='*' so uvicorn trusts the Render
# proxy's X-Forwarded-For: without it every user looks like the same IP and
# per-client rate limiting collapses to a single shared bucket.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
