# backend/models/

This directory intentionally contains **no model weights**.

The Sign->Text checkpoint is loaded at runtime from wherever
`SIGNSAGE_MODEL_BUNDLE_DIR` points (see `.env.example` and
`app/core/config.py`) -- typically a path under
`research/artifacts/<run-id>/signsage_deployment_bundle/`, not this
directory. See `app/infrastructure/storage/model_storage.py` for the
resolution logic.

## Current status

As of research run `352868924` (2026-09-26, notebook `signsage.ipynb`):
**NO-GO**.

- Test WER: 1.0046
- Test BLEU: 0.24
- Test ROUGE-L: 0.0891
- Output diversity: 0.99% unique outputs (120 / 12,153 test examples)
- Top single prediction ("the police said that the government has been
  arrested.") covers 44.84% of all test outputs -- severe collapse
- Notebook's own verdict: `RESULT: NO-GO for deployment. Do not connect
  this checkpoint to the webcam backend yet.`

`SIGNSAGE_MODEL_PROMOTED` must stay `false` until a future run's
`evaluation_results.json` genuinely passes the deployment gate
(`transformer_beats_trivial` AND `output_diversity_ok`, per the notebook's
own Section 17 check). See
`research/manifests/models/signsage-s2t_v352868924.yaml` and
`docs/ai-ml/model-contract.md`.

With `SIGNSAGE_MODEL_PROMOTED=false` (the default), `/ws/translate`
and `POST /api/v1/translation/sign-to-text` will refuse to serve
translations (`ModelNotPromotedError`) rather than silently returning
NO-GO-quality output as if it were reliable.

## Promoting a future GO run

1. Confirm `evaluation_results.json` passes both deployment-gate checks.
2. Point `SIGNSAGE_MODEL_BUNDLE_DIR` at that run's
   `signsage_deployment_bundle/` directory.
3. Set `SIGNSAGE_MODEL_PROMOTED=true`.
4. Run `python -m scripts.model_check` to confirm the bundle loads and
   passes the activity-gate self-test before deploying anywhere.

Do not copy checkpoint files into this directory, and do not commit them to
git -- see `research/artifacts/` and the project's `.gitignore`.

## Avatar gloss generation (English -> ISL, the OTHER direction)

This is unrelated to the Sign->Text model above and works today regardless
of the NO-GO status: `POST /api/v1/text-to-sign` turns English text into a
sequence of your 14 existing avatar signs. Three stages, first one
configured and working wins:

1. **Groq** (if `SIGNSAGE_GROQ_API_KEY` is set) — free tier, get a key at
   [console.groq.com/keys](https://console.groq.com/keys), paste it in `.env`. Done.
2. **Ollama** (if `SIGNSAGE_OLLAMA_BASE_URL` is set) — a local LLM, used only
   if Groq is unset or fails/rate-limits. Install from [ollama.com](https://ollama.com),
   run `ollama pull llama3.2 && ollama serve`, then set
   `SIGNSAGE_OLLAMA_BASE_URL=http://localhost:11434`.
3. **Deterministic word-lookup** — always available, no network, no key.
   Exact-word matches only.

**What the LLM stages add:** they still only ever pick from your 14
pre-authored signs (see `app/data/sign_vocabulary.json`) — they don't
invent new ones, that still needs new animation clips. What they fix is
phrasing: "could you assist me finding water" shares zero literal words
with HELP, so the deterministic stage misses it; an LLM reads intent.
Every LLM reply is filtered against the vocabulary before use (never
trust generated output blindly), and any failure — no key, network error,
rate limit, malformed reply — falls straight through to the next stage,
so this feature can only degrade gracefully, never break translation.

Verified in this session: 11 unit tests against mocked Groq/Ollama HTTP
(success, vocabulary-filtering of hallucinated IDs, 429+retry-after
handling, network failures, malformed responses) plus one live end-to-end
run — a real FastAPI server talking to a real (mock) Groq-shaped HTTP
server over the actual network stack.
