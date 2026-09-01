# SignSage — Security & Privacy

## Core principle: privacy-preserving by construction

- **Landmarks only, never video.** MediaPipe Holistic runs client-side; raw camera
  frames never leave the user's device or reach a server. Only numeric landmark
  coordinates are transmitted. This is a hard requirement, not a configurable option.
- **No raw biometric storage.** Landmark sequences are processed in-memory per session
  and discarded unless a future, explicitly-scoped feature requires persistence (none
  does today — see DATA_MODEL.md).

## Data licensing

- iSign and INCLUDE are used under their respective research-only licenses (iSign:
  "free for research use, not commercial purposes"). No commercial use, no
  redistribution beyond what those licenses permit.
- No new volunteer data collection is in scope for this phase (see PRD.md). If that
  ever changes, informed consent and (per the user's institution's policy) ethics/IRB-
  equivalent clearance apply *before* collecting anything from real signers, even
  informal usability testing.

## Explicit scope boundary: not a substitute for human interpreters

Per Deaf-community research (Krausneker & Schugerl, ICCHP-AAATE 2022; Northeastern
University research, Imai & Alikhani), AI sign-language tools deployed in high-stakes
contexts (legal, medical, emergency) without a certified human interpreter present
carry real risk of harm from mistranslation, and are viewed with justified skepticism by
Deaf advocacy organizations. **SignSage makes no claim to be usable in such contexts**
and should not be presented or deployed as one, even informally.

## Secrets

- LLM API keys (Groq) via environment variables only. Never hardcoded, never committed.
- No secrets belong in this repo, including in notebooks.
- **HuggingFace token (`HF_TOKEN`) — required in Kaggle mode.** The current notebook
  (`signsage.ipynb`) streams the gated iSign pose archive directly
  from HuggingFace rather than reading pre-extracted local files, so it does need a
  token at runtime. It is obtained at execution time, never stored: the notebook reads
  `os.environ["HF_TOKEN"]` first, then falls back to Kaggle's
  `UserSecretsClient().get_secret("HF_TOKEN")`, and raises immediately if neither is
  present rather than proceeding unauthenticated. The token is used only as a `Bearer`
  header on HuggingFace range requests. Nothing writes it to disk, to the knowledge
  repository, or to cell output — but note that **the token does live in kernel memory
  for the whole session**, so do not share a Kaggle notebook *version with output* from
  a session where anything was added that prints it. Set it as a Kaggle Secret; never
  paste it into a cell.
- **Local mode needs no token.** With `SIGNSAGE_DATA_DIR` pointing at an already-
  extracted archive, the notebook performs no HuggingFace call at all — dataset access
  and any `hf auth login` happened separately, out of band, on the storage machine.
- `GROQ_API_KEY`, if set for the optional gloss step, is read from the environment
  only, never hardcoded.

## Deferred (only relevant once built — do not implement early)

- **Session/room codes** (if the two-device relay in ARCHITECTURE.md's "deferred" table
  is ever built): must be short-lived, rate-limited, and unguessable enough to prevent
  a stranger joining someone else's conversation. Not needed for the single-device MVP.
- **Dual-use risk:** none identified for the current scope (assistive communication,
  landmarks-only, no video/biometric identification capability). Revisit if scope ever
  expands toward identification/surveillance use cases — it should not.
