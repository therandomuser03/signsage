# Privacy and Limitations

## Privacy commitments documented by the project

- Camera capture and MediaPipe landmark extraction are performed in the browser for the documented live interpreter path.
- The documented WebSocket path sends landmark sequences rather than raw camera recordings.
- Do not claim that every upload or video-processing path has been independently privacy-tested. Check the specific endpoint and current implementation before making that claim.
- History is intended to be user-isolated through authentication and database row-level security. The auth/database report classifies some live multi-account verification as conditional; do not describe all auth scenarios as fully proven.
- Never commit `.env` files, Hugging Face tokens, Supabase service-role credentials, Groq keys, or other secrets.
- Do not retain user video by default. Any new recording, upload, or retention behavior requires an explicit privacy review.

## Product limitations

- SignSage is a research prototype, not a certified interpreter or a replacement for qualified human interpreters.
- No continuous ISL-to-English model is currently promoted. `SIGNSAGE_MODEL_PROMOTED=false` remains the production gate.
- The six-word isolated-video pilot is experimental and does not establish broad vocabulary or signer-independent performance.
- The static alphabet classifier is a separate image-classification feature. Its success must not be presented as evidence of continuous sentence translation.
- English-to-ISL output is bounded by the documented gloss vocabulary and available avatar assets. Unknown words or unsupported signs must not be silently presented as correctly signed.
- Model smoke tests prove execution only. They do not prove translation accuracy.
- Medical, legal, emergency, and other high-stakes interpretation use is out of scope.

## Required user-facing behavior

- Distinguish verified output, experimental output, unsupported input, and model-unavailable states.
- Show uncertainty or refusal when evidence is insufficient.
- Never fabricate a translation to make the interface appear functional.
- Advertise only the language coverage and accuracy supported by documented evaluation.
- Preserve history controls and explain what data is retained.
