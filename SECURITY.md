# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's private vulnerability reporting (repository **Security → Report a vulnerability**). If that is unavailable, contact the maintainer privately through the contact details on the maintainer's GitHub profile.

Include what you found, how to reproduce it, and the impact. You will get an acknowledgement; there is no bug bounty. Only the default branch is supported.

> Maintainer: enable *Private vulnerability reporting* in the repository's Security settings so the link above works.

## Privacy model

- **Landmarks, not video, on the live camera path.** The browser extracts landmarks locally and sends only coordinates. Landmark sequences are not kept as a permanent biometric store.
- **The one exception: upload a video.** The Live Interpreter page has an optional file-upload flow (`apiClient.uploadVideo`) that sends the video file you pick to the backend. It must process the file in a temporary location and delete it afterwards, enforce file type and size limits, and tell the user plainly that the file is uploaded.
- **No retention by default.** Any new recording, upload or retention behaviour needs an explicit privacy review before it ships.
- **History is per user.** It exists only for signed-in users, is isolated by row-level security, can be deleted by the user, and sign-out clears protected UI state. Signed-out use works without storing anything.
- **Consent** is stamped at signup.
- Do not claim every path has been independently privacy-tested; check the specific endpoint and current code first.

> **To verify in the backend:** the frontend upload flow exists; confirm the matching route deletes the file in a `finally` block, enforces type and size limits, and is documented in [ARCHITECTURE §5.5](docs/ARCHITECTURE.md#55-rest-apiv1).

## Authentication and authorisation

- Supabase owns identity; the backend never issues tokens. It verifies the bearer token with Supabase (short timeout, TTL cache; failure → 503).
- The WebSocket authenticates in its **first message**; REST uses `Authorization: Bearer`. Guests send no token.
- The backend forwards the **user's** JWT to the database — never a service-role key — so row-level security enforces isolation. Another user's history id returns 404.
- Sign-in and sign-up go straight to Supabase; the backend cannot see login attempts. Password complexity is enforced **client-side**; configure a Supabase password policy for a server-side guarantee.

## Secrets

Never commit, print or paste into notebooks or issues:

`SIGNSAGE_GROQ_API_KEY` / TTS key · `SIGNSAGE_HF_TOKEN` · Supabase **service-role** credentials · the FDMSE-ISL access key · any `.env` file.

- Use environment variables or platform secret stores (Render, Vercel, Colab Secrets).
- Keep keys server-side; the frontend gets only public values.
- Treat any URL that carries a key as a secret; redact it from errors and logs. If a key leaks, rotate it immediately.
- `.env`, `.env.local`, `staging_*`, dataset archives and weights are gitignored.

## Input handling and abuse limits

- WebSocket and REST inputs are validated: non-finite numbers, wrong dimensions, oversized messages and over-long sequences are rejected with documented codes.
- Per-client rate limits apply to word recognition, composition, history writes, login-related calls and TTS (defaults in `backend/app/core/config.py`).
- Security headers are set in the backend middleware. `Permissions-Policy` must **not** block camera or microphone because the backend can serve the frontend.
- CORS must list the production frontend origin explicitly.

## Model and data safety

- Only load checkpoints from a trusted, access-controlled source and verify the SHA-256 — `.pt`/`.pth` loading can execute pickled code.
- Never enable `SIGNSAGE_ALLOW_UNPROMOTED_DEV` on a public service; keep `SIGNSAGE_MODEL_PROMOTED=false` without a documented promotion.
- Datasets stay out of git. FDMSE-ISL and iSign terms forbid redistribution; weights trained on them do not inherit the code's MIT licence ([CREDITS.md](CREDITS.md)).
- Avatar motion clips come from people's recordings: commit only retargeted bone tracks, keep raw video local, and honour deletion requests.

## Safety of use

SignSage is not a certified interpreter. Do not use it for medical, legal, emergency or other high-stakes communication. The interface separates unavailable, uncertain and experimental results from success and never replaces a refusal with a made-up translation.

## Checklist for security-relevant changes

- [ ] No secret in code, tests, notebooks, logs or error messages
- [ ] New endpoint: auth requirement decided, input validated, rate limit considered
- [ ] New data stored: per-user isolation and deletion path
- [ ] No new raw-video path
- [ ] Targeted authorisation or privacy test added ([TESTING](docs/TESTING.md))
