# SignSage — Security, Privacy & Safety

## 1. Privacy principle

**Landmarks, not video.**

MediaPipe Holistic is intended to run in the browser. The translation transport carries numerical landmark data rather than raw camera frames.

No camera-upload endpoint is part of the documented translation architecture.

## 2. Stored data

Translation history is application data and must be scoped to the authenticated user.

Raw camera video is not stored by the translation backend.

Landmark sequences are not intended as a permanent biometric store.

## 3. Authentication

Supabase handles authentication. Backend authorization must verify authenticated identity before returning account-scoped history.

Row-level database protections should remain enabled for user-owned records.

## 4. Secrets

Never commit:

- `GROQ_API_KEY`
- `HF_TOKEN`
- Supabase service-role credentials
- other private credentials

Use environment variables or the platform's secret store.

The Hugging Face token used by the research notebook must remain a runtime secret and must not be printed into shared notebook output.

## 5. Dataset licensing

The current documentation records iSign as CC BY-NC-SA 4.0 and warns against committing the dataset, extracted pose files, or Hugging Face caches to version control.

Dataset access instructions belong in research documentation; the dataset itself does not.

### 5a. Model weights inherit the data license (review before any public/commercial use)

The repository code is MIT-licensed, but Sign→Text **weights trained on iSign
derive from CC BY-NC-SA 4.0 data** and do NOT automatically carry the code's
MIT terms. Before any public or commercial deployment of a trained checkpoint,
review and record answers to:

1. Whether the iSign license (including NonCommercial and ShareAlike) permits
   the intended use of the derived weights, or a separate agreement is needed.
2. Whether serving predictions from those weights (including via the Render
   download-at-start bundle) counts as commercial use in the deployment context.
3. Attribution and ShareAlike obligations for the bundle and any adapted model.
4. Whether the tokenizer/vocabulary derived from dataset text needs the same treatment.

This does not block local research development (training, evaluation, dev-only
inference). It does block presenting an iSign-trained checkpoint as cleared
for unrestricted commercial deployment. The promotion gate must include this
license review alongside the quality evidence.

## 6. Safety boundary

SignSage must not be presented as a replacement for certified human interpreters in legal, medical, emergency, or other high-stakes contexts.

The current research results specifically document repeated and incorrect Sign→Text outputs. A fluent-looking sentence can therefore still be wrong.

## 7. UI safety

The UI should:

- distinguish unavailable models from successful inference,
- surface tracking and connection errors,
- avoid false certainty,
- avoid presenting experimental pose generation as validated signing,
- preserve the distinction between demo output and research evidence.

## 8. Future security work

If two-device relay is introduced later, the security review must cover:

- room-code entropy,
- expiration,
- authorization,
- session isolation,
- WebSocket reconnect behavior,
- abuse/rate limiting,
- cross-user data leakage.
