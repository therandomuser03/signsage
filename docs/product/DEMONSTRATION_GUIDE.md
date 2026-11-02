# SignSage Demonstration Guide

**Purpose:** demonstrate only the flows supported by the current evidence. This guide does not claim that continuous sentence translation is working.

## Before the demonstration

1. Start the backend and frontend using [DEVELOPMENT_SETUP.md](../development/DEVELOPMENT_SETUP.md).
2. Check `GET /api/v1/health` and `GET /api/v1/capabilities`.
3. Confirm the model status is reported honestly. Continuous sign-to-text should remain unavailable while `SIGNSAGE_MODEL_PROMOTED=false`.
4. Test the exact browser, camera, microphone, and network setup that will be used during the presentation.
5. Keep a short screen recording or screenshots of verified flows as a fallback, but label recorded material clearly.

## Recommended live-demo sequence

### 1. Product and architecture overview

Explain the intended bidirectional workflow:

- ISL input is captured in the browser and converted to landmarks.
- The backend exposes translation, capability, history, and supporting APIs.
- English input maps to a bounded gloss vocabulary and available avatar assets.

Make clear that the intended continuous translation pipeline is not yet validated.

### 2. Static alphabet image classification

This is a separate, documented working feature.

- Use the image-classification mode and a known test image.
- Show the predicted letter, confidence, and acceptance/rejection state.
- Describe it as static image classification, not continuous signing.

### 3. English-to-ISL gloss/avatar flow

- Enter a sentence containing supported vocabulary.
- Show the deterministic gloss mapping and the available avatar output.
- Explain the bounded vocabulary and what happens to unsupported words.
- Do not claim unrestricted English-to-ISL translation or natural fluent signing.

### 4. Honest model-unavailable behavior

If useful, show the capability/status response or explicit unavailable state for continuous sign-to-text. Explain that the promotion gate prevents an unvalidated model from returning misleading translations.

Do not present synthetic smoke-test text as a real translation.

## Do not demonstrate as completed

- Continuous ISL sentence-to-English translation as an accurate working feature.
- Broad-vocabulary or signer-independent translation.
- The Sid-V5 model as integrated or available; checkpoint access and end-to-end translation remain unverified.
- The six-word isolated-video pilot as production-ready.
- The 20-class static word image classifier as reliable.
- Full real-camera continuous browser inference unless it has been separately tested on the actual presentation machine.

## Claims suitable for the presentation

Use statements such as:

- "SignSage is a bidirectional ISL-English research prototype."
- "The application includes a verified static alphabet image-classification flow."
- "The English-to-gloss/avatar path currently supports a bounded vocabulary."
- "Continuous ISL-to-English translation remains under evaluation; no model has been promoted."

Avoid claiming real-time sentence translation accuracy, complete ISL coverage, or certified accessibility performance.

## Evidence to have ready

- Backend health/capabilities response.
- Screenshot or live result from alphabet classification.
- A supported English-to-gloss/avatar example.
- Current model status and explicit limitations.
- A brief explanation of the next model-validation gates.
