# SignSage — Design

## Status

No standalone UI exists yet. This document captures design *principles* to build toward
once Phases 1-2 (see PHASES.md) prove the underlying pipeline works — it is not a
finished spec, and should be revisited (probably substantially rewritten) once there's
something on screen to react to.

There *was* a throwaway HTML/JS client inside the Phase-0 notebook (Section 18, gated
behind `RUN_LIVE_DEMO = False`): a two-panel layout with a mirrored webcam feed and a
large-text caption area on one side, a text input and a `<canvas>` on the other. It has
since been **removed from the notebook** (see DECISIONS.md). It was a plumbing spike, not
an implementation of this document, and it violated two principles below — recorded here
so the real UI doesn't inherit either:

- **It rendered a single static frame, not an animation.** The canvas drew only
  `frames[0]` of the returned pose sequence as a scatter of points, so "skeleton
  animation" was never demonstrated — and the points were drawn unconnected, so it wasn't
  a skeleton either.
- **It failed silently.** There was no low-confidence warning and no error surface; a
  dropped socket just retried in the background. See the `error` message type in API.md,
  which the spike did not implement.

Do not treat that spike as a starting point for the real UI without fixing both.

## Who this is for

Primarily: a Deaf/HoH ISL signer and a hearing non-signer, sharing one device, having a
conversation neither could have alone. Secondarily: anyone evaluating/demoing the thesis
project (guide, committee).

## Core interaction principles

- **Turn-taking must be obvious.** On a single shared device, it should always be clear
  whose "turn" it is (camera active for signing vs. text/speech input active), without
  relying on reading instructions.
- **Captions are the primary UI, not a bonus.** Text output from Sign->Text needs to be
  large, high-contrast, and persistent long enough to read — this is an assistive tool,
  not a chat app with a message that scrolls away.
- **The skeleton animation is a research placeholder, not a product feature.** Do not
  spend design effort making it look "finished" — an honest stick-figure/skeleton
  communicates the actual state of the technology. Dressing it up as an avatar this
  early would misrepresent what's been validated (see PRD.md non-goals).
- **Fail loudly, not silently.** If landmark confidence is low (bad lighting, hand out
  of frame, camera too far away), say so in the UI rather than silently producing a
  low-quality guess. This matters more here than in most apps because a wrong
  translation in an assistive-communication context can cause real harm, not just
  annoyance.

## Accessibility requirements (non-negotiable, not aspirational)

- Minimum text size and contrast ratio suitable for low-vision users (this tool already
  serves a disability community; do not neglect others).
- No interaction that requires audio alone to understand system state (a Deaf user is a
  primary user, not an edge case).
- Works with keyboard/switch input for the text-input half of the conversation.

## Explicit non-goals for design

- No photorealistic avatar (see PRD.md).
- No gamification, streaks, onboarding flows, or other product-growth patterns — this
  is a research tool, not a consumer app being optimized for engagement.
