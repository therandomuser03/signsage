# Live Interpreter performance notes

## Bottleneck identified

The main frontend bottleneck was the render loop in `frontend/src/features/translate/TranslatePage.tsx`.

- The page scheduled a requestAnimationFrame loop for the camera preview.
- Every frame called `HolisticExtractor.detect()` with MediaPipe Pose, Face, and Hand detectors in parallel.
- The code then sent only a subset of frames to the backend, but the detector work itself still ran at full animation-frame cadence.
- This created redundant per-frame inference and caused unnecessary CPU/GPU load while the model could only consume a limited subset of frames.
- There was no deliberate throttling between detector runs, so camera preview and backend submission were effectively coupled to the browser refresh rate rather than real inference needs.

## Changes made

- Added a detection throttle in the animation loop so MediaPipe runs on a bounded cadence instead of every animation frame.
- Kept overlay drawing tied to the same detector cadence, preserving the preview without running full landmark extraction on every frame.
- Added a `Browse Signs` sheet with a search input, alphabetical A–Z grouping, and a clear empty state using the existing `Sheet` component and UI primitives.
- Populated the sign list from the project’s procedural-sign vocabulary metadata, clearly distinguishing `live-supported` entries from `browse-only` entries.
- Added tests covering grouping, case-insensitive filtering, empty results, and category distinction.

## Remaining limitations

- This change removes redundant per-frame detector work but does not claim a verified recognition speedup because no before/after benchmark was performed on the same hardware and browser setup.
- The live interpreter still does not provide general continuous ISL sentence translation; the app only supports the project’s limited pilot vocabulary and constrained recognition rules.
- The Browse Signs panel is a catalog preview, not a claim of live recognition coverage.
