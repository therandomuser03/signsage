# SignSage — Decisions Log

**Status:** Current decision record

## Purpose

Record non-trivial architectural/research choices so agents do not repeatedly relitigate settled decisions.

| Decision | Rationale | Status |
|---|---|---|
| FastAPI over Flask | Native async/WebSocket support and suitable streaming model | Adopted |
| React + TypeScript + Vite | Existing frontend architecture and suitable responsive application stack | Adopted |
| Single-device MVP | Avoid remote relay/session complexity before the core communication workflow is established | Adopted |
| WebSocket for translation loop | Persistent transport suits landmark streaming better than per-frame HTTP | Adopted |
| Browser-side MediaPipe | Keeps raw camera processing local and sends landmarks rather than video | Adopted |
| GRU baseline before Transformer | Establish a simple learning baseline before increasing model complexity | Adopted |
| Plain Transformer before GAT | Current research objective does not justify graph-specific complexity without evidence of need | Adopted |
| 543 image-space landmarks | Uses pose + face + both hands while excluding the duplicated world-pose block | Adopted |
| `pose.header.components` introspection | Prevents incorrect hardcoded feature masks during face/non-manual-marker ablation | Adopted |
| Split by `video_id` | Reduces leakage from multiple segments of the same source video | Adopted |
| Groq with local/deterministic fallback | Allows text→gloss functionality to degrade gracefully when an external model is unavailable | Adopted |
| Research notebook separate from serving | Training/evaluation and application serving have different lifecycle and reproducibility requirements | Adopted |
| No raw-camera upload in translation path | Privacy-by-construction design | Adopted |
| No two-device relay in current scope | Deferred until the single-device workflow justifies the added infrastructure | Deferred |
| No photorealistic avatar requirement | Research focus is communication pipeline and measurable evaluation | Deferred/out of scope |
| No high-stakes deployment | Current model evidence does not justify such use | Out of scope |

## Research-specific rules

### Model promotion

A checkpoint is not promoted merely because it:

- trains,
- saves,
- loads,
- runs inference,
- improves one metric,
- or passes software tests.

Promotion requires a documented evaluation showing acceptable behavior against appropriate baselines and failure checks.

### Text→Pose evaluation

DTW/MSE must be accompanied by motion-energy analysis because near-static poses can produce deceptively favorable similarity scores.

### Documentation

Architectural changes must be added here rather than silently replacing an existing decision.
