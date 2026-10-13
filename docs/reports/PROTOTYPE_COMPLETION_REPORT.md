# SignSage — Prototype Completion Report

Date: 2026-10-12. Archive `results.zip` and all checkpoints untouched.
No commits made (working tree only).

## What works now

**Direction B — Text/Speech → Sign (fully demonstrable, no assets needed)**
- Typed text → gloss mapper (dictionary offline, or backend-assisted) →
  supported-sign lookup → animation queue → visible 3D avatar playback.
- New procedural avatar (`frontend/.../ProceduralAvatarViewer.tsx` +
  `proceduralSigns.ts`): cartoon figure with visible hands, upper body and
  face; a distinct motion for each of the 14 manifest signs; plays sequences
  in order with progress, Play/Stop/Replay/Clear, no overlap or stale
  playback (generation tokens on both VRM and procedural paths).
- VRM path preserved: if `signsage.vrm` + GLB clips are ever bundled, the
  page uses them automatically; otherwise it uses the procedural figure and
  says so on screen ("Procedural motion preview — not authored ISL").
- Microphone input via the browser's built-in speech recognition (no new
  dependency); unsupported browsers get an honest "type instead" message.
- Unmatched words are listed as "dropped, not guessed" (unchanged).

**Direction A — ISL → English (limited but real, 5-sign vocabulary)**
- The transformer stays correctly gated (NO-GO, no servable bundle — see
  `MODEL_DEPLOYMENT_DIAGNOSIS.md`; gate code untouched).
- New checkpoint-free fallback (`backend/app/ml/inference/heuristic_signs.py`):
  majority vote over static one-hand shapes from live (T,543,3) landmarks.
  Vocabulary: open palm→Hello, thumbs-up→Yes, fist→No, index-only→You,
  index+middle→Help. All IDs are in the avatar vocabulary, so signs
  round-trip to the Text→Sign page.
- Served through the existing WS/REST/session/history path (rolling interim
  + sentence boundary + repeat-agreement dedup unchanged). Empty frames →
  honest `no_hand_detected`; mixed/unknown shapes → `unrecognized_shape`;
  nothing is fabricated.
- `/capabilities` now reports `sign_to_text_source` (`transformer` |
  `heuristic-fallback` | `unavailable`) and `fallback_vocabulary`; the Live
  Interpreter page shows a "Recognizable signs" hint panel with how to
  perform each sign. `SIGNSAGE_FALLBACK_ENABLED=false` restores the old
  refuse-with-`MODEL_UNAVAILABLE` behavior.

## Verified test results (all executed 2026-10-12)

- Backend: `tests/unit` 91 passed (8 new heuristic tests); `tests/contract`
  20 passed; `tests/integration` 51 passed (3 new/updated fallback tests);
  `tests/ml` 1 passed. (Had to `pip install pytest-asyncio`: it is in
  `requirements.txt` but was missing from this machine; 32 async tests
  failed before that for that reason alone.)
- Live TestClient probes: capabilities → `heuristic-fallback` + 5-sign vocab;
  REST fist→`No` accepted, palm→`Hello` accepted, zeros→rejected;
  WS 80 thumbs-up frames → interim `Yes` + final `Yes`, no error/close;
  text-to-sign `Hello thank you water please xyzzy` → 4 glosses + unmatched.
- Frontend: `tsc -b` clean, `npm test` 20/20 (5 new motion tests),
  `npm run build` succeeds (`dist/` written).
- NOT performed: real-camera recognition and in-browser avatar viewing
  (this machine has no camera; Playwright's Chrome is not installed and
  `npx playwright install chrome` needs sudo). These are the manual demo
  steps below.

## How to run and demonstrate

Backend (from `backend/`, venv with `requirements.txt`):

```bash
cp .env.example .env   # defaults work: fallback on, auth optional
uvicorn app.main:app --reload --port 8000
```

Frontend (from `frontend/`): `npm install && npm run dev` (Vite, default
`:5173`; point `VITE_API_URL`/WS config at the backend).

Demo A — sign to text: open Live Interpreter → Start session (allow camera)
→ hold one shape steady ~2 s (see on-page hint panel) → Stop to finalize.
Expect the English word + confidence; waving randomly gives "Not confidently
recognized", never a guessed sentence.

Demo B — text to sign: open Text → Sign → type `hello thank you water` →
Generate (gloss chips) → Play (figure performs 3 motions in order) →
Replay/Stop/Clear. Mic button dictates where the browser supports it.

## Known limitations (explicit boundaries)

- Sign→text covers 5 static shapes only — no motion signs, no sentences, no
  general ISL translation. Anything else is rejected, not guessed.
- Procedural avatar motions are distinct placeholders, not authored ISL.
  Sentence-level signing needs real vocabulary, grammar and clips.
- Transformer path unchanged: still NO-GO with no servable bundle; a future
  `beats_trivial AND diversity_ok` run + `export_bundle` + `model_check.py`
  + promotion is still the route to full-vocabulary recognition.
- `results.zip` was only listed, never extracted or modified.
