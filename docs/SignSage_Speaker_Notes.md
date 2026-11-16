# SignSage — Speaker Notes for the 12-Slide Deck

*Presenter cues for the college evaluation. Approximately 30–60 seconds per slide. Tone: natural, student voice — do not read verbatim.*

**30-second opener:**
Hi, I'm presenting SignSage — a bidirectional communication platform for Indian Sign Language, built as my M.Tech CSE Phase-2 evaluation. The idea: an ISL speaker signs at a browser, the computer translates to English for the hearing person; the hearing person types or speaks, and an avatar signs back. Everything is open-source and the ML is gated honestly — no model is served as reliable unless it passes an explicit promotion gate.

**1-minute architecture walkthrough:**
On the left is the browser: React 19 with Vite, capturing camera, running MediaPipe Holistic in WebAssembly. It extracts 543 landmarks per frame — 33 pose, 468 face, 21 per hand. That's 1,629 numbers per frame. The browser streams those over a WebSocket to the FastAPI backend. The backend does inference (CNN classifier on a 60-frame window), and the result comes back as interim and final text. No video ever leaves the browser.
The English-to-ISL side is separate: text comes in, gets chunked into a small gloss list, and a 3D GLB avatar signs the clips back-to-back.

**1-minute ML pipeline:**
The camera pipeline: MediaPipe Holistic pose, face, left hand, right hand. 543 points per frame. We take a rolling window of 60 frames, hand-engineered 122 features (`sign_features_v3`), feed them to a 1D CNN, and softmax gives top-3 predictions. If confidence is below 0.7, the backend honestly reports unavailable rather than inventing. Today the only deployed model is word_v3 — nine words, one signer, experimental.
The ISL→English translation we show in the demo is genuinely limited; explain that clearly.

**Closing:**
SignSage is a working bidirectional ISL prototype with live landmark transport, a real 3D avatar, and an honest model-promotion gate. The next major milestone is training the FDMSE-ISL 200-word baseline — the training notebook exists but hasn't run yet, and we're explicit about that.

---

## Slide 1 — SignSage (title)

**Goal:** name the project; credit the team; signal academic status.

**Script:** M.Tech Phase-2 Evaluation, team from VVCE Mysuru — Anwesa Sinha, Satyaki Dawn, Draksha and Subhadip Das, building on the original V1 ASL prototype. SignSage is a bidirectional communication platform for Indian Sign Language users and non-signers. The one idea behind the project is honest ML: show what's verified, refuse to serve what isn't.

**Remember:** open with confidence, not with a caveat. The honesty framing goes in slide 3.

**Likely questions:** Why "SignSage"? → Sign + Sage (scholar of signs). What is a bidirectional platform? → ISL→English and English→ISL in one tool.

---

## Slide 2 — Problem Statement & Motivation

**Goal:** frame the social problem and the technical lever we chose.

**Script:** India has roughly 98 lakh deaf people — wider than Ireland. There is still no general-purpose bidirectional ISL translation service. The blocker used to be that every commercial offering wanted server-side video, which is slow, expensive, and private-hostile. With MediaPipe running in WebAssembly in the browser, that cost collapses: we ship one thousand numbers per frame instead of frames themselves. The social motivation is access; the technical motivation is that browser WASM plus a commodity backend can actually do this at ~30 fps.

**Remember:** landmarks only — raw video never crosses the wire. That is sign-off-grade for a privacy review.

**Likely questions:** 98 lakh — source? → ISLRTC (Indian Sign Language Research and Training Center). Why browser and not native app? → Deployment cost, latency, privacy, and reach (any laptop with a camera).

---

## Slide 3 — Objectives & Proposed Solution

**Goal:** put the four objectives and the two-directional design on one slide.

**Script:** Four objectives. One, build a real browser↔backend bidirectional app, not a mock. Two, data stays on device — landmarks only, never video. Three, no model is served as reliable unless it passes an explicit promotion gate — that single idea shapes the whole codebase. Four, establish a reproducible training path for a 200-word ISL recogniser using the FDMSE-ISL dataset. That is why every model we ship is gated, and why the demo always shows the honest refusal state.

**Remember:** objective 3 is the punchline — gate-first ML.

**Likely questions:** Is the promotion gate a criticism of the team? → It is a design principle, not a claim about quality. How is it enforced? → `SIGNSAGE_MODEL_PROMOTED=false` in production, and the backend returns refusals.

---

## Slide 4 — System Architecture

**Goal:** lay out frontend, backend, data, flow.

**Script:** Left — React 19 with Vite, Tailwind, shadcn/ui. MediaPipe Holistic runs in the browser's WebAssembly; camera at ~30fps Â→ browser downsamples to ~10fps; WebSocket streams to backend. Middle — FastAPI plus PyTorch on Python 3. Right — Supabase for optional auth and history with row-level security, Groq for STT/TTS and gloss polish. Vercel frontend, Render backend. Tests: 80 frontend, 292 backend. That's the actual stack.

**Remember:** do not invent numbers. Vite 8, React 19, tail 4, FastAPI, PyTorch CPU, Supabase — those are real.

**Likely questions:** Why not WebGL for landmarks? MediaPipe already done in WASM. Why not GPU backend? CPU is cheaper and sufficient for 60-frame clips.

---

## Slide 5 — Sign → Text: Recognition Pipeline

**Goal:** explain the two tracks and the honest state of the live model.

**Script:** Two tracks. Track A is live webcam: MediaPipe Holistic pose/face/hands, 543 landmarks per frame, WebSocket into FastAPI, PyTorch inference on a 60-frame window. Track B is static image classification — that's how the alphabet classifier reaches ~87%. The vocabulary is small: today word_v3 is nine signs — but honestly, adapted to one signer, and not promoted. The word CNN on static images is unreliable and we're explicit about that.

**Remember:** alphabet CNN is verified but not retrained; word CNN is not credible; word_v3 is experimental and unpromoted. Never mix them up.

**Likely questions:** What's the accuracy? → Alphabet ~87% on its test split, 9-word recogniser zero verified evaluation at this venue. Why not? → Adapted to one signer; signer-independence not established. What does the gate do? → Below-confidence returns refused rather than faked.

---

## Slide 6 — Dataset & Model Development

**Goal:** ground everything in FDMSE-ISL and today's honest status.

**Script:** The new dataset is FDMSE-ISL — Patra et al., 2025, Pattern Analysis and Applications. 2,002 classes, 40,000+ videos, frontal RGB view only, signer-independent split 10/2/8. ~30 GB archive, portal-gated. We staged and synthetically smoke-tested a complete training pipeline — download resumable, 206/200/timeout-safe — but no real training has been executed on this machine. The 200-word BiGRU target is a plan, not a trained model.

**Remember:** the only verified accuracies today come from the alphabet CNN (~87%) and word_v3's training split (0.98, but single-signer and not promoted).

**Likely questions:** Did you train the 200-word model? → No — staged and validated; needs Kaggle/Colab + gated key. When? → Next milestone. What's the 0.98 accuracy? → On the single-signer validation split only. What does the gate enforce? → 80% accuracy/80% recall/60% per-class recall, signer-independent split required for promotion.

---

## Slide 7 — English → ISL Processing

**Goal:** show the working English→ISL path with its limits.

**Script:** English text in, we tokenize, lookup every word in the 14-word gloss table, optionally Groq-polish a sentence to the same vocabulary — never invent — then play the corresponding 3D avatar clips. The avatar is a real human GLB we load from Three.js. Understood as a *gloss-bounded* path: words we don't own get dropped. Unknown words don't get made up.

**Remember:** limited vocabulary is the honest answer, not a bug.

**Likely questions:** Why not full English-to-ISL-grammar mapping? → ISL-grammar mangling is an LLM-scale problem for 200+ classes — not in scope for this phase. What's your fallback? → deterministic lookup always; comma-join fallback for prose.

---

## Slide 8 — 3D Avatar Integration

**Goal:** show the realistic GLB and be explicit about what articulation does and does not mean.

**Script:** The model is signsage-human.glb — 6.5 megabytes, MakeHuman-style rig with 164 joints, 34 morph targets. In the dev avatar page you can see fingers flex, wrists extend, head nod/tilt, facial expressions via visemes. It's real articulation. That's not the same as ISL-correct signs — there's no gloss→motion mapping wired in yet. Don't confuse the two.

**Remember:** the distinction slide is here for a reason.

**Likely questions:** Can you show the avatar signing "hello"? → Not as a linguistic ISL sign — we can show HELLO gloss → GLB lookup, but the motion is a demo, not a validated ISL performance. When will it sign correctly? → after FDMSE-ISL training + gloss mapping to motion — that's the next milestone.

---

## Slide 9 — Implementation & Tech Stack

**Goal:** list the actual tools so evaluators can verify the claim they walked in with.

**Script:** Frontend: React 19 + Vite + Tailwind + TypeScript. Backend: FastAPI + PyTorch (CPU) + WebSocket. ML: MediaPipe Holistic, cnn-v3 recognizer, alphabet CNN. Infra: Vercel frontend, Render backend, Supabase auth, Groq APIs. Tests: 292 backend, 80 frontend, Playwright E2E green. They're existing tools — nothing exotic, and everything above runs locally on a laptop without internet.

**Remember:** list exact versions calmly — that's what "hands-on" looks like under questioning.

**Likely questions:** Can I run it? → Yes, just clone and `uvicorn app.main:app --reload`. What's the deployment path? → Vercel → backend Render → Supabase. Is Supabase mandatory? → No — sign-out mode works.

---

## Slide 10 — Features, Testing & Achievements

**Goal:** present the verification evidence.

**Script:** Here are the test results — no rounding, no cherry-picking: 292 backend passing, 80 frontend passing, 7 of 7 Playwright. Console errors: zero (after fixes). FPS measured: ~9–11 FPS sustained. These are numbers from this session, not aspirations.

**Remember:** the qualifier is part of every claim — "this session", "on this machine".

**Likely questions:** How long were the tests run? → Minutes to seconds; not a load test. Any flakiness? → Frontend tests occasionally serialize slowly on first run but pass. What about an FPGA/mobile target? → Future work.

---

## Slide 11 — Limitations & Future Scope

**Goal:** name the limits and the roadmap.

**Script:** Today: no sentence-level sign recognition, one signer adapted model unpromoted, limited 14-gloss output, no viseme lip-sync, no CI pipeline. Next: train FDMSE-ISL 200-word BiGRU, gate promotion, studio recordings from ISLRTC, wire gloss to avatar motion, publish evaluation in a peer-reviewed venue. We don't hide the limits — the entire point of the project is showing what can be said today and what can't.

**Remember:** future scope is tied to actual dataset names and papers, not wishful thinking.

**Likely questions:** When will the 200-word model be ready? → next milestone; depends on Kaggle/Colab compute. What's blocking? → Compute budget and CPU-only backend; no GPU until Render paid tier. Who else has signed this? → One signer was recorded and is the basis for word_v3.

---

## Slide 12 — Conclusion & References

**Goal:** close with the honest win.

**Script:** SignSage today is a verified, honest bidirectional ISL prototype: live transport, real avatar, real test results, explicit promotion gate. The academic thread — Patra et al.'s FDMSE-ISL paper — is what the next phase builds on. Thank you; questions welcome.

**Remember:** don't let the room turn into a defence — restate the contribution crisply.

**Likely questions:** What is your single biggest achievement? → The honesty gate pattern applied to model promotion — it's why every table in this deck is real, not aspirational. What will you publish? → The FDMSE-ISL 200-word training-set evaluation.
