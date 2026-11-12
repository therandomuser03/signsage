# SignSage — Cleanup & Honesty Audit (pre-presentation)

Done on `master`, working tree only. **Nothing committed, pushed, staged, or stashed.** No `.env` / `render.yaml` / `SIGNSAGE_MODEL_PROMOTED` touched. No SQL run. No secret values printed.

---

## 1. INVENTORY

At the start of this session the tree was clean and `origin/master == HEAD == c3e071a`.

```
$ git status --short            -> (empty, clean)
$ git diff --stat               -> (empty)
$ git log --oneline -5
c3e071a Refactor and enhance various pages and components
9df00e3 feat(history): enhance history tracking with word recognition details
99d5a24 test(api): add unit tests for ApiClient's history requests
c7de8dc Add sign feature extraction and testing for S3 recordings
e66f512 docs update and ISL word-recognition research
$ git log origin/master..HEAD --oneline  -> (empty)
```

Verdict: every one of footer / route renames / Help page / password rules / rate limits / security headers / "prototype/demo/pilot" removal is **committed and pushed** (they're all in HEAD on master, matching origin/master). **Nothing was only in the working tree** at the start.

(The edits I made below are now uncommitted working-tree changes, per your instruction not to commit.)

---

## 2. HONESTY (frontend)

Removed untrue research-prototype claims and kept only links to real pages.

### Footer (`frontend/src/components/footer.tsx`)
- Kept layout, kept only real-page links: How it works (`/how-it-works`), Capabilities (`/capabilities`), Help (`/help`), About (`/about`), Translation Log (`/app/translation-log`), Privacy (`/privacy`).
- Removed `href="#"` links that went nowhere / made untrue claims: Case studies, Safety & AI models, Resource Center, Status, Developer Docs, Medical Advisory, Research, Accessibility commitment, HIPAA & privacy, Security posture, Terms of service, GDPR compliance, and the LinkedIn/Twitter/X/GitHub socials (all `href="#"`).
- Brand tagline: `"Clinical-grade AI bridging..."` → `"Research prototype for Indian Sign Language communication. Not for medical, legal or emergency use."`
- Bottom: `"Clinical communication platform for sign language."` removed; now `© 2026 SignSage.`

### AboutPage
- `"SignSage is an Indian Sign Language communication platform — ..."` → `"SignSage is a research prototype for Indian Sign Language communication — ... Not for medical, legal or emergency use."`

### LandingPage
- `"SignSage is an Indian Sign Language communication platform — real-time ..."` → `"SignSage is a research prototype for Indian Sign Language communication — real-time ... Not for medical, legal or emergency use."`

### HelpPage (`frontend/src/features/help/HelpPage.tsx`)
- Removed the whole "Train the model / Teach SignSage new words / Kaggle training notebook" user-facing section.
- Every word card now links to the portal homepage `https://indiansignlanguage.org/` instead of the unverified `/dictionary/<word>`.
- Kept the per-word URL as a clearly marked, unverified constant (`ISL_BASE_URL = "https://indiansignlanguage.org/dictionary"`) and added a third-party disclaimer ("The portal belongs to a third party; SignSage only links to it and is not affiliated with it.").

### PrivacyPage / AboutPage vs `docs/product/PRIVACY_AND_LIMITATIONS.md`
- Privacy camera section no longer claims "the backend never needs — and never receives — your video" absolutely (the experimental video-upload path does receive video). Scoped to the live camera/interpreter path.
- The "one rule" alert now says camera data that never leaves the device can't leak from that path; the live camera path sends landmarks, not raw video.
- About privacy section likewise scoped to the live camera path. These are now consistent with the doc (don't claim every upload/video-processing path is privacy-tested).

### Word-mode honest label
- Backend `word_recognition_label`: `"word-level recognition; not sentence translation"` → `"word-level recognition, adapted to one signer; not sentence translation"` (displayed on the interpreter page when word mode is on).

### Full UI string change list (old → new)

| File | Old | New |
|------|-----|-----|
| footer.tsx | `Clinical-grade AI bridging sign language, text, and visual representations.` | `Research prototype for Indian Sign Language communication. Not for medical, legal or emergency use.` |
| footer.tsx | `Clinical communication platform for sign language.` (bottom) | removed; now `© 2026 SignSage.` |
| footer.tsx | 14 `href="#"` links (Case studies, Safety & AI models, Resource Center, Status, Developer Docs, Medical Advisory, Research, Accessibility commitment, HIPAA & privacy, Security posture, Terms of service, GDPR compliance, LinkedIn, Twitter/X, GitHub) | removed; only 6 real-page links remain |
| AboutPage.tsx | `SignSage is an Indian Sign Language communication platform — ...` | `SignSage is a research prototype for Indian Sign Language communication — ... Not for medical, legal or emergency use.` |
| LandingPage.tsx | `SignSage is an Indian Sign Language communication platform — real-time ...` | `SignSage is a research prototype for Indian Sign Language communication — real-time ... Not for medical, legal or emergency use.` |
| HelpPage.tsx | "Train the model / Teach SignSage new words" Kaggle section | removed entirely |
| HelpPage.tsx | word links → `https://indiansignlanguage.org/dictionary/<word>` | word links → `https://indiansignlanguage.org/` (homepage) |
| capabilities.py | `word-level recognition; not sentence translation` | `word-level recognition, adapted to one signer; not sentence translation` |
| PrivacyPage.tsx | camera: "...has nowhere to send pictures... the network protocol only accepts numerical landmark arrays." | scoped to live camera path; only numerical landmark arrays travel to the backend for that path |
| PrivacyPage.tsx | alert: "If data never leaves your device... backend never needs — and never receives — your video." | alert: scoped to live camera interpreter path sends landmarks, not raw video |
| AboutPage.tsx | privacy: "...raw video never leaves the device, because the system never needs it to." | scoped: "For the live camera path, only those numerical representations travel to the backend." |
| AuthPages.tsx | login used same complexity as signup (min 8 + UC + LC + digit + special) | login: email + non-empty password only; signup keeps full complexity |

---

## 3. CODE SAFETY ("prototype/demo/pilot" mass replacement)

```
$ git diff   (filtered for removed lines containing prototype|demo|pilot in backend/ and frontend/)
```
Result: **working tree was clean at the start**, so `git diff` is empty — there are no removed lines in the working tree to restore. The mass replacement was already committed.

I confirmed the committed code still contains the pilot/smoke bundle detection intact (no identifier/env-var/test-name/bundle-detection/logic changed):
- `backend/app/ml/models/registry.py` (unchanged): `adapter.training_scope in ("pilot_subset", "smoke_synthetic")` still blocks serving a pilot/smoke bundle as a promoted model.
- `backend/tests/integration/test_training_scope_guard.py` and `backend/tests/unit/ml/test_training_path_probe.py` still reference pilot/smoke scopes.

**Backend unit tests:** run with `python -m pytest tests` (backend `.venv`). Result: **215 passed, 58 failed, 4 errors**. I verified these 58 failures + 4 errors are **pre-existing/environmental** — running the same suite on the HEAD versions of my changed files produced the identical count (58 failed / 215 passed / 4 errors). My backend changes introduced **zero new failures**. The 4 errors are collection errors (missing `train_sign_to_text` / `evaluation_metrics` modules; `app/main.py` has a pre-existing `@app.middleware(...)` before `app = FastAPI(...)` NameError).

---

## 4. RATE LIMITS

Restored defaults in `backend/app/core/config.py`:
- `rate_limit_word`: `20/min` → **`30/min`**
- `rate_limit_compose`: `15/min` → **`20/min`**
- `rate_limit_history_write`: `20/min` → **`30/min`**
- `rate_limit_login` stays `5/min` but is no longer applied to `/auth/me`.

Removed the dedicated login limiter from `/auth/me` (`backend/app/api/routes/auth.py`): dropped `login_rate_limit_dependency` import and its `Depends(...)`, and removed the now-unused `login_rate_limit_dependency()` from `backend/app/core/ratelimit.py`.

**Frontend callers of `/auth/me` (`endpoints.me`):**
1. `frontend/src/app/shell.tsx:91` — `apiClient.me()` once per app load (AppShell mount).
2. `frontend/src/features/settings/SettingsPage.tsx:54` — `apiClient.me()` once when Settings opens (lazily, only if a token exists).

So `/auth/me` is called at most once per session (shell) plus once when Settings is opened — far below any login throttle. Note: **the backend cannot see login attempts at all** — sign-in/sign-up go to Supabase directly (`frontend/src/services/supabase/auth.ts`: `signInWithPassword`, `signUp`), the backend only verifies tokens on `/auth/me`.

---

## 5. HEADERS

- Security headers are set **only in the backend** (`backend/app/main.py`, `security_headers_middleware`). No `vercel.json` exists; `frontend/index.html` has no security-headers meta tags; `vite.config.ts` sets none. ✓ headers only in backend.
- Removed `camera=(), microphone=()` from the backend `Permissions-Policy` (now `"geolocation=()"`). Since the backend serves the frontend, the old header would have blocked camera/mic on the frontend; now it doesn't.
- Camera in Word mode locally: code-read confirms `TranslatePage` calls `navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 } } })` and the Word-mode section records a 2.5 s landmark window when `capabilities.word_recognition_enabled`. **Could not test**: no camera/headless display available here, and the backend model isn't promoted locally, so an end-to-end camera → word-recognition run wasn't performed.

---

## 6. PASSWORD

**Final rule:** min 8 characters + at least one uppercase + one lowercase + one digit + one special character. Enforced client-side in `frontend/src/features/auth/AuthPages.tsx` via `credentialsSchema` (signup) — and **was** also applied to login via the shared schema.

**Fix:** Login now uses a separate `loginSchema` (email + non-empty password only); signup keeps the full complexity rule. Existing accounts with older/simpler passwords can still log in.

**Plain statement:** these rules are **client-side only** unless a Supabase Auth password policy is configured in the Supabase dashboard. → added to MANUAL CHECKLIST below.

---

## 7. HELP PAGE

I attempted to reach `https://indiansignlanguage.org/`: the site is reachable (returns HTTP 403) but blocks the request (bot protection), so **I could not verify any of the 5 per-word dictionary URLs** (`/dictionary/hello` etc. all 403).

Per the fallback instruction: every word card now links to the portal homepage `https://indiansignlanguage.org/`; the per-word URL is kept as a clearly marked, unverified constant (`ISL_BASE_URL`) with a note in the page. Added a third-party disclaimer that indiansignlanguage.org belongs to a third party and this page only links to it.

Removed the "Train the model" / Kaggle section from the user-facing Help page.

---

## 8. NOTEBOOK

- Moved `research/train_isl_sign_to_text.ipynb` → `research/archive/train_isl_sign_to_text.ipynb` (plain `mv`, not `git mv`, no staging).
- Added `research/archive/README.md` with one line: `Not used: unvalidated, different feature pipeline, not the word_v3 model.`
- Not deleted.

Search for `SIGNSAGE_ALLOW_UNPROMOTED_DEV=true` as a "way forward" in docs/scripts — found in docs only (described as the local-dev escape hatch, never as a production way forward):
- `docs/architecture/DATA_AND_MODEL.md` (servable in dev via the flag; NOT promoted)
- `docs/datasets/TEMPORAL_ISL_DATASET_AUDIT.md` (pilot servable via the flag; NOT promoted)
- `docs/architecture/API_REFERENCE.md`, `docs/architecture/BACKEND_ARCHITECTURE.md` (local only, never Render)
- `backend/.env` (preserved, `=false`; I did not edit it)

I did not change any code for this.

---

## 9. LIVE INTERPRETER (code-read only, report only)

Page: `frontend/src/features/translate/TranslatePage.tsx` ("Live Interpreter").

**Visitor sees when the backend is gated** (`SIGNSAGE_MODEL_PROMOTED=false` and no `SIGNSAGE_ALLOW_UNPROMOTED_DEV`):
- `/capabilities` reports `sign_to_text_available=false`, `sign_to_text_source="unavailable"`, reason `model_not_promoted`.
- An amber banner renders: **"Translation model unavailable"** + reason + "Camera mode is still available."
- The session badge reads **"Camera mode"** (never "Disconnected" when gated).
- The start button reads **"Start camera preview"** (not "Start interpreting").
- The live-translation panel shows "Translation unavailable … Camera mode is still available." / "Waiting for signing…".
- Camera + landmark overlay still work (start → preview with MediaPipe dots); no continuous translation/sentence output is produced.

**Where Word mode appears:** a separate **"Word recognition"** card below the live interpreter, shown only when `capabilities.word_recognition_enabled` (`SIGNSAGE_WORD_MODE_ENABLED=true`). It shows: "word-level recognition, adapted to one signer; not sentence translation", a Record word button (3-2-1 countdown → 2.5 s capture), recognised-words list with confidences, a "Make sentence" compose step, and the composed sentence. Labeled as isolated-word, not sentence translation.

---

## 10. FINAL

**Backend unit tests** (`python -m pytest tests`, backend `.venv`): **215 passed, 58 failed, 4 errors** — all pre-existing/environmental (verified identical with and without my changes; 4 collection errors are pre-existing). My changes: 0 new failures.

**Frontend tests** (`npm test`): **30 total, 29 passed, 1 failed**. The 1 failure is `sessionStatus.test.ts` expecting "Preview only" but getting "Camera mode" — in `sessionStatus.ts`/its test, which I did **not** touch (pre-existing).

**Frontend build** (`npm run build` = `tsc -b && vite build`): **passed** (typecheck + production build OK; `dist/` emitted).

```
$ git status --short
 M backend/app/api/routes/auth.py
 M backend/app/api/routes/capabilities.py
 M backend/app/core/config.py
 M backend/app/core/ratelimit.py
 M backend/app/main.py
 M frontend/src/components/footer.tsx
 M frontend/src/features/about/AboutPage.tsx
 M frontend/src/features/auth/AuthPages.tsx
 M frontend/src/features/help/HelpPage.tsx
 M frontend/src/features/landing/LandingPage.tsx
 M frontend/src/features/privacy/PrivacyPage.tsx
 D research/train_isl_sign_to_text.ipynb
(research/archive/ contents are gitignored via the archive/ rule, so they don't appear)
$ git add -n . | wc -l            -> 12
$ du -sh .                         -> 51G total (44G excluding node_modules/dist/.git)
$ git add -n . | grep -iE 'recordings|\.venv|\.npz|\.mp4|\.task|\.ckpt|\.zip|\.env'
 -> (empty: nothing sensitive would be staged)
```

### MANUAL CHECKLIST (for you)
- [ ] Confirm Supabase Auth password policy is (or isn't) configured in the Supabase dashboard — the client-side min-8 + 4-class rule is client-side only unless the dashboard policy is set.
- [ ] Verify the backend serves under `render.yaml` with `SIGNSAGE_MODEL_PROMOTED=false` (untouched) and `SIGNSAGE_ALLOW_UNPROMOTED_DEV` absent from Render.
- [ ] Manually check camera + Word mode in a real browser locally (couldn't test here — no camera/headless).
- [ ] Re-verify `indiansignlanguage.org` dictionary links by hand later (site returned 403 to automated fetches).
- [ ] Confirm no `.env` secrets committed (`.env` is gitignored; I did not touch it).
- [ ] Review that the 58 backend test failures / 4 errors / 1 frontend test failure are acceptable as pre-existing before presenting.

### NOT TESTED
- End-to-end camera → word recognition in a live browser (no camera/headless environment).
- Backend under a real uvicorn/Render deploy with `SIGNSAGE_WORD_MODE_ENABLED=true`.
- `indiansignlanguage.org` per-word dictionary URLs (site blocks automated requests with 403).
- Supabase Auth dashboard password policy (client-side rules only).
- Mobile APK / `mobile/` changes (none made).
- `results-v4.zip` / dataset archives (untouched, not opened).
