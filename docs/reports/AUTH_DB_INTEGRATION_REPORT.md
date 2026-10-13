# SignSage — Auth & Database Integration Report

Date: 2026-10-12. Authority is code + executed checks; docs were read first
(`docs/AGENTS.md`, `README.md`, `PRODUCT_READINESS_AUDIT.md`, `docs/` consolidated set).
No secrets are recorded here: env values were never printed, only key names and
reachability classes. No training, no model/promotion changes, no commits.

## 1. Architecture and request-flow summary

Supabase owns identity; FastAPI never issues tokens; Postgres (via PostgREST) owns
persistent data. Guest mode is a first-class design state, not a fallback.

```text
Signup (consent checkboxes, client-side gate)
  → supabase.auth.signUp email+password (+ user_metadata consent stamps, new)
  → Supabase Auth owns the account; session persisted by supabase-js (localStorage)
Sign-in (password / Google OAuth / reset-mail via Supabase)
  → StartPage routes on getAccessToken() presence (/app vs /login, safeNext-guarded)
API call: ApiClient attaches Authorization: Bearer <access_token> (guest: none)
  → get_optional_user → authenticate_header → verify_access_token
      → GET <supabase>/auth/v1/user (5 s timeout, TTL-cached, failures → 503)
  → history endpoints require get_current_user (401 otherwise)
  → SupabaseRestClient forwards the USER's JWT to PostgREST (never a service key)
  → RLS policies on translation_sessions enforce per-user isolation
WS /ws/translate: token arrives in first {"type":"auth"} message (browsers cannot
set WS headers); same verify_access_token path.
Sign-out: supabase signOut → shell clears account UI (lists now reset too, new).
```

Saved Signs has no separate backend: `SavedSignsPage` is a view over `GET /history`
(search + delete-by-id). History rows are immutable (no UPDATE policy, no endpoint).

## 2. Files inspected and changed

Inspected (auth): `frontend/src/services/supabase/{client.ts,auth.ts}`,
`frontend/src/features/auth/AuthPages.tsx`, `frontend/src/app/{routes.tsx,shell.tsx}`,
`frontend/src/services/api/{client.ts,endpoints.ts,errors.ts}`,
`frontend/src/features/{history/HistoryPage.tsx,saved/SavedSignsPage.tsx,
settings/SettingsPage.tsx}`, `backend/app/core/{security.py,dependencies.py,errors.py}`,
`backend/app/services/{authentication_service.py,history_service.py}`,
`backend/app/api/routes/{auth.py,history.py}`,
`backend/app/infrastructure/database/{supabase.py,repositories/translation_repository.py,
migrations/001_translation_sessions.sql}`,
`backend/app/{schemas/history.py,domain/translation/entities.py}`,
existing tests `test_security.py`, `test_supabase_repository.py`,
`test_history_service.py`, `test_api_auth_history.py`.

Changed (4 source/test files, all in scope):
- `frontend/src/features/history/HistoryPage.tsx` — auth-change subscription.
- `frontend/src/features/saved/SavedSignsPage.tsx` — auth-change subscription.
- `frontend/src/services/supabase/auth.ts` — consent timestamps on signup.
- `backend/tests/unit/infrastructure/test_supabase_repository.py` — delete-JWT assertion.

## 3. Issues, root causes, fixes

1. **Sign-out left the previous account's history/saved signs on screen.**
   Root cause: `HistoryPage`/`SavedSignsPage` load once on mount and never subscribe to
   auth changes (only the shell `UserArea` did), so a shared-device sign-out kept rendering
   another user's entries until navigation. In-memory only, but exactly the state the scope
   requires cleared. Fix: both pages now subscribe via existing `onAuthChange` — on sign-out
   they reset to the identical signed-out (401) state (items cleared, sign-in prompt); on
   sign-in they reload. No new behavior invented; mirrors the existing 401 path.
2. **Consent acceptance was required but never recorded.**
   Root cause: `signUp()` sent only email+password; the Terms/data checkboxes lived and died
   in component state, with no consent table anywhere. Fix (within existing schema, no
   migration): signup now stamps `terms_accepted_at` + `data_consent_accepted_at` (ISO time
   of acceptance) into Supabase Auth `user_metadata` via the documented
   `signUp({…, options: { data } })` shape. Observable on the Auth user record.
3. **Delete-by-id test did not pin the authorization header.**
   The cross-user-ID-manipulation defense rests on the caller's JWT riding every PostgREST
   request so RLS can reject foreign rows. `add` asserted this; `delete` did not. Fix (test
   only): assert `Authorization: Bearer <user JWT>` on the delete request.

Not changed by design: no route-guard redirects (guest mode is intended; protected pages
render sign-in prompts on 401), no saved-sign update API (immutability is intended), no new
consent table (would invent schema), no ORM/auth-provider changes.

## 4. Test commands and exact results

- `cd backend && ./.venv/bin/python -m pytest tests/unit/core/test_security.py tests/unit/infrastructure/test_supabase_repository.py tests/unit/services/test_history_service.py tests/integration/test_api_auth_history.py -q` → **35 passed** (mocked Supabase HTTP via `MockTransport`, fake auth boundary, memory backend).
- `cd frontend && npm test` → **15 pass / 0 fail** (unchanged suites still green).
- `cd frontend && npm run build` (tsc -b + vite) → **PASS** (~8.5 s; covers changed `.ts`/`.tsx`).
- Safe live probes (no credentials sent except the public publishable key as `apikey`, nothing created/modified/deleted): backend+frontend configured against the **same** Supabase project host; `/auth/v1/user` → 401/403 correctly without/with-malformed token; bogus sign-in → 400 `invalid_credentials`; anonymous PostgREST read → **200 `[]`** (RLS yields zero rows, leaks nothing).

## 5. Verification table

| Check | Verdict | Evidence |
|---|---|---|
| Protected endpoints reject unauthenticated requests | **PASS** (live local) | 401 `UNAUTHORIZED` on `/auth/me`, `/history` without token |
| Invalid/expired/malformed/missing tokens fail safely | **PASS** (mocked + live-negative) | 401→AuthError, 403→AuthError, outage→503 (tests); live 400/403/401 on auth endpoints |
| Token transmission (Bearer on REST, first-frame on WS) | **PASS** (code + mocked request-shape tests) | ApiClient header injection; add/delete assert user JWT, never service key |
| Per-user isolation + RLS expectations | **PASS** (mocked) / **BLOCKED** (live) | Memory-repo isolation + clear-scoped-to-own-id + delete-JWT tests; RLS SQL reviewed but migration-application and live two-user check need a real account |
| History CRUD round-trip | **PASS** (mocked) / **BLOCKED** (live) | Round-trip + pagination + 404-on-missing tests; no live rows written (see §7) |
| Saved-signs list/search/delete | **PASS** (mocked, shares history path) | Same repository + endpoints; no separate backend surface exists by design |
| Sign-out clears protected UI state | **PASS** (code + build; fixed this pass) | New auth-change reset mirrors 401 state on both pages |
| Consent acceptance persisted | **PASS** (code; fixed this pass) | Timestamps in Auth user_metadata; observable, no schema change |
| Signup/login/session-restore end-to-end with a real account | **BLOCKED** | Disposable-address signup rejected (`email_address_invalid`); needs operator test account |
| Guest-mode translation/gloss without auth | **PASS** (prior live pass) | Guest translation + deterministic gloss verified against live backend |

## 6. Verdicts

- **Authentication: CONDITIONAL PASS.** Token issuance (Supabase), transmission, verification, caching, failure taxonomy, guest handling, and sign-out UI reset are verified by tests + live negative probes. Real account signup/login/refresh remains BLOCKED (§7).
- **Database persistence: CONDITIONAL PASS.** Request shapes, RLS policy design, pagination/total parsing, immutability, and outage behavior are verified by tests; anonymous access provably returns nothing live. Actual row round-trips against the real project are BLOCKED (§7), as is confirmation the migration is applied there.
- **Authorization/ownership: CONDITIONAL PASS.** Every layer binds the verified user id (explicit `user_id` on insert, caller JWT on all requests, RLS own-rows policies, no UPDATE surface). Live two-account proof is BLOCKED with the rest of live verification.
- **Consent logging: PASS (with caveat).** Acceptance is now stamped on the Auth user at signup. Pre-existing accounts predate the stamp (expected); no backfill invented.

## 7. Prerequisites for real end-to-end verification

1. An operator-created Supabase test account (or a project-owned test mailbox), since disposable domains are rejected and residual-account cleanup needs a decision.
2. Confirmation that `migrations/001_translation_sessions.sql` is applied on that project (or a fresh dev project), plus permission to write/delete disposable `translation_sessions` rows.
3. Then: sign in → `GET /auth/me` → gloss-with-save → `GET /history` contains row → `DELETE` row → second account proves cross-user 404 → sign out → UI shows guest state. Delete all test rows after.

## 8. Next steps (verified blockers only)

1. Operator supplies test account + confirms migration state (§7); run the script above; record results here.
2. If live RLS or round-trip fails, fix forward from the failing layer (client headers → repository → policy) with the existing mocked suites as the safety net.
3. Consider a migration idempotency check (`create policy if not exists` has no native form; currently re-running the file errors on duplicate policies) only if re-application is ever needed — not now.
