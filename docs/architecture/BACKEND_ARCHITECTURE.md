# Backend Architecture

Source of truth: `docs/development/DEVELOPMENT.md`, `docs/architecture/end-to-end-contract.md`, `docs/reports/MODEL_DEPLOYMENT_DIAGNOSIS.md`, `backend/app/`.

## Layers
- API routes (`app/api/routes/`): auth (`auth.py`), history (`history.py`), translation (`translation.py` — `POST /translation/sign-to-text`, live backend, uncalled by frontend — B6). REST returns 503 when `promoted=false` (`MODEL_NOT_PROMOTED`).
- WebSocket (`app/api/websocket/translation.py`): unversioned `/ws/translate`. Auth message first (`auth` type with JWT). Landmark frames (`landmarks:543×[x,y,z]`). Interim `translation.text{final:false}` with repeat-agreement dedup. Boundary `sentence.boundary`. Final `final:true`. Errors: `MODEL_UNAVAILABLE` (gate correct), `INVALID_MESSAGE`, `SEQUENCE_TOO_LONG`.
- Services: `TranslationService` (`app/services/`), `video_service.py`, `avatar_gloss_service.py` (Groq→Ollama→deterministic, vocab-filtered to 14 IDs from `sign_vocabulary.json`, never invents IDs). `ImageClassifierAdapter` (alphabet CNN verified, 87.1% test acc). `IsolatedWordAdapter` (6-word pilot, 61.1% test accuracy, unpromoted). `SignToTextAdapter` (adapter works, no promoted model feeds it).
- ML adapters (`app/ml/adapters/`): `image_classifier_adapter.py`, `isolated_words.py`, `sign_to_text_adapter.py`, `i_sign_t5_adapter.py`. Registry caches once per process (`registry.py`).
- Preprocessing: `normalization.py` (`pose_normalize_v1`), `preprocessing/landmarks.py` (543→adapter input), `preprocessing/islim_landmarks.py` (543→94×4 conversion for ISLiM; face selection approximate — PARTIAL compatibility; ordering unverified without `.npy` source).
- Security: `core/security.py` (Supabase `GET /auth/v1/user` + TTL cache). `ApiClient` attaches `Authorization: Bearer <token>`. Guest-safe (`guest` header when no token). `ALLOW_UNPROMOTED_DEV` (local only, never Render — `render.yaml` omits it).

## Bundle / model state
- Configured dir: `SIGNSAGE_MODEL_BUNDLE_DIR=../research/artifacts/352868924/signsage_deployment_bundle`. Actual directory empty (no `checkpoint/`, `tokenizer/`, contracts). `SIGNSAGE_IMAGE_ALPHABET_BUNDLE_DIR` and `SIGNSAGE_IMAGE_WORD_BUNDLE_DIR` point to real verified artifacts (`alphabet/model.pt`, `labels.json`, `evaluation.json`).
- `backend/scripts/model_check.py`: verifies load + zero-rejection (`zeros-(10,543,3)` must be REJECTED). Passes against smoke-test bundle, fails against empty/missing bundle.
- `fetch_bundle.sh`: checksum-verified download from HF Hub (private repo `therandomuser03/signsage-deployment-bundle`). Uses `SIGNSAGE_HF_TOKEN` (not exposed in frontend, not in logs). Bundle ZIP must have `checkpoint/`, `tokenizer/`, `landmark_contract.json`, `sign_to_text_deployment.json`.

## Blockers (verified, ordered B1-B6 from STATUS_AND_ROADMAP.md)
- B1 (root): No promotable model. 352868924 broken (WER 1.0046, diversity 0.99%, collapse). Isolated-word pilot 61.1% (below 80/80/60 promotion bar, not signer-independent). No continuous model promoted.
- B2: Landmark order trusted, never verified at load (`end-to-end-contract.md` notes gap). No runtime order comparison.
- B3: Confidence gate ships OFF (`min_confidence=0.0` in adapter/default). Must calibrate on promotion.
- B4: No numeric norm-equivalence check against golden vector (training export vs backend).
- B5: Tokenizer incompatibility (SP-16k serving vs T5 32k). T5 model would need separate adapter/export (exists as research only — `t5_poc_gate` FAIL, `WIRING-FAIL`).
- B6: Frontend REST translation path (`POST /api/v1/translation/sign-to-text`) uncalled by frontend; only WS path active.
