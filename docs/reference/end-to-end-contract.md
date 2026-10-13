# SignSage End-to-End Contract (authoritative pin, reference)

> Value-level contract pin for the live pipeline. Summarized in
> `docs/DEVELOPMENT.md` (Interfaces) and `docs/STATUS_AND_ROADMAP.md` (audit);
> this file is the exact-values authority.

Any change that alters a value below must update the contract files, the code constants, and the tests together. Counts alone are not the contract — order and versions are.

## 1. Landmark representation (single source of truth: `landmark_contract.json` from the training export header)
- Total 543 landmarks × 3 coords = 1629 floats per frame. `TOTAL_LANDMARKS=543`, `COORDINATES_PER_LANDMARK=3`, `RAW_FEATURE_DIM=1629` (`backend/app/ml/preprocessing/landmarks.py:12-21`, `frontend/src/features/translate/landmarks.ts:14-20`).
- Component order (values, not just counts): POSE 33 (indices 0–32, hips = pose 23/24) → FACE 468 (33–500) → LEFT_HAND 21 (501–521) → RIGHT_HAND 21 (522–542). Frontend `landmarks.ts:104-124` emits this order; `normalization.py:35-58` consumes it. **Gap: no runtime check compares these values** (blocker B2).
- Coordinates: frontend sends raw MediaPipe normalized x/y (+z, `z??0`), missing landmarks → `[0,0,0]` (`landmarks.ts:34-47`). Never pre-normalize or pre-delta on the frontend.
- Normalization: `pose_normalize_v1` — hip-midpoint center (fallback body-mean), scale = median per-frame body-33 std, `nan_to_num→0` (`normalization.py:21-61`). Must be numerically identical to training export (blocker B4).

## 2. Temporal convention
- Frontend: stateless, 10 fps (`SEND_INTERVAL_MS=100`), sends only `frame.detected` frames (`TranslatePage.tsx:36,151-164`).
- Backend: `max_frames_per_sequence=160`, live window 80 / stride 20, hard cap 640 then `SequenceTooLongError→reset` (`config.py:61,70-71`, `session_service.py:18-55`, `websocket/translation.py:89-103`).
- `prepare_live_landmarks((T,543,3))` → `((160,1629), true_len=min(T,160))`; `resize_sequence` linspace-downsamples or zero-pads; `true_len` feeds the padding mask (`sequence.py:12-44`). Minimum 2 frames or `ValueError`.

## 3. Model input
- Adapter input `(T,543,3)`; model input per frame 1629 raw, internally catted with frame-delta to 3258 (`MODEL_INPUT_DIM=3258`, `FrameProjection`, `models/sign_to_text.py:17-31`). `model_contract` pins `feature_dim=1629`, `model_input_dim=3258`, `normalization_version`, 4 distinct special ids, decoding block.

## 4. Tokenizer / decoding (serving track)
- SentencePiece unigram 16k, `PAD0/UNK1/BOS2/EOS3`; adapter load requires exact id match (`sign_to_text_adapter.py:126-135`). Greedy decode bans pad/bos/unk, enforces min-length EOS, no-repeat-ngram, repetition penalty (`inference/sign_to_text.py:44-68`). T5 32k vocab is a different universe — never interchangeable.

## 5. Transport messages
- WS `/ws/translate` (unversioned): `auth` → `landmarks.frame{landmarks:543×[x,y,z] finite}` → interim `translation.text{final:false}` (deduped by repeat-agreement) → `sentence.boundary{save_history?}` → `translation.text{final:true}` or `error{MODEL_UNAVAILABLE|INVALID_MESSAGE|…}` + close (`schemas/websocket.py`, `api/websocket/translation.py`, `frontend …/websocket/*`, `WS_PATH` in `endpoints.ts:15`). Max message bytes enforced; no send-queue when socket not open.
- REST: `POST /api/v1/translation/sign-to-text` ↔ `SignToText{Request,Response}` (`s schemas/translation.py`, `types/api.ts:68-87`); backend live, frontend uncalled (blocker B6).
- Status: `GET /api/v1/health`, `/api/v1/capabilities` → `sign_to_text_available/unavailable_reason`, `text_to_sign_mode="avatar_gloss"`.

## 6. Verdict vocabulary (only honest outputs)
`accepted:true + text` (MODEL/AVATAR_GLOSS per policies) or `accepted:false + reason ∈ {insufficient_motion, low_model_confidence, empty_output}` (+`MODEL_UNAVAILABLE` at transport when unpromoted). Empty text is never presentable (`policies.py:7-22`). Frontend renders `accepted ? "accepted" : reason · result_type · confidence%`.
