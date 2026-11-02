# Features and Requirements
Consolidated: STATUS_AND_ROADMAP.md §1-2, DATA_AND_MODEL.md §4, EXTERNAL_EVALUATION.md, ISOLATED_ISL_RECOVERY_AUDIT.md §4, SIGN_RECOGNITION_EXPANSION_PLAN.md §5-7, PROTOTYPE_COMPLETION_REPORT.md, LIVE_INTERPRETER_PERFORMANCE.md.

## Requirements (verified, not aspirational)
1. SIGNSAGE_MODEL_PROMOTED=false (MASTER.md, STATUS_AND_ROADMAP.md, DATA_AND_MODEL.md §5, MODEL_DEPLOYMENT_DIAGNOSIS.md).
2. No raw camera upload; only 543-point landmarks (AGENTS.md rule 8, end-to-end-contract.md).
3. Vocabulary evidence-bound: only 6-word pilot (friend/hello/hospital/market/school/thank-you) recognized; 14 avatar gloss IDs (HELLO/THANK_YOU/etc.) for generation; no broader vocabulary claim (MASTER.md §4; DATA_AND_MODEL.md §4; SIGN_RECOGNITION_EXPANSION_PLAN.md §Vocabulary; DEVELOPMENT.md §2).
4. No synthetic predictions presented as real (PROTOTYPE_COMPLETION_REPORT.md; FINAL_DEMONSTRATION_READINESS_REPORT.md; MODEL_DEPLOYMENT_DIAGNOSIS.md).
5. No dataset/archive deletion; archive untouched (FINAL_INTEGRITY_AUDIT_REPORT.md; DOCUMENTATION_AUDIT.md).
6. No promotion without quality/diversity evidence + adapter verification + license review (MASTER.md §6; DATA_AND_MODEL.md §5; SIGN_RECOGNITION_EXPANSION_PLAN.md §7).
7. Auth isolation via Supabase JWT + RLS (AUTH_DB_INTEGRATION_REPORT.md — conditional pass for live multi-account, mocked integration passes); history isolated; consent stamps at signup; sign-out clears protected UI state.
8. Security: .env excluded by .gitignore; never commit HF_TOKEN/GROQ keys; no server secrets in frontend; no raw upload path; WS auth first frame; REST Bearer header.

## Feature evidence (verified / unverified / blocked)
- Image classification (alphabet): VERIFIED (Playwright E2E 7 pass, real dataset fixtures, endpoint structured response with real CNN label/confidence/model_version).
- Image classification (word): UNPROMOTED (19.0% test accuracy, artifacts present but never deployed).
- Isolated-word recognition: EXPERIMENTAL (61.1% accuracy, below promotion bar, uncalibrated confidence, not signer-independent, vocabulary 6 words only).
- Continuous sentence translation: BLOCKED (no promoted model; adapter contracts preserved; 352868924 broken; ISLiM adapter PARTIAL; no paired keypoints.pt locally; paired dataset BLOCKED).
- Text-to-sign: VERIFIED DETERMINISTIC (14 gloss IDs, vocabulary verified, unknown words dropped honestly, missing .glb handled with empty state, avatarReady state verified).
- Camera pipeline (landmarks): VERIFIED BY CODE + E2E SERVICE CONTRACT (not full continuous session with real camera; Playwright image-mode E2E passes; camera lifecycle code-correct; continuous session requires promoted model + real paired dataset).
- Health/capabilities/models: VERIFIED LIVE (honest refusal; 503 MODEL_NOT_PROMOTED; MODEL_UNAVAILABLE WS close; capabilities reports source/unavailable_reason; model_bundle_found false; model_loaded false).
- Deployment (Render/Docker): VERIFIED (render.yaml, Dockerfile, fetch_bundle.sh checksum-verified; .env never baked into image; promotion preserved; bundle download non-destructive).
- Browser end-to-end (continuous): NOT FULLY EXECUTED (service contracts verified; Playwright requires sudo for dependencies; full session deferred; no synthetic predictions as real).
