# SignSage Documentation Audit — Submitted `docs/` Archive

**Review date:** 2026-11-02  
**Scope:** `docs.zip` supplied for review; 38 Markdown files and three reference assets.  
**Out of scope:** root README, source-code comments, backend/frontend READMEs, research docs outside `docs/`, and actual repository-wide link resolution.

## 1. Executive findings

The documentation contains useful technical and historical evidence, but the current navigation and summary layer needed correction.

Main problems found:

- No central `docs/README.md` index.
- `MASTER.md` still points readers toward the older documentation layout.
- `STATUS_AND_ROADMAP.md` starts with a long duplicated audit paragraph before the actual title.
- `DOCUMENTATION_RESTRUCTURE_REPORT.md` contained repeated filler text and claims about destination files not present in the supplied archive.
- `DEVELOPMENT_SETUP.md`, `PRIVACY_AND_LIMITATIONS.md`, and `DATASET_CATALOG.md` were dominated by repeated cross-reference/verification prose rather than usable instructions.
- Model status needed a clearer distinction between the next investigation candidate and an adopted/production model.
- The cleanup report records a retained ISLiM checkpoint A because existing endpoints depend on it, despite the decision to reject ISLiM for production continuous translation.

## 2. Current documentation inventory

The supplied archive contains the following Markdown files:

### Root
- `AGENTS.md`
- `MASTER.md`
- `STATUS_AND_ROADMAP.md`

### Architecture
- `architecture/API_REFERENCE.md`
- `architecture/BACKEND_ARCHITECTURE.md`
- `architecture/contract-verification.md`
- `architecture/DATA_AND_MODEL.md`
- `architecture/end-to-end-contract.md`
- `architecture/ML_ARCHITECTURE.md`
- `architecture/SYSTEM_ARCHITECTURE.md`

### Datasets
- `datasets/TEMPORAL_ISL_DATASET_AUDIT.md`
- `datasets/VOCABULARY_ALIGNMENT_REVIEW.md`

### Development
- `development/DEVELOPMENT.md`
- `development/DEVELOPMENT_SETUP.md`
- `development/KAGGLE_SMALL6_RUNBOOK.md`
- `development/RENDER_MODEL_DEPLOYMENT.md`

### Evaluation
- `evaluation/2026-10-30-next-evaluation-gate-proposal.md`
- `evaluation/EXTERNAL_EVALUATION.md`
- `evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md`
- `evaluation/LIVE_INTERPRETER_PERFORMANCE.md`

### ML
- `ml/DATASET_CATALOG.md`
- `ml/MODEL_STATUS.md`

### Product
- `product/FEATURES_AND_REQUIREMENTS.md`
- `product/PRODUCT_OVERVIEW.md`
- `product/PRIVACY_AND_LIMITATIONS.md`

### Reports
- `reports/ALTERNATIVE_ISL_MODEL_AUDIT.md`
- `reports/AUTH_DB_INTEGRATION_REPORT.md`
- `reports/CORE_FUNCTIONALITY_IMPLEMENTATION_REPORT.md`
- `reports/DOCUMENTATION_AUDIT.md`
- `reports/DOCUMENTATION_RESTRUCTURE_REPORT.md`
- `reports/FINAL_DEMONSTRATION_READINESS_REPORT.md`
- `reports/MODEL_ASSET_CLEANUP_REPORT.md`
- `reports/MODEL_DEPLOYMENT_DIAGNOSIS.md`
- `reports/PRODUCT_READINESS_AUDIT.md`
- `reports/PROTOTYPE_COMPLETION_REPORT.md`
- `reports/REMAINING_FUNCTIONALITY_AUDIT.md`
- `reports/RUNTIME_ARTIFACT_STORAGE_AUDIT.md`
- `reports/SIGN_RECOGNITION_EXPANSION_PLAN.md`

### Other
- `DEMONSTRATION_GUIDE.md` (added in this reviewed copy; now located under `product/`)
- `superpowers/specs/` contains no Markdown file in the supplied archive.

## 3. Classification and action

| Area | Review decision |
|---|---|
| `MASTER.md` | Keep; update its navigation table against the current tree |
| `STATUS_AND_ROADMAP.md` | Keep; remove duplicated pre-title audit paragraph and clearly date current status |
| `AGENTS.md` | Keep; source-code and full-repository references need a repository-wide check |
| Architecture contracts | Keep; detailed and potentially important; verify against code before implementation changes |
| Dataset audits | Keep; preserve their detailed evidence and limitations |
| Development guides | Keep; use `DEVELOPMENT_SETUP.md` for quick setup and `DEVELOPMENT.md` for broader workflow |
| ML summaries | Keep; make `MODEL_STATUS.md` the concise model decision index |
| Product summaries | Keep; remove repeated audit-verification prose |
| Historical reports | Keep as dated evidence; do not treat every report as current instructions |
| Superseded experiment runbooks | Retain with a visible superseded label until all unique commands/evidence are safely captured elsewhere |
| Reference PDF/images | Keep; they are source/reference assets, not documentation clutter |

## 4. Current model facts that documentation must preserve

- `SIGNSAGE_MODEL_PROMOTED=false`.
- No continuous ISL-to-English model is approved for production.
- Sid-V5 is a candidate for further checkpoint-access and architecture verification only.
- Sid-V5's reported 30.77% WER concerns continuous gloss recognition, not English translation quality.
- Sanket is isolated-sign/category-oriented and does not satisfy the continuous sentence task.
- ISH-NEWS is task-aligned research, but a pretrained checkpoint was not verified.
- The cleanup report states that the Sanket checkpoint, ISLiM checkpoint B, and two ISLiM mobile exports were deleted.
- The cleanup report states that ISLiM checkpoint A remains because current ISLiM endpoints reference it. This is a code dependency, not model approval.
- The cleanup report identifies a stale `ALT_CKPT` reference to deleted ISLiM checkpoint B; source-code cleanup remains outstanding.

## 5. Limitations of this audit

This is an audit of the supplied `docs/` archive, not a fresh scan of the full Git repository. It cannot establish that external links resolve, that the source code matches the docs, or that the cleanup report's filesystem claims are true. Those require access to the actual repository and executable environment.
