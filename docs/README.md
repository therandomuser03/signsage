# SignSage Documentation

This folder contains the current project guides, technical contracts, research evidence, and historical audit reports.

## Start here

| If you need to... | Read |
|---|---|
| Understand the project and its scope | [MASTER.md](MASTER.md) |
| See what is currently working and blocked | [STATUS_AND_ROADMAP.md](STATUS_AND_ROADMAP.md) |
| Understand model decisions and promotion status | [ml/MODEL_STATUS.md](ml/MODEL_STATUS.md) |
| Set up the development environment | [development/DEVELOPMENT_SETUP.md](development/DEVELOPMENT_SETUP.md) |
| Understand the system architecture | [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) |
| Inspect backend interfaces and contracts | [architecture/BACKEND_ARCHITECTURE.md](architecture/BACKEND_ARCHITECTURE.md), [architecture/API_REFERENCE.md](architecture/API_REFERENCE.md), [architecture/end-to-end-contract.md](architecture/end-to-end-contract.md) |
| Understand datasets and their limitations | [ml/DATASET_CATALOG.md](ml/DATASET_CATALOG.md), [datasets/](datasets/) |
| Prepare a truthful live demonstration | [product/DEMONSTRATION_GUIDE.md](product/DEMONSTRATION_GUIDE.md) |
| Review a past investigation | [reports/](reports/) |

## Directory guide

- `architecture/` — system design, backend/ML architecture, API and data contracts.
- `datasets/` — detailed dataset and vocabulary audits.
- `development/` — setup, development workflow, deployment, and experiment runbooks.
- `evaluation/` — evaluation results and proposals.
- `ml/` — current model and dataset status summaries.
- `product/` — product scope, feature evidence, privacy, limitations, and demo guidance.
- `reference/` — external ISL learning/reference material.
- `reports/` — dated audit reports and historical evidence.

## Authority and freshness

1. Runtime code and executable configuration are authoritative for what the application actually does.
2. `STATUS_AND_ROADMAP.md` and `ml/MODEL_STATUS.md` are the current project/model status summaries.
3. Architecture contracts document expected interfaces; verify them against code before changing implementation.
4. Files under `reports/` are evidence snapshots. A newer report may supersede an older one; historical reports must not be treated as current instructions.
5. Research metrics are task- and dataset-specific. Do not compare isolated-sign BLEU-1, continuous gloss WER, and sentence-translation BLEU as if they measured the same task.

**Current model gate:** `SIGNSAGE_MODEL_PROMOTED=false`. No continuous ISL-to-English model is approved for production.
