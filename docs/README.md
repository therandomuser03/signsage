# SignSage Documentation

| Document | Read it to |
|---|---|
| [PRD.md](PRD.md) | Understand what is being built, for whom, scope, requirements, roadmap |
| [ARCHITECTURE.md](ARCHITECTURE.md) | See how the system works and the exact contracts between parts |
| [MODELS_AND_DATA.md](MODELS_AND_DATA.md) | Datasets, models, evaluation results, promotion gates, training pipeline |
| [DESIGN_SYSTEM.md](DESIGN_SYSTEM.md) | Build UI that matches the product and its honesty rules |
| [CODE_STYLE.md](CODE_STYLE.md) | Follow code conventions |
| [TESTING.md](TESTING.md) | Know what to test and how |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deploy, configure and verify; ship a model bundle |
| [SIGN_ANIMATION.md](SIGN_ANIMATION.md) | Produce and validate avatar sign motion |
| [references.bib](references.bib) | Cite the dataset papers |

At the repository root: [README](../README.md) · [AGENTS](../AGENTS.md) · [SECURITY](../SECURITY.md) · [CONTRIBUTING](../CONTRIBUTING.md) · [CHANGELOG](../CHANGELOG.md) · [CREDITS](../CREDITS.md).

## How these documents are kept honest

1. **Code and tests are authoritative** for what the application does. Contract values are pinned in `ARCHITECTURE.md` §5.
2. **One topic, one file.** Edit in place; never add numbered copies or one-off reports. Dated history goes in `CHANGELOG.md`.
3. **Status words** mean something ([PRD §11](PRD.md#11-status-vocabulary)); never turn a metric into a quality claim.
4. Research metrics are task- and dataset-specific; do not compare isolated-sign accuracy, gloss WER and sentence BLEU.
5. When behaviour changes, update the code, tests, the affected document and the changelog together.

**Current model gate:** `SIGNSAGE_MODEL_PROMOTED=false`.
