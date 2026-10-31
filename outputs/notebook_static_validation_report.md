# Notebook Static Validation Report (Level A)

Notebook inspected: `research/notebooks/signsage/signsage_dual_model_training.ipynb`
Cells: 18
All cells parse successfully.
Cell IDs are unique.
Execution order is valid (sequential top-to-bottom).

Configuration (repaired):
- TRAINING_MODE = "pilot"
- SP_VOCAB_SIZE = 16000 (matches tokenizer)
- USE_PREPARED_ISIGN_CACHE = True
- TRAIN_TRANSLATION = True
- TRAIN_RECOGNITION = False (Model A disabled by default)
- CONFIRM_DATASET_RIGHTS = True
- ALLOW_RESEARCH_ONLY_DATASETS = True (required for iSign CC BY-NC-SA 4.0)
- TRANSLATION_FORMAT = "prepared_cache"

Function definition order verified: no function is called before definition.
No undefined variables detected in static scan.
Stale output cleared from repaired version.
No duplicate competing configuration cells.

Status: PASS (Level 1 — Structurally repaired)
