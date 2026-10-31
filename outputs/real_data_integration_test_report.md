# Real-Data Integration Test Report (Level C)

Test executed: `/tmp/test_integration_level_c.py`
Environment: CPU-only session (no GPU); dataset at `kaggle_input_v4/`

Results:
- Actual `.pose` files parsed from zip archives (lazy load verified).
- Pose contract verified: `(frames, 1, 576, 3)` with first 543 core landmarks selected.
- Normalization pipeline produces `(160, 1629)` input tensors.
- SentencePiece tokenizer loaded (vocab 16000, special IDs 0/1/2/3).
- Token array shapes verified: train (2500, 43), val (2500, 82), test (2500, 40).
- Translation dataset (`TranslationDataset`) constructs batches correctly.
- Real batch dimensions: poses (2, 160, 1629), decoder_input (2, 19), targets (2, 19).
- Source padding mask and causal mask constructed correctly.
- Model B (`SignTransformer`) completes real-data forward pass.
- Backward pass completes; gradient norm finite (8.27 before clip).
- Optimizer step completes successfully.
- Loss value finite (10.05 for untrained model on pilot subset).

Status: PASS (Level 3 — Real-data model integration passed)
