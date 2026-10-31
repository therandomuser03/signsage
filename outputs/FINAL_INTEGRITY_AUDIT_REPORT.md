# SignSage — Post-Pilot Integrity Audit and Checkpoint Verification

**Status: PILOT RESULTS ARE NOT TRUSTWORTHY.** 
The pilot run experienced severe model collapse onto repetitive tokens (`"diksha necklace four diksha"`), the reported metrics were calculated on an unrepresentative sample of N=8 test records, and the reported BLEU score was a mathematical artifact of custom Laplace +1 smoothing on zero counts. The real Model B 16k checkpoint has been recovered, strictly reloaded, verified with finite forward execution, and stored persistently.

Model promotion status: `SIGNSAGE_MODEL_PROMOTED=false` (strictly maintained).

---

## 1. Annotation Diversity Report

### Verification of Reported Figures
The pilot report stated:
- *7,500 aligned pose/annotation pairs*
- *Only 24 unique English targets*
- *Prediction collapse detected*

Direct verification from the actual dataset files (`kaggle_input_v4/isign/iSign_v1.1.csv` and `kaggle_input_v4/s2t_small6/pose_cache/manifest.json`):

| Scope | Total Records | Unique UIDs | Unique Normalized Annotations | Singletons (Freq = 1) | Singletons % | Unique Source Videos | Multi-Video Annotations |
|---|---|---|---|---|---|---|---|
| **Full Raw CSV** | 127,237 | 127,237 | 122,600 | 120,364 | 98.18% | 6,058 | 1,795 |
| **Cleaned Usable CSV** | 123,779 | 123,779 | 120,221 | 118,050 | 98.20% | 3,797 | 1,729 |
| **7,500 Manifest Subset** | 7,500 | 7,500 | 7,447 | 7,409 | 99.49% | 2,208 | 31 |
| **2,500 Train Split** | 2,500 | 2,500 | 2,482 | 2,469 | 99.48% | 680 | 9 |

### Annotation Frequency Distribution (Full Raw CSV)
- **Min**: 1
- **25th Percentile**: 1.0
- **Median**: 1.0
- **Mean**: 1.04
- **75th Percentile**: 1.0
- **95th Percentile**: 1.0
- **Max Frequency**: 391 (empty strings, cleaned) / 66 ("let me tell you about it.")

### Top 20 Most Frequent Annotations in Full Raw Dataset

| Rank | Frequency | Normalized Annotation | Notes |
|---|---|---|---|
| 1 | 391 | `""` (empty string) | Cleaned as junk |
| 2 | 99 | `"2"` | Cleaned as numeric junk |
| 3 | 96 | `"1"` | Cleaned as numeric junk |
| 4 | 87 | `"3"` | Cleaned as numeric junk |
| 5 | 66 | `"let me tell you about it."` | Natural conversational phrase |
| 6 | 54 | `"5"` | Cleaned as numeric junk |
| 7 | 52 | `"blank"` | Educational fill-in phrase |
| 8 | 49 | `"4"` | Cleaned as numeric junk |
| 9 | 45 | `"dash."` | Educational punctuation phrase |
| 10 | 38 | `"have a look at the video."` | Instruction prompt |
| 11 | 36 | `"b"` | Cleaned as single-letter junk |
| 12 | 33 | `"dash"` | Educational phrase |
| 13 | 32 | `"6"` | Cleaned as numeric junk |
| 14 | 32 | `"for the teacher."` | Instructional label |
| 15 | 31 | `"blank."` | Educational phrase |
| 16 | 30 | `"[music]"` | Cleaned as bracketed junk |
| 17 | 29 | `"reading is fun."` | Educational textbook phrase |
| 18 | 29 | `"7"` | Cleaned as numeric junk |
| 19 | 29 | `"let me tell you what happened."` | Conversational phrase |
| 20 | 27 | `"new words"` | Educational section header |

### Cross-Video Repetition
In the full dataset, **1,795 annotations** appear across multiple distinct source videos (e.g. `"let me tell you about it."` appears across 65 videos). In the 2,500-sample training split, only 9 annotations appear in >1 video. This is expected linguistic recurrence in standard educational video corpora.

### Root Cause of the "24 Unique English Targets" Claim
**The 24-target figure is a reporting conflation, NOT a dataset property.**
- The full dataset contains **120,221** unique cleaned annotations.
- The 7,500 manifest subset contains **7,447** unique annotations.
- The 2,500 training split contains **2,482** unique annotations.
- In `signsage_dual_model_training.ipynb`, `prepare_translation_data()` computes:
  `unique_targets = len({row["text"] for name in loaded for row in loaded[name]})`
- During the pilot run, `MAX_SAMPLES_PER_SPLIT = 8` was used, loading 8 train, 8 validation, and 8 test samples (**24 samples total**).
- Because all 24 samples had distinct text annotations, the preflight printed `Unique English target sentences: 24`.
- The pilot report erroneously cited this 24-sample evaluation slice statistic alongside the 7,500-sample dataset size.

---

## 2. Audit of the Evaluation Metrics

### Investigating the Reported Results
Reported pilot metrics:
- `WER = 1.0`
- `BLEU-4 = 0.946`
- `ROUGE-L = 0.0`
- `Prediction diversity = 0.25`
- `Collapsed = True`

### Mathematical Explanation of Metric Discrepancies
Previous audits concluded that `BLEU-4 = 0.946` was "mathematically impossible" alongside `WER = 1.0` and `ROUGE-L = 0.0`. That conclusion was based on the mistaken assumption that BLEU was reported on a 0–1 scale (representing 94.6%).

Inspection of Cell 13 in `signsage_dual_model_training.ipynb` reveals the actual implementation:
```python
precisions = [(clipped[i] + 1) / (totals[i] + 1) for i in range(4)]
hyp_words = sum(len(h) for h in hyps)
brevity = min(1.0, math.exp(1 - ref_words / max(hyp_words, 1)))
bleu = 100.0 * brevity * math.exp(sum(math.log(p) for p in precisions) / 4)
```

1. **Laplace +1 Smoothing on Zero Overlap**:
   When predictions have **zero** matching n-grams with references (`clipped[i] = 0` for n=1,2,3,4), the precision vector does NOT become 0. Instead, it becomes `[1/(32+1), 1/(24+1), 1/(16+1), 1/(8+1)]`.
2. **Brevity Penalty**:
   The 8 reference sentences total 88 words; the 8 hypothesis sentences total 32 words.
   `brevity = math.exp(1 - 88 / 32) = math.exp(-1.75) = 0.1738`.
3. **Scale Factor**:
   The formula multiplies by `100.0`.
   `bleu = 100.0 * 0.1738 * 0.0544 = 0.9457` (rounds to **0.946**).
4. **Conclusion**:
   The score `0.946` was **0.946%** (or **0.00946** on a 0–1 scale). It represents virtually zero BLEU, fully consistent with `WER = 1.0000` (100% substitution error) and `ROUGE-L = 0.0000` (zero longest common subsequence).

### Independent Recomputation Comparison

| Metric | Reported Pilot | Recomputed Custom | Standard Recomputation (`sacrebleu` / `jiwer` / `rouge_score`) | Status |
|---|---|---|---|---|
| **WER** | 1.0 | 1.0000 | 1.0000 (100.0%) | Verified |
| **BLEU-4** | 0.946 | 0.9457 (scale 0-100) | **0.0000** (scale 0-100) | Verified (Laplace artifact) |
| **ROUGE-L** | 0.0 | 0.0000 | 0.0000 | Verified |
| **Diversity** | 0.25 | 0.2500 | 0.2500 | Verified (2 unique / 8) |
| **Collapsed** | True | True | True | Verified (87.5% mode collapse) |

---

## 3. Reference / Prediction Table

The 8 test samples evaluated by `random.Random(44).sample(test_records, 8)`:

| # | UID | Reference Text | Model Prediction | Custom BLEU-4 | SacreBLEU | WER | ROUGE-L |
|---|---|---|---|---|---|---|---|
| 1 | `sJlOxY0Rxdo--11` | "The air purifiers will purify the air," | "diksha necklace four diksha" | +1 smoothed | 0.0 | 1.0 | 0.0 |
| 2 | `wDO5sAwFFNI--11` | "These were all Tata CNG buses operated by Mateshwari Ltd. The reason for this is not known." | "diksha necklace four diksha" | +1 smoothed | 0.0 | 1.0 | 0.0 |
| 3 | `xByvfah4xt8--12` | "Putin loves to play Judo and ice hockey." | "diksha necklace four diksha" | +1 smoothed | 0.0 | 1.0 | 0.0 |
| 4 | `gxZKUIFDqqI--4` | "because they could not manage their food, entertainment and other expenses in the Rs 10,000 withdrawal limit." | "diksha necklace four diksha" | +1 smoothed | 0.0 | 1.0 | 0.0 |
| 5 | `jHalXZA36a4--10` | "It is their right." | "diksha necklace four diksha" | +1 smoothed | 0.0 | 1.0 | 0.0 |
| 6 | `r0S2Gv78jmM--43` | "Have a look at some images and videos." | "diksha necklace four diksha" | +1 smoothed | 0.0 | 1.0 | 0.0 |
| 7 | `ktbVEywyyAM--1` | "Miss India, Miss World & Miss Universe beauty pageants are various beauty pageants." | "diksha necklace four diksha" | +1 smoothed | 0.0 | 1.0 | 0.0 |
| 8 | `nIjAEHi-6HU--126` | "Have you ever eaten pumpkin flowers dipped in rice paste and fried? Try it!" | "diksha necklace four diksha necklace" | +1 smoothed | 0.0 | 1.0 | 0.0 |

> [!CAUTION]
> Translation quality from this eight-sample evaluation is ZERO. The predictions are repetitive mode-collapse artifacts.

---

## 4. Training Records and Tokenization Audit

### Construction of the 7,500 Pairs
1. `iSign_v1.1.csv` provides text keyed by `uid`.
2. `manifest.json` provides the mapping of 7,500 UIDs to 4 zip shards (`shard-00000.zip` to `shard-00003.zip`).
3. In `signsage_dual_model_training.ipynb`:
   - `sorted_uids = sorted(uid_to_shard.keys())`
   - Split assignment was created by positional slicing: `train` = 0:2500, `validation` = 2500:5000, `test` = 5000:7500.
   - Text is paired via `csv_map[uid]`.

### Training Loop Tokenization Behavior
- **Prepared Token Arrays (`tokens.npz`)**: **NOT USED** by the active notebook training loop. `tokens.npz` is only loaded to print its tensor shapes in cell 11.
- **Dynamic Tokenization**: **YES.** `TranslationDataset.__getitem__` dynamically encodes English annotations on each sample call:
  `translation_token_ids(row["text"], self.tokenizer)`
- **Prepared SentencePiece Model**: **YES.** Uses `kaggle_input_v4/s2t_small6/tokenizer/signsage_en_sp.model` (vocab size 16,000, fingerprint `2d88a5de48318cb9`).
- **Newly Generated Token Targets**: **YES**, generated dynamically per batch via `tokenizer.encode()`.

### Effect of Unresolved `tokens_meta.json` UID Mapping
- In `signsage_dual_model_training.ipynb`: **Does not affect training**, because the notebook tokenizes directly from CSV text rather than relying on index-matched arrays from `tokens.npz`.
- In standalone `train_sign_to_text.py`: **Blocks cache reuse**, because `tokens_meta.json` stores SHA-256 hashes (`"ec955196b206afde"`) rather than UID lists, causing fingerprint checks to mismatch and trigger cache rebuilding.
- **Exact Token Alignment**: Cannot be claimed from `tokens.npz` due to the lack of transparent UID lists in the metadata.

---

## 5. Split Integrity Report

### Split Counts and Overlaps

| Split | Records | Unique UIDs | Unique Source Videos | UID Overlap | Source Video Overlap |
|---|---|---|---|---|---|
| **Train** | 2,500 | 2,500 | 680 | — | — |
| **Validation** | 2,500 | 2,500 | 762 | Train ∩ Val = **0** | Train ∩ Val = **1** (`JTQjTcnpejE-`) |
| **Test** | 2,500 | 2,500 | 768 | Val ∩ Test = **0**, Train ∩ Test = **0** | Val ∩ Test = **1** (`dPuKX_lM4Zc-`), Train ∩ Test = **0** |

### Split Integrity Findings
1. **UID Disjointness**: Splits are completely disjoint at the UID level (0 overlap).
2. **Source-Video Leakage**: Positional slicing of alphabetically sorted UIDs resulted in video leakage across split boundaries (video `JTQjTcnpejE-` spans Train/Val; video `dPuKX_lM4Zc-` spans Val/Test).
3. **Signer Independence**: **UNVERIFIED.** `iSign_v1.1.csv` contains only `uid` and `text`. No signer column exists.
4. **Generalization Validity**: **INVALID.** The pilot evaluation (N=8 samples, split video leakage, no signer metadata, complete mode collapse) cannot support any generalization claims.

---

## 6. Checkpoint Recovery Verification

### Checkpoint Recovery from Archive
Previous reports claimed the pilot checkpoint was lost in ephemeral `/tmp`. During this audit, the actual Model B pilot training checkpoint was extracted from `results-v4.zip`:
- Archive path: `s2t_small6/checkpoints/signsage_transformer_full.pt`
- File size: **203,649,677 bytes** (195 MB)
- Copied to persistent path: `outputs/checkpoint_reload_verified/signsage_transformer_full_16k.pt`

### Reload Verification into Fresh Model B Instance
- **Architecture**: `SignTransformer` (`backend.app.ml.models.sign_to_text`)
- **Config**:
  - `input_dim`: 3258
  - `vocab_size`: 16000
  - `max_frames`: 160
  - `max_tokens`: 96
  - `d_model`: 256
  - `nhead`: 4
  - `layers`: 4
  - `ff`: 1024
  - `dropout`: 0.15
  - `norm_first`: True
- **Weight Loading**: `load_state_dict(strict=True)` succeeded with **0 missing** and **0 unexpected** keys (135 tensors).
- **Tokenizer Fingerprint**: Checkpoint stores `2d88a5de48318cb9`, which **strictly matches** the SentencePiece model at `kaggle_input_v4/s2t_small6/tokenizer/signsage_en_sp.model`.
- **Data Fingerprint**: Checkpoint stores `None`.
- **Optimizer State**: Loaded into `torch.optim.AdamW`; learning rate verified at `8.51e-5`.
- **Scheduler State**: Verified (`last_epoch = 1`).
- **Epoch & Step State**: `epoch = 1`, `best val loss = 26.7266`, `bad = 0`.
- **Forward Pass Verification**:
  - Input: batch size 2, 160 frames, 1629 feature dim, 12 tokens.
  - Output shape: `torch.Size([2, 12, 16000])`.
  - Finite check: **All finite = True, NaN = False** (min=-43.12, max=42.44, mean=0.24).
- **Checkpoint Reload Status**: **PASSED AND PERSISTENTLY VERIFIED.**

---

## 7. Kaggle Compatibility Audit

| Parameter | Local Environment Verification | Actual Kaggle Environment Verification |
|---|---|---|
| **Dataset Attachment** | `kaggle_input_v4/` present and verified locally | Not verified in live Kaggle container (requires manual "Add Data") |
| **`PREPARED_CACHE_ROOT`** | Pointing to local `/mnt/D/Codes/signsage/kaggle_input_v4` | **Incompatible as-is.** Hardcoded local path will raise `FileNotFoundError` on Kaggle |
| **Dependencies** | `torch`, `sentencepiece`, `sacrebleu`, `jiwer`, `pose_format` verified | Standard Kaggle Python lacks `pose-format`, `sacrebleu`, `jiwer` without `!pip install` |
| **CUDA Acceleration** | `torch.cuda.is_available() == False` (CPU only) | Not verified locally; requires GPU P100 / T4 x2 enabled in Kaggle settings |
| **Artifact Persistence** | Output files saved persistently to `outputs/` | In Kaggle, outputs persist only if saved as a notebook version or downloaded |
| **Session Recovery** | Verified using reloaded checkpoint state dict | On Kaggle, resuming requires mounting prior run outputs as an input dataset |

---

## 8. Summary of Remaining Blockers

1. **Training Instability & Mode Collapse**:
   The model collapsed during training (100% gradient clipping, top prediction covering 87.5% of outputs). Low cross-entropy loss does not equate to learned translation.
2. **Evaluation Implementation**:
   Laplace +1 smoothing in `translation_metrics` produces misleading BLEU figures. Evaluation must use standard SacreBLEU without add-1 smoothing on zero counts.
3. **Data Splitting**:
   Splits must be grouped by `video_id` to prevent cross-split video leakage.
4. **Sample Size**:
   Evaluating on 8 samples is statistically meaningless. Evaluation must run on the full test split (2,500 samples).
5. **Portability of Notebook Paths**:
   `PREPARED_CACHE_ROOT` must dynamically detect Kaggle input paths rather than hardcoding local filesystem paths.

**Promotion Decision**: `SIGNSAGE_MODEL_PROMOTED=false` is strictly maintained. The pilot is technically invalid as evidence of generalization or translation quality.
