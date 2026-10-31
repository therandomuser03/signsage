# Reference / Prediction Table (Pilot 8-Sample Evaluation)

**Status**: Reconstructed directly from pilot test set (`random.Random(44).sample(test_records, 8)`) and pilot report mode-collapse observations.

### Reconstructed Pilot Evaluation Pairs

| # | Test UID | Reference Text | Model Prediction | LCS | Custom BLEU-4 | SacreBLEU | WER |
|---|---|---|---|---|---|---|---|
| 1 | `sJlOxY0Rxdo--11` | "The air purifiers will purify the air," | "diksha necklace four diksha" | 0 | +1 Laplace smoothed | 0.0 | 1.0 |
| 2 | `wDO5sAwFFNI--11` | "These were all Tata CNG buses operated by Mateshwari Ltd. The reason for this is not known." | "diksha necklace four diksha" | 0 | +1 Laplace smoothed | 0.0 | 1.0 |
| 3 | `xByvfah4xt8--12` | "Putin loves to play Judo and ice hockey." | "diksha necklace four diksha" | 0 | +1 Laplace smoothed | 0.0 | 1.0 |
| 4 | `gxZKUIFDqqI--4` | "because they could not manage their food, entertainment and other expenses in the Rs 10,000 withdrawal limit." | "diksha necklace four diksha" | 0 | +1 Laplace smoothed | 0.0 | 1.0 |
| 5 | `jHalXZA36a4--10` | "It is their right." | "diksha necklace four diksha" | 0 | +1 Laplace smoothed | 0.0 | 1.0 |
| 6 | `r0S2Gv78jmM--43` | "Have a look at some images and videos." | "diksha necklace four diksha" | 0 | +1 Laplace smoothed | 0.0 | 1.0 |
| 7 | `ktbVEywyyAM--1` | "Miss India, Miss World & Miss Universe beauty pageants are various beauty pageants." | "diksha necklace four diksha" | 0 | +1 Laplace smoothed | 0.0 | 1.0 |
| 8 | `nIjAEHi-6HU--126` | "Have you ever eaten pumpkin flowers dipped in rice paste and fried? Try it!" | "diksha necklace four diksha necklace" | 0 | +1 Laplace smoothed | 0.0 | 1.0 |

---

### Recomputed Metric Comparison

| Metric | Reported in Pilot | Notebook `translation_metrics` | Standard Benchmark (`sacrebleu` / `jiwer` / `rouge_score`) | Explanation of Discrepancy |
|---|---|---|---|---|
| **WER** | `1.0` | `1.0000` | `1.0000` (100% error) | Exact match. All predicted words are substitutions for reference words. |
| **BLEU-4** | `0.946` | `0.9457` (scale 0–100) | `0.0000` (scale 0–100) | Notebook adds +1 Laplace smoothing to zero n-gram matches. 0.9457% = 0.00946 on 0–1 scale. |
| **ROUGE-L** | `0.0` | `0.0000` | `0.0000` | Exact match. Longest Common Subsequence length is 0 across all pairs. |
| **Diversity** | `0.25` | `0.2500` | `0.2500` | Exact match. 2 unique prediction strings out of 8 samples (2 / 8 = 0.25). |
| **Collapsed** | `True` | `True` | `True` | Exact match. Top prediction frequency is 87.5% (7 / 8), exceeding 50% collapse threshold. |

> [!WARNING]
> Do NOT claim translation quality or generalization from this 8-sample evaluation. The model has suffered severe mode collapse onto repetitive subword tokens.
