# Evaluation report — external iSign t5-small-motion (dataset-level, real data)

Evaluation verdict: **NO_GO** — exact-match accuracy 0/1887 (normalized) on real held-out data; output distribution severely degenerate (dominant prediction 21.3% of all samples, unique-prediction ratio 0.1579)
(Verdict scope: candidate-level evaluation verdict recorded in this notebook only; it does NOT change `SIGNSAGE_MODEL_PROMOTED`, the production registry, or any deployment gate.)

## 1. Model
- Repository: `manavdhamecha77/iSign-t5-pose-to-text` — checkpoint `t5-small-motion` ONLY (pose_dim=600, motion=True).
- Architecture: PoseEncoder MLP (Linear 600→256 → ReLU → Dropout 0.1 → Linear 256→512) + T5-small decoder.
- Tokenizer: `t5-small`. Generation: max_length=128, beams=4, early_stopping=True, length_penalty=2.0.
- Checkpoint location this run: `/root/.cache/huggingface/hub/models--manavdhamecha77--iSign-t5-pose-to-text/snapshots/80e60781f37f5c2720a7bc927398fdd9b0c32298/t5-small-motion`.
- No training, no fine-tuning, no weight changes anywhere in this notebook (`model.eval()` + `torch.no_grad()` only).

## 2. Dataset and held-out split
- Source: iSign held-out test split (video-level 80/10/10, seed 42) from `/root/.cache/huggingface/hub/datasets--Exploration-Lab--iSign/snapshots/e4ee6c5f0d9dfcbc74205e3f1388ce94da26c298/iSign_v1.1.csv`; full target 1887.
- Requested: 1887 | recorded: 1887 | fetched OK: 1887 | fetch failed: 0 | inferred OK: 1887.
- Pose access: authenticated HTTP Range requests against the 4-part ZIP (ONE ZIP64 archive); only evaluated members retrieved, archive never downloaded (source: remote, chunk 4 MB).
- Production pipeline untouched; registry untouched; `SIGNSAGE_MODEL_PROMOTED` untouched (no promotion).

## 3. Preprocessing contract (frozen, verified)
- Verified external contract ONLY: 75 landmarks (POSE 33 + LEFT_HAND 21 + RIGHT_HAND 21; FACE discarded), confidence=1.0 fallback → 300 values/frame, z-score across axis=0 BEFORE motion, velocity[0]=0 + velocity[t]=pose[t]-pose[t-1], concat [pose|velocity] → 600/frame, pad/truncate to exactly 500 frames. No SignSage hip-centering, no preprocessing changes.
- Eval rule: normalized exact-match (prediction == ground truth after lowercase/whitespace norm) counts as correct; raw byte-exact equality reported alongside. Fluent English output was NEVER counted as correct unless it exactly matched ground truth. Near-paraphrases count as incorrect by design (no silent leniency).

## 4. Hardware and timing
- Device: cuda.
- Avg fetch: 0.529s | avg inference: 0.204s | median inference: 0.175s.
- Total fetch: 998.0s | total inference: 384.3s | elapsed: 1385.8s.

## 5. Exact-match results
- Normalized: 0.0000 (0/1887).
- Raw: 0.0000 (0/1887).
- Incorrect: 1887.

## 6. Prediction distribution
- Unique predictions: 298 / 1887 (ratio 0.1579).
- Top-1 prediction covers 21.3% of all samples (401x: "isn't that amazing!").
- Top-5 predictions cover 34.3% of all samples.
- Singleton predictions (occur exactly once): 140 (7.4%).
- Empty predictions: 0 (rate 0.0000).
- Prediction length (chars): mean 46.2 | median 40.0 | min 3 | max 115.
- Prediction length (words): mean 10.0 | median 8.0 | min 1 | max 28.
- Unique expected labels: 1815.

Top-10 predictions:
- 401x (21.3%): "isn't that amazing!"
- 83x (4.4%): 'he said that he is very happy that he will not be able to make ends meet.'
- 57x (3.0%): 'he said that if he wants to become a doctor, he will be able to work hard and earn money.'
- 54x (2.9%): 'he was very happy to see such a huge crowd.'
- 52x (2.8%): 'he said that if he wants to resign, he will not be able to win the elections.'
- 51x (2.7%): 'he was very happy. he was very happy.'
- 49x (2.6%): 'he was very happy. he was very happy to see him.'
- 40x (2.1%): 'he said that if he wants to become a doctor, he will be able to work hard and earn a living.'
- 39x (2.1%): 'he said that if he wants to become a doctor, he would not be able to work hard and earn money.'
- 34x (1.8%): 'a tiger and a lion.'

Top-10 expected labels:
- 5x: '3'
- 4x: 'reading is fun.'
- 4x: "let's talk."
- 4x: '2'
- 4x: 'page number'
- 4x: '1'
- 4x: '4'
- 3x: 'nest'
- 3x: 'to the teacher.'
- 3x: 'reading is fun!'

## 7. Mode-collapse diagnostic (ANALYSIS ONLY)
- ANALYSIS diagnostic only -- describes output concentration; it is NOT an official promotion criterion (the repository defines no cutoff for this candidate) and changes no gate.
- Dominant prediction: 401x (21.3%): "isn't that amazing!".
- Top-5 cumulative share: 34.3%.
- Unique-prediction ratio: 0.1579.
- Highly concentrated: True (descriptive heuristic for this analysis line only: top1_share >= 10% OR unique_ratio <= 5%).

## 8. Hallucination-risk observations
- Wrong predictions are NOT automatically hallucinations; this section reports repetition/fluency patterns that warrant the controlled negative tests in docs/EXTERNAL_EVALUATION.md (§4).
- Random/noise probe: not performed in this notebook (execution smoke tests used synthetic input only and make no quality claim); empty-frame / random-movement negative tests belong to the separate webcam protocol
- Incorrect samples: 1887; top wrong prediction covers 21.3% of them (401x: "isn't that amazing!").
- Empty-output count: 0 (rate 0.0000); empty output is a correct rejection, not a translation.

Most common incorrect predictions:
- incorrect prediction 401x: "isn't that amazing!"
- incorrect prediction 83x: 'he said that he is very happy that he will not be able to make ends meet.'
- incorrect prediction 57x: 'he said that if he wants to become a doctor, he will be able to work hard and earn money.'
- incorrect prediction 54x: 'he was very happy to see such a huge crowd.'
- incorrect prediction 52x: 'he said that if he wants to resign, he will not be able to win the elections.'
- incorrect prediction 51x: 'he was very happy. he was very happy.'
- incorrect prediction 49x: 'he was very happy. he was very happy to see him.'
- incorrect prediction 40x: 'he said that if he wants to become a doctor, he will be able to work hard and earn a living.'
- incorrect prediction 39x: 'he said that if he wants to become a doctor, he would not be able to work hard and earn money.'
- incorrect prediction 34x: 'a tiger and a lion.'

Top expected->predicted pairs:
- 2x expected='wind' predicted='and a tiger.'
- 2x expected="let's talk." predicted="isn't that amazing!"
- 2x expected='4' predicted='i am going to tell you about it.'
- 2x expected='i am an animal.' predicted="isn't that amazing!"
- 2x expected='muvendar' predicted="isn't that amazing!"
- 1x expected='page 53' predicted="isn't that amazing!"
- 1x expected='unit 5' predicted="isn't that amazing!"
- 1x expected='merry go round.' predicted="isn't that amazing!"
- 1x expected='listen, sing and dance.' predicted="isn't it?"
- 1x expected='i climb up on the merry-go-round.' predicted='he said that he is very happy that he will not be able to do anything.'

## 9. Limitations
- Sentence-level labels: no word-level confusion matrix; pairs above are the compact substitute.
- Confidence fallback 1.0 (no per-landmark confidence in this pipeline; training distribution unknown).
- Studio training data vs any webcam distribution shift is NOT measured here.
- Held-out split reconstructed via video-level policy (seed 42); if the publisher's exact 1,887 IDs differ, counts shift -- actual n is recorded above.
- Accuracy is exact-match only; near-paraphrases count as incorrect by design (no silent leniency).
- Remote member fetch can fail transiently (network/CDN); failures are recorded per-sample as fetch_error and never silently retried into invented predictions. predictions.csv is written incrementally so interruptions lose nothing.
- Chunk granularity (4 MB) bounds per-member transfer waste.
- Verdict scope: Candidate-level evaluation verdict for t5-small-motion recorded in this notebook only; does NOT change SIGNSAGE_MODEL_PROMOTED, the production registry, or any deployment gate.

## 10. Evaluation conclusion
- Real-data evaluation completed successfully (preflight PASS; 1887/1887 requested samples fetched and inferred).
- Exact-match accuracy was 0/1887 (normalized 0.0000; raw 0/1887).
- The model therefore does NOT demonstrate usable recognition quality on this evaluation set.
- Repeated fluent incorrect outputs (dominant 401x at 21.3%, top-5 cumulative 34.3%) indicate severe prediction concentration / failure behavior.
- This notebook does NOT authorize production deployment or model promotion.

Artifacts: predictions.csv, metrics.json, confusion_matrix.csv, evaluation_report.md (total ~0.42 MB).
