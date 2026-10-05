# T5-small Pose-to-Text POC (research only — never serving)

## Reused existing code (no duplication)
- `research/training/audit_dataset.py`: `clean_dataframe`, `video_split`
  (official cleaning + video-level 80/10/10 split; subset keeps whole videos)
- `research/training/train_sign_to_text.py`: `read_normalized_pose`,
  `load_pose_item`, `RemotePoseSource`/`RemoteConcatFile`, `preflight_pose_source`
  (canonical `pose_normalize_v1`, MAX_FRAMES=160, [160,1629] + true length)
- `backend/app/ml/preprocessing/`: `normalize_pose_sequence`,
  `resize_sequence`, landmark constants (543×3 = 1629)
- V1 collapse reference: `research/evaluation/reports/352868924/`
  (diversity 0.99%, top-pred 44.8% → the bar this POC must beat)

## Tokenizer decision (architectural)
The existing 16k SignSage SentencePiece unigram vocabulary is INCOMPATIBLE
with pretrained T5: T5-small's embeddings/decoder are trained for its own
32k SentencePiece vocabulary (pad=0, eos=1, unk=2). Reusing the 16k vocab
would discard all pretrained language knowledge. Therefore this POC uses
the official `google-t5/t5-small` tokenizer + pretrained decoder, and the
checkpoint records `tokenizer: google-t5/t5-small` for reproducibility.

## Files
- `model.py` — PoseProjection + PoseT5 (`inputs_embeds` wiring)
- `materialize.py` — small video-level subset → fp16 `.npz` shards (Kaggle)
- `train.py` — Stage A (frozen T5) / Stage B (joint) + synthetic wiring gate
- `evaluate.py` — WER/BLEU-1..4/ROUGE-L/EM/diversity/baseline/sensitivity

## Commands
Local wiring gate (CPU, no dataset, no token):
`backend/.venv/bin/python research/t5_pose_poc/train.py --synthetic-gate --out ./research/evaluation/t5_poc_gate`
(committed record: `research/evaluation/t5_poc_gate/gate_report.json`)

Kaggle GPU (needs HF_TOKEN with iSign access):
`python materialize.py --csv /path/iSign_v1.1.csv --pose-source remote --out ./t5_poc_shards`
`python train.py --shards ./t5_poc_shards --out ./t5_poc_5k --epochs 8`
`python evaluate.py --checkpoint ./t5_poc_5k/t5_pose_projection.pt --shards ./t5_poc_shards --out ./t5_poc_eval`

## Extra deps (research venv only, NOT added to backend/requirements.txt)
`transformers huggingface_hub pose-format` (+ `nltk` for BLEU-1..4)
