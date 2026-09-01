# SignSage — Data Model

## Training data: iSign

Source: Joshi et al., "iSign: A Benchmark for Indian Sign Language Processing," ACL
Findings 2024. Hosted (gated, research-only license) at
`huggingface.co/datasets/Exploration-Lab/iSign`.

**Known facts (confirmed by running Sections 0-4 of the notebook against the real archive):**
- The published iSign description refers to 118,228 video-sentence/phrase pairs, but the
  local v1.1 archive actually has **127,237 CSV rows and 127,237 `.pose` files**. The
  notebook does not silently reconcile this discrepancy; it reports the local counts and
  uses the files actually present.
- **Pose files are one-per-UID (row-level), not one-per-`video_id`.** File count equals
  row count 1:1, which resolves the earlier "more files than rows" mystery: each CSV row
  is resolved through its full UID, i.e. `<video_id>-<seq>.pose`. `df["video_id"].nunique()`
  is used only for the leak-safe split (Section 3), never for file naming.
- Only **593** of these pairs were verified by certified ISL signers (limited signer
  availability, per the paper). The rest is aligned/scraped at unknown quality. Treat
  this as a real data-quality risk, not a footnote — see the audit section of
  `notebooks/signsage.ipynb`.
- Poses are pre-extracted and shipped as `.pose` files (read via the `pose-format`
  Python package, `Pose.read()`), not raw video, for the pose-based tasks.
- Translations live in `iSign_v1.1.csv`. **Confirmed columns include `uid` and `text`**
  (the notebook asserts both are present, then derives `video_id`/`seq` via
  `uid.str.rsplit("-", n=1)` — rsplit is deliberate because some `video_id`s end in `-` —
  plus `n_words` from the text). UID example: `1782bea75c7d-7`.
- **Per-segment timing: confirmed absent, and not needed.** The local CSV has no
  "start"/"end"/"time" columns (the notebook checks and does not invent them). This is
  fine here precisely because the archive is one `.pose` file per UID/row, not one file
  per `video_id` — so there is no multi-segment-per-file slicing problem; each row
  already has its own dedicated pose sequence.
- Recommended split: by `video_id`, not by row, to avoid leaking segments of the same
  video across train/val/test (per the dataset's own documentation).
- iSign defines 5 tasks total; this project uses 3 (ISL-to-English via video and via
  pose, and English-to-ISLPose). The other 2 (CISLR isolated-word recognition, semantic
  similarity) are not used, though CISLR could later serve as an isolated-sign baseline
  from the *same* signer/domain family as iSign, which would be more consistent than
  mixing in INCLUDE (see PRD.md competitive/baseline notes).

**Pose file shape (confirmed):** `(frames, people, keypoints, xyz)`. The notebook uses
person 0 and the first **543 image-space keypoints** — `POSE_LANDMARKS` (33) +
`FACE_LANDMARKS` (468) + `LEFT_HAND_LANDMARKS` (21) + `RIGHT_HAND_LANDMARKS` (21) — out
of 576 total; the trailing `POSE_WORLD_LANDMARKS` (33) is a world-coordinate duplicate
block and is excluded. Flattened feature dim = 1629 (543×3); the no-face ablation drops
the 468 face points, leaving 225 (75×3). A per-point confidence array
(`p.body.confidence`) is also present.

## Secondary baseline data: INCLUDE

263 isolated ISL signs, 15 categories, ~4,292 source videos. Used only as an isolated-
sign baseline comparator (per the proposal), not for continuous translation training.

## Dataset access paths (two, environment-dependent)

The notebook resolves `DATA_DIR` from its environment, and the two modes reach the same
logical dataset by very different routes:

**Kaggle mode (the training path).** Only `iSign_v1.1.csv` (9.81 MB) is downloaded, via
`hf_hub_download`. The ~231 GB pose archive is **never uploaded to Kaggle and never
extracted**. Its four published parts —

```text
iSign-poses_v1.1_part_aa
iSign-poses_v1.1_part_ab
iSign-poses_v1.1_part_ac
iSign-poses_v1.1_part_ad
```

— are consecutive byte ranges of one ZIP archive. `RemoteConcatFile` presents them as a
single seekable virtual file so Python's `zipfile` can read the ZIP64 central directory
and then fetch only the members the selected rows need, in 256 KB HTTP range chunks (32
chunks cached in memory, FIFO). Member lookup tries `iSign-poses_v1.1/{uid}.pose` then
`{uid}.pose`. Requires Kaggle Internet and an `HF_TOKEN` secret (see SECURITY.md).

**Local mode (fallback).** `SIGNSAGE_DATA_DIR` points at an already-extracted tree:

```text
iSign/
├── iSign_v1.1.csv
└── poses/
    └── iSign-poses_v1.1/
        └── *.pose
```

Both modes produce identical arrays downstream — `read_pose_bytes_for_uid()` is the seam.

## Notebook-local pose cache (derived data, not source data)

The notebook writes `<POSE_CACHE_DIR>/<uid>.npy` — one float16 array per UID, holding
the already-normalized, variable-length 543-landmark sequence (the output of
`normalize_pose_sequence()`, before any `MAX_FRAMES`/`POSE_OUTPUT_FRAMES` resizing or
face masking), so both the Sign→Text and Text→Pose loaders read from the same cached
array. This exists purely to avoid re-parsing (locally) or re-fetching (on Kaggle) the
original `.pose` data on every epoch (see DECISIONS.md).

`POSE_CACHE_DIR` differs by mode, deliberately:

| Mode | Path | Lifetime |
|---|---|---|
| Kaggle | `/kaggle/temp/signsage_cache/normalized_v1/` | Session only — **not** persisted |
| Local | `<SIGNSAGE_DATA_DIR>/cache/normalized_v1/` | Persists across runs |

On Kaggle it is kept out of `/kaggle/working` so a multi-GB regenerable cache doesn't
consume the 20 GB output budget meant for checkpoints and reports. Size is roughly
521 KB per sample (160 frames × 1629 features × 2 bytes), so ~2-3 GB at current limits.

It is derived data, not source data: safe to delete entirely and let it rebuild. Writes
are atomic (temp file + `os.replace`), and a failed load is treated as a cache miss and
rebuilt rather than crashing the pipeline.

## Project knowledge repository (SQLite + JSON)

Every run writes a structured record of itself, so a checkpoint can always be traced back
to the experiment, hyperparameters, split, and dataset version that produced it. This is
experiment provenance, **not** a RAG knowledge base — nothing retrieves from it at
inference time.

Artifacts land under `signsage_outputs/` (`/kaggle/working/signsage_outputs/` on Kaggle,
`./signsage_outputs/` locally). The completed bounded Kaggle run has now been observed to
produce this structure, including trained checkpoints, a report/vocabulary artifact, and the
SQLite/JSON provenance artifacts. Exact metric values remain an experimental-result concern,
not an artifact-existence claim:

```text
signsage_outputs/
├── checkpoints/                        trained model checkpoints, best model(s)
├── reports/                            metrics, vocabulary, experiment summaries
├── signsage_knowledge.db               queryable relational record
└── signsage_knowledge_manifest.json    portable per-run snapshot
```

**Observed in the completed Kaggle output:** four checkpoint files are present under
`checkpoints/`, a `token_vocabulary.json` report artifact is present under `reports/`, and both
`signsage_knowledge.db` and `signsage_knowledge_manifest.json` were generated. These observations
confirm artifact creation; they do not substitute for reading the recorded metrics and verdict.

**Split of responsibility.** SQLite holds the relational history, because "which
experiment produced the current best model, and under what hyperparameters" is a
join-shaped question. JSON holds the flat portable snapshot (configuration, manifests,
vocabulary info, run summary) that the backend can read without opening a database.

**Schema** — seven tables, all keyed on `run_id` (a UTC timestamp, `%Y%m%dT%H%M%SZ`),
with `PRAGMA foreign_keys = ON`:

| Table | Grain | Holds |
|---|---|---|
| `runs` | one row per run (PK) | `started_at`, `completed_at`, `data_mode`, `run_mode`, `device`, `gpu_count`, `seed` |
| `dataset` | one row per run (PK) | dataset name/version, CSV path and row count, unique `video_id` count, pose source, selected rows |
| `splits` | one row per run (PK) | train/val/test row and video counts, `leakage_free` flag |
| `experiments` | one row per model per run | `name`, `family`, `input_features`, epochs, batch size, learning rate, weight decay, `checkpoint_path`, `best_validation_loss` |
| `metrics` | one row per (experiment, metric, split) | `metric_name`, `metric_value`, `split_name` |
| `artifacts` | one row per named artifact per run | `artifact_type`, `path`, `description` |
| `observations` | append-only log | arbitrary key/JSON-value notes (data audit findings, go/no-go verdict) |

**Idempotency — this is a deliberate design point, not an implementation detail.**
Notebook cells get re-run. `dataset` and `splits` are keyed on `run_id` alone, so
`INSERT OR REPLACE` suffices. `experiments`, `metrics`, and `artifacts` all use
autoincrement primary keys and therefore have no natural uniqueness, so their writers
delete the matching prior row before inserting:

- `knowledge_insert(..., delete_keys={...})` — callers writing to `experiments` pass the
  keys that identify the logical row. The generated `WHERE` uses `IS ?`, not `= ?`, so it
  matches correctly when a key is `NULL`.
- `knowledge_metric(...)` — replaces on `(run_id, experiment_name, metric_name, split_name)`.
- `knowledge_artifact(...)` — replaces on `(run_id, artifact_name)`.
- `knowledge_observation(...)` — **append-only by design.** Observations are log-style
  notes where a re-run recording a second entry is the correct behavior, not a duplicate.

The traceability chain this supports: dataset → data preparation → experiment → model →
metrics → selected model → deployment artifact.

## Runtime data (proposed, not yet implemented)

Once the backend exists, a minimal SQLite schema for sentence caching:

```sql
CREATE TABLE translations (
    id INTEGER PRIMARY KEY,
    direction TEXT NOT NULL,      -- 'sign_to_text' | 'text_to_sign'
    input_hash TEXT NOT NULL,     -- hash of input (landmark sequence or text), not raw content
    output TEXT NOT NULL,         -- resulting text or a reference to a stored pose sequence
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Deliberately minimal: no user accounts, no raw video/landmark storage by default (see
SECURITY.md — landmarks are processed in-memory per session unless a future feature
explicitly requires persistence, which does not exist yet).

## Vocabulary / tokenizer

The feasibility notebook uses a deliberately dumb whitespace + lowercase tokenizer with
`<pad>/<bos>/<eos>/<unk>` specials, vocab built from the training split only. No subword
tokenization at this stage (see DECISIONS.md). Confirmed vocab size on a full local
training split: 60,384 word types.
