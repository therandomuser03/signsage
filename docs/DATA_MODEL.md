# SignSage — Data Model

**Status:** Current research/data reference  
**Important:** Dataset facts below are documented from the project's completed notebook audits.

## 1. iSign dataset

Source: Joshi et al., *iSign: A Benchmark for Indian Sign Language Processing*, ACL Findings 2024.

The project uses the gated `Exploration-Lab/iSign` dataset. The documented license is **CC BY-NC-SA 4.0**.

### Confirmed local archive facts

- Published descriptions refer to 118,228 video-sentence/phrase pairs.
- The audited v1.1 archive contains **127,237 CSV rows and 127,237 `.pose` files**.
- Pose files are one-per-UID/row, not one-per-`video_id`.
- The UID follows the general `<video_id>-<sequence>` structure.
- `uid` and `text` are confirmed CSV columns.
- `video_id` and `seq` are derived from `uid` using a right split because some video IDs may contain `-`.
- `n_words` is derived/used for text-quality filtering.
- Per-segment start/end timing columns were not found in the audited CSV.
- The project uses `video_id` for the train/validation/test split to avoid leakage between segments from the same source video.
- Only a small subset of examples was verified by certified ISL signers; the remaining aligned data should be treated as a data-quality risk.

## 2. Text-quality issue

The Phase 0 audit found roughly 1% junk/placeholder text.

Examples recorded in the research documentation include:

- blank strings,
- single digits,
- single letters,
- `blank`,
- caption artifacts such as `[Music]`.

Frequency-based statistics must therefore use the dataset's `n_words` field or an equivalent quality filter. Filtering only empty strings is insufficient.

## 3. Pose representation

The confirmed pose component structure is:

| Component | Count |
|---|---:|
| `POSE_LANDMARKS` | 33 |
| `FACE_LANDMARKS` | 468 |
| `LEFT_HAND_LANDMARKS` | 21 |
| `RIGHT_HAND_LANDMARKS` | 21 |
| `POSE_WORLD_LANDMARKS` | 33 |

The application/research Sign→Text representation uses the first 543 image-space landmarks:

```text
33 pose
+ 468 face
+ 21 left hand
+ 21 right hand
= 543 landmarks
```

With three coordinates per landmark this is:

```text
543 × 3 = 1629 scalar inputs/frame
```

The trailing world-coordinate pose block is excluded from this default representation.

## 4. Research/application separation

### Research data

Used for:

- training,
- validation,
- test evaluation,
- experiment provenance,
- checkpoint creation.

The application does not retrain during normal use.

### Application data

Application-side persistence may include:

- authenticated user records,
- translation history,
- model/capability metadata,
- application settings where implemented.

These are separate concerns from the research dataset.

## 5. Data handling rules

Do not:

- commit the iSign dataset,
- commit extracted pose archives,
- commit Hugging Face caches,
- hardcode access tokens,
- assume dataset counts from a paper are identical to the local archive,
- silently change a confirmed split or representation without recording the decision.

When the dataset schema changes, update this document and the relevant research code together.
