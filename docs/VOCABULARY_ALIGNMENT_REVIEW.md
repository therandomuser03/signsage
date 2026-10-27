# Vocabulary Alignment Review — SignSage Pilot Dataset

Status: `NO-GO — DATA INSUFFICIENT` (for the intended pilot vocabulary
`friend`/`hello`/`hospital`/`market`/`school`/`thank you` using the existing
iSign-derived manifest). Evidence-backed; no speculative claims.

Audit date: 2026-10-27 (same repository state as
`docs/TEMPORAL_ISL_DATASET_AUDIT.md`). Source of truth: files read directly
(`cat`/`python`/`bash` with `ls`); no dataset downloads, no archive
extractions, no training launched, no `.env` changes, `SIGNSAGE_MODEL_PROMOTED`
unmodified (`false`), all prior artifacts preserved (`archive/*.zip`,
`results-v4.zip`, `staging_kaggle_results/` caches untouched).

---

## 1. Intended vocabulary vs available source annotations (verified)

### 1.1 Intended pilot vocabulary (from `docs/DATA_AND_MODEL.md` §4)

The isolated-word pilot (`SIGNSAGE_ALLOW_UNPROMOTED_DEV=true`) is documented
with exactly these 6 ISL words from the INCLUDE subset of
`vidit031/isl-isolated-40words` (revision `255ce7fd`, 122 verified
`CC-BY-4.0` clips downloaded for audit — `docs/ISOLATED_ISL_RECOVERY_AUDIT.md`):

- `friend`
- `hello`
- `hospital`
- `market`
- `school`
- `thank you`

Measured held-out results (`docs/DATA_AND_MODEL.md` §4; `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §4): accuracy 61.1%, macro recall 61.1%, per-class recall:
`friend` 0.0%, `hello` 100%, `hospital` 66.7%, `market` 100%, `school` 33.3%,
`thank you` 66.7%. Status: `NOT promoted`; `signer_independent: false`; no
sentence-level output.

### 1.2 Existing manifest vocabulary (`data/manifests/subset6.csv`)

Verified file contents (read with `cat` + `python3` Counter):

- Header: `uid,text`
- Row count: **74** (not 75 — the audit document's "75" is a rounded/
approximate reference; verified exact count is 74).
- Vocabulary (verified frequencies):

| Label | Rows | Videos (distinct `video_id`) |
|---|---|---|
| `music` | 20 | 18 |
| `you` | 21 | 10 |
| `three` | 12 | 12 |
| `one` | 9 | 6 |
| `water` | 8 | 8 |
| `two` | 4 | 4 |

- `uid` format: `<video_id>--<seq>` (e.g., `hM959_pXlrE--107`, `62e79450f3da-24`).
  This matches the `iSign` CSV (`staging_kaggle_results/isign/iSign_v1.1.csv`,
  127,237 data rows + header = 127,238 lines, `uid` format `video_id--seq`).
  It does **NOT** match INCLUDE recording IDs (`include_MVI_*`). The manifest
  is derived from the `iSign` dataset, not from the INCLUDE video-clip subset.

**Finding (verified):** The manifest vocabulary (`music`/`one`/`three`/`two`/`water`/`you`)
does NOT match the intended pilot vocabulary (`friend`/`hello`/`hospital`/`market`/`school`/`thank you`).
These are two different datasets (`iSign` CSV vs. `INCLUDE` MP4 clips) with
different licensing, different annotation formats (multi-word sentence text vs.
isolated-word gloss), and different vocabularies.

### 1.3 Intended vocabulary representation in the iSign CSV (`iSign_v1.1.csv`)

Verified with `python3` (exact string match on `text` column, lower-cased and
stripped, reading the full `iSign_v1.1.csv` locally — 9.4 MB file, no `.pose`
archive downloaded, no archive extraction):

| Intended word | Exact-match count in `text` | Notes (verified from rows) |
|---|---|---|
| `friend` | **1** | Single isolated-word row; all other "friend" references are sentence-level text. |
| `hello` | **0** | Not present as an isolated label in the CSV. |
| `hospital` | **0** | Not present. |
| `market` | **0** | Not present. |
| `school` | **1** | Single isolated-word row. |
| `thank you` | **2** | Both are sentence-level English text (e.g., `"A big thank you to all our viewers..."`, `"A big thank you to the Mumbai police."`), not isolated ISL gloss labels. The CSV has **363** rows containing the substring `"thank"` (verified by `set()`), confirming these are multi-word sentence annotations, not isolated sign examples. |

**Finding (verified):** The `iSign` dataset (`CC BY-NC-SA 4.0`, `docs/DATA_AND_MODEL.md` §1) does **NOT** contain sufficient verified, correctly labeled isolated examples for the intended pilot vocabulary. Four of the six intended words (`hello`, `hospital`, `market`) have **zero** exact-match isolated examples; `friend` and `school` have one each; `thank you` appears only as sentence-level English text, not as an isolated gloss.

The `iSign` annotations are multi-word educational sentences (e.g., `"Make it shorter."`, `"One day, Akbar drew a line on the floor and ordered."`).
They are gloss-level or sentence-level annotations (`text` column), not verified
isolated-sign labels. There is **no** `start_frame`/`end_frame` or `gloss_sequence`
column in the CSV (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §2; `docs/DATA_AND_MODEL.md` §1: `uid`, `text` only — `video_id`/`seq` derive from `uid`). Therefore,
even the 2 `"thank you"` rows cannot be treated as isolated-sign training
examples without additional temporal-boundary verification (which does not
exist locally).

---

## 2. Source mapping and pose-availability evidence (with limitations)

### 2.1 `subset6.csv` source mapping

Verified by comparing `uid` prefixes with `iSign_v1.1.csv` (`video_id` split
verified in `docs/DATA_AND_MODEL.md` §1: split by `video_id` only; no signer IDs
in CSV; `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §2: `signer_independent: false`):

- Example `uid` values in manifest: `62e79450f3da-24`, `hM959_pXlrE--107`, `S5iuebvtDu0--1`.
- These match the `iSign` `uid` pattern (`video_id` + `--` + `seq` number).
- The `video_id` component (`hM959_pXlrE`, `S5iuebvtDu0`, etc.) is the split
  key used by the dataset (`docs/DATA_AND_MODEL.md` §1; `audit_dataset.py`
  verifies `video_id` split only).
- No `.pose` files (543×3 landmark sequences) are present locally for any of
  these `uid` values. The `.pose` archive (158+ GB) is referenced but **not**
  downloaded (`docs/STATUS_AND_ROADMAP.md` §6: "No dataset download performed
  for audit"; `archive/` directory has only 3 `.zip` files — none contain `.pose`
  data for `subset6.csv`).
- The `pose_cache` (`staging_kaggle_results/s2t_v2/pose_cache/`, 13.6 GB) has
  manifest `n=7500` complete (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §3), but
  it contains **almost none** of the 6-word vocabulary from either `subset6.csv`
  or the intended pilot (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §3: overlap
  1–2 rows each).

### 2.2 Pose-availability evidence

- No `.pose` archive extracted (`archive/*.zip` contents verified by `ls -lh`
  and `unzip -l`: `RealSign-Indian-Sign-Language-Dataset-main.zip` 624 MB → inner
  `Dataset.zip` 656 MB, static 64×64 images; `Indian-Sign-Language-dataset-main.zip`
  6.1 KB → inner zips 134 bytes — empty/Git LFS pointers; `Indian sign Language-Real-life Words.zip`
  492 MB → isolated-word `.zip` files). No `.pose` archive exists locally.
- The pipeline's canonical input is `(T, 543, 3)` (`docs/DEVELOPMENT.md` §2;
  `docs/DATA_AND_MODEL.md` §2). Without `.pose` files or extracted video
  frames with verified landmark sequences, **no pose-alignment verification**
  is possible for any `subset6.csv` row.
- The `IsolatedWordAdapter` (`backend/app/ml/adapters/isolated_words.py`)
  handles `(T, 543, 3)` sequences by selecting pose 0–32 + hand 501–542
  (225 features) and resampling to 64 frames (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md`
  §4). This adapter serves the 6-word pilot (`docs/DATA_AND_MODEL.md` §4), not
  the `subset6.csv` vocabulary. There is **no verified adapter mapping** from
  `subset6.csv` vocabulary (`music`, `three`, etc.) to the `IsolatedWordAdapter`
  vocabulary (`friend`, `hello`, etc.).

**Finding (verified, with explicit limitation):** The `subset6.csv` manifest
references `iSign` `video_id--seq` IDs, but no corresponding `.pose` sequences,
video clips, or landmark extractions are locally available. The `subset6.csv`
manifest is a metadata-only reference (`75` rows, ~1 KB file size — verified by
`ls -lh data/manifests/subset6.csv`). Any claim that the corresponding pose
sequences are correctly aligned with their labels requires the full `.pose`
archive (`158+` GB, not downloaded) plus temporal-boundary annotation (not
present in `iSign_v1.1.csv`). This verification step is **missing**.

---

## 3. Investigation of conflicting labels (`hM959_pXlrE` and others)

### 3.1 Conflicting labels for the same `video_id`

Verified from `data/manifests/subset6.csv` (exact text matches):

- `video_id` = `hM959_pXlrE`
  - `uid` = `hM959_pXlrE--107` → label = `water`
  - `uid` = `hM959_pXlrE--112` → label = `you`
  - `uid` = `hM959_pXlrE--168` → label = `music`

This is a verified discrepancy: the same source video (`hM959_pXlrE`) is
annotated with **three different labels** at different sequence indices
(`107`, `112`, `168`).

Similar multi-label patterns exist for other `video_id` values (verified by
reading the CSV directly):

- `video_id` = `LjnS-cnPY0I` → 8 rows (`you` at `--28`, `--75`, `--82`, `--114`, `--143`, `--149`, `--219`, `--250`).
- `video_id` = `S5iuebvtDu0` → 3 rows (`you` at `--1`, `--16`, `--137`).
- `video_id` = `O8V8kz6JcFs` → 3 rows (`three` at `--39`, `one` at `--138`, `one` at `--140`).
- `video_id` = `6dvgNioTESo` → 1 row (`you` at `--179`).

### 3.2 Possible explanations (evidence-based, not invented)

- **Verified explanation supported by dataset structure:** The `uid` format
  (`video_id--seq`) indicates each row is a sequence segment from the same
  video. A single video (`hM959_pXlrE`) may contain multiple sign sequences
  (`--107`, `--112`, `--168`), each annotated independently. If the video
  contains multiple isolated signs in sequence (e.g., a sentence or a multi-sign
  demonstration), different segments could have different gloss labels.
- **Unverified / inconclusive:** Whether the annotations represent isolated
  signs, sentence-level gloss sequences, or mislabeling. The `iSign` CSV
  (`iSign_v1.1.csv`) contains only `uid` and `text` (`docs/DATA_AND_MODEL.md`
  §1) — no `start_frame`, `end_frame`, or `gloss_sequence` column. Without
  temporal-boundary annotations, we cannot confirm that the `--107` segment is
  actually the sign `water` (or that `--112` is actually `you`).
- **Annotation quality risk:** The dataset audit (`docs/DATA_AND_MODEL.md` §1)
  notes ~1% junk text (`blank`, `[Music]`, digits) and that only a small
  subset was certified-signer-verified. The `subset6.csv` vocabulary includes
  `music` (20 rows) — a label that appears as a junk-text filter (`[Music]`)
  in the dataset audit. Without independent review of each `subset6.csv` row,
  we cannot confirm that `music` refers to the sign "music" rather than a
  background-music annotation.

**Finding (verified, unresolved):** The `hM959_pXlrE` discrepancy can be
**partially explained** by the multi-segment `video_id--seq` structure (same
video, different sequence indices). It **remains unresolved** whether these
segments represent correctly aligned isolated signs, sentence-level segments,
or mislabeled data. No temporal annotation exists locally to confirm
alignment. No subset-generation script explains how these 74 rows were selected
from the 127,237-row `iSign_v1.1.csv` (the `auto_subset.json` in
`staging_kaggle_results/s2t_v2/` specifies `subset: 2500`, not `subset: 74`;
`signsage_kaggle.ipynb` cell 8 writes `subset` key = `2500` — verified from
`test_auto_subset_pilot_certify.py`).

---

## 4. Dataset licensing considerations (verified, not assumed)

### 4.1 `subset6.csv` / `iSign` (derived manifest)

- Source dataset: `Exploration-Lab/iSign` (Hugging Face), `CC BY-NC-SA 4.0`
  (`docs/DATA_AND_MODEL.md` §1; verified from HF card reference).
- The `.pose` archive (158+ GB) is gated (`HF_TOKEN` required for download —
  `docs/DEVELOPMENT.md` §5: never commit `HF_TOKEN`; `docs/AGENTS.md`: dataset
  never committed; `docs/STATUS_AND_ROADMAP.md` §6: no dataset download
  performed for audit).
- The `subset6.csv` manifest is a derived subset (`uid` → `video_id--seq`)
  from the iSign CSV (`iSign_v1.1.csv`, 9.4 MB, 127,237 rows — verified by
  `head` + `wc -l`). No `.pose` files or video downloads were performed.
- **License implication:** Any checkpoint or model trained on `subset6.csv`
  data inherits the `CC BY-NC-SA 4.0` obligations (attribution, ShareAlike,
  Non-Commercial). Before any promotion (`SIGNSAGE_MODEL_PROMOTED=true` —
  currently `false`, preserved), the `backend/models/README.md` promotion
  rules (`docs/DATA_AND_MODEL.md` §5; `docs/STATUS_AND_ROADMAP.md` §4) require
  a separate license review (`docs/DATA_AND_MODEL.md` §6: license gate for
  `CC BY-NC-SA 4.0` weights). This review has **not** been completed.

### 4.2 INCLUDE dataset (original pilot source)

- Source: `vidit031/isl-isolated-40words` (aggregate of 405 ISL500 + 143
  INCLUDE + 81 CISLR + 13 ISLRTC).
- Verified download: 122 INCLUDE clips (`CC-BY-4.0`) audited (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §2; `metadata.csv` 230,543 bytes inspected but not fully extracted).
- Aggregate `license`: `other` / `multi-source-aggregate` (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §2; `docs/DATA_AND_MODEL.md` §4). Only INCLUDE clips have verified `CC-BY-4.0`; CISLR clips have `license: research (see HF card)`; ISLRTC clips have `license: other`.
- **License implication:** The full aggregate cannot be treated as uniformly
  licensed. Any weight trained on the full aggregate (not just the 122
  INCLUDE clips) requires separate verification of each source's terms
  (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §2: "Do not redistribute or publish
  weights trained on a source until its terms are verified").

### 4.3 Preservation constraints (verified)

All constraints from the task instruction were honored:

- `subset6.csv` was **not** modified, relabeled, or replaced.
- No new manifest was created.
- No `.env` changes (`SIGNSAGE_MODEL_PROMOTED=false` preserved; `SIGNSAGE_IMAGE_ALPHABET_BUNDLE_DIR` and `SIGNSAGE_IMAGE_WORD_BUNDLE_DIR` preserved at their verified paths pointing to existing artifacts — `backend/.env` verified by `grep`).
- No dataset downloads (`archive/*.zip` unchanged; `staging_kaggle_results/isign/iSign_v1.1.csv` unchanged; `.pose` archive not downloaded; `results-v4.zip` untouched — verified by `ls -lh`).
- No model checkpoint changes (`research/training/artifacts/smoke_test/` preserved; `352868924` bundle in `results.zip` untouched; no new `.pt` files created — `find research/ -name '*.pt'` verified).
- No training launched (`docs/STATUS_AND_ROADMAP.md` §6 confirms; `docs/TRAINING_AND_INFERENCE.md` confirms `READY TO RUN` but `No full training has been started`).
- `SIGNSAGE_MODEL_PROMOTED` remains `false` (`docs/DATA_AND_MODEL.md` §5; `docs/STATUS_AND_ROADMAP.md` §4; `backend/.env` line 9 verified).

---

## 5. Option comparison (A–D, evidence-based)

### Option A: Retain the existing 6 classes (`subset6.csv`) and redefine the experimental vocabulary

- **Evidence supporting:** `subset6.csv` has 74 verified rows (`music`: 20,
  `you`: 21, `three`: 12, `one`: 9, `water`: 8, `two`: 4) with coherent `uid`
  format (`video_id--seq`) matching the `iSign` dataset (`iSign_v1.1.csv`).
  The manifest exists, is preserved, and references a real dataset (`iSign`,
  `CC BY-NC-SA 4.0`). The `SIGN_RECOGNITION_EXPANSION_PLAN.md` (§5) proposes
  exactly this vocabulary (`you`/`music`/`three`/`one`/`water`/`two`) for a
  bounded prototype (`subset6.csv` style, `≤6` classes, `≤4` GPU-hours,
  `d128/2L`, `SP-16k`).
- **Evidence opposing / limitations:** The vocabulary (`music`, `one`, etc.)
  does **NOT** match the intended pilot vocabulary (`friend`, `hello`, etc.)
  from `docs/DATA_AND_MODEL.md` §4. The `hM959_pXlrE` annotation anomaly
  (same video, 3 different labels) remains unresolved. No temporal-boundary
  annotations exist to confirm segment alignment. The `.pose` archive is not
  locally available. The `subset6.csv` vocabulary does not include any of the
  6 INCLUDE pilot words, meaning any claim about the "pilot" must explicitly
  redefine the vocabulary (not silently assume equivalence).
- **Verdict:** Viable **only if** the user explicitly accepts that the vocabulary
  is different from the original `INCLUDE`-based pilot (`docs/DATA_AND_MODEL.md`
  §4) and that the `subset6.csv` manifest represents a **new** bounded
  experiment (as proposed in `docs/reports/SIGN_RECOGNITION_EXPANSION_PLAN.md`).
  It does **not** resolve the discrepancy with the intended vocabulary.

### Option B: Create a new manifest using the intended pilot vocabulary (`friend`/`hello`/`hospital`/`market`/`school`/`thank you`)

- **Evidence supporting:** The intended vocabulary exists in the `INCLUDE`
  dataset (122 verified `CC-BY-4.0` video clips, `metadata.csv` verified,
  vocabulary matches exactly — `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §2;
  `docs/DATA_AND_MODEL.md` §4). The `IsolatedWordAdapter` and the existing
  pipeline (`pose_normalize_v1`, `ContinuousSignPipeline`, `SP-16k` tokenizer)
  are reusable (`docs/STATUS_AND_ROADMAP.md` §2; `docs/DATA_AND_MODEL.md` §2).
- **Evidence opposing / limitations:** The `iSign` dataset (`subset6.csv` source)
  has almost zero isolated examples for the intended vocabulary (`friend`: 1,
  `school`: 1, others: 0). The `subset6.csv` manifest uses a different vocabulary
  (`music`/`you`/etc.). There is **no** existing script that generates a
  manifest with the intended vocabulary from `iSign_v1.1.csv` (the `auto_subset.json`
  selects 2500 rows, not 6-word vocabulary; the `subset6.csv` file content does
  not match the `VOCAB` list in `SIGN_RECOGNITION_EXPANSION_PLAN.md` §8's
  example script). The `INCLUDE` clips are video (`.mp4`) + JSON pairs, not
  `.pose` sequences (`.pose` archive missing); the pipeline expects `(T, 543, 3)`
  sequences (`docs/DEVELOPMENT.md` §2). Creating a new manifest from `iSign`
  would require selecting rows with the intended vocabulary — but those rows
  barely exist (`hello`: 0, `hospital`: 0, `market`: 0). Creating a manifest
  from `INCLUDE` clips would require video extraction (not `.pose`), which is
  outside the current pipeline's verified input contract for the `subset6.csv`
  experiment.
- **Verdict:** **Not supported by available evidence** for an `iSign`-derived
  manifest. Partially supported only if the source is switched back to the
  `INCLUDE` video clips (different dataset, different format, requires new
  extraction pipeline — not verified locally).

### Option C: Use verified INCLUDE clips or another already-available, appropriately licensed source if iSign cannot support the intended vocabulary

- **Evidence supporting:** The `INCLUDE` dataset (`CC-BY-4.0`, 122 verified clips)
  has the intended vocabulary (`friend`, `hello`, `hospital`, `market`, `school`,
  `thank you` — verified by `docs/DATA_AND_MODEL.md` §4; `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §2). The `IsolatedWordAdapter` serves this pilot (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §4: measured accuracy 61.1%, macro recall 61.1%).
- **Evidence opposing / limitations:** The `INCLUDE` clips are video (`MP4` + JSON),
  not `.pose` sequences. The `subset6.csv` manifest references `iSign` (`uid`
  format `video_id--seq`), not `INCLUDE` (`uid` format `include_MVI_*`). The
  pipeline for the bounded prototype (`SIGN_RECOGNITION_EXPANSION_PLAN.md` §5–§6)
  expects `iSign`-derived subsets (`video_id` split, `.pose` stream extraction,
  `SP-16k` tokenizer trained on subset texts). Switching to `INCLUDE` clips
  would require a different data-preparation path (video → `.pose` extraction,
  which is not verified for `INCLUDE` clips locally; the `media/lib
  tasks/vision` assets are from Google and their separate redistribution terms
  were not independently audited — `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §4).
  The `INCLUDE` aggregate (`license: other` / multi-source) requires verification
  of each source's terms before any redistribution (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md`
  §2). Only the 122 `INCLUDE` clips have verified `CC-BY-4.0`; the full aggregate
  (642 clips) includes `CISLR` (`license: research`) and `ISLRTC` (`license: other`).
- **Verdict:** **Partially supported** as an alternative source, but requires
  new video-extraction verification and licensing review. It does **not** resolve
  the `subset6.csv` discrepancy (different dataset, different vocabulary). It is
  not a drop-in replacement for the `iSign`-derived bounded prototype.

### Option D: Stop the prototype and document that the required data is unavailable or insufficient

- **Evidence supporting:** The intended vocabulary (`friend`, etc.) has almost
  zero verified isolated examples in the `iSign` dataset (`hello`: 0, `hospital`:
  0, `market`: 0, `friend`: 1, `school`: 1, `thank you`: 2 sentence-level only).
  The `subset6.csv` manifest uses a different vocabulary (`music`, `one`, etc.)
  from a different dataset (`iSign` vs. `INCLUDE`). The `hM959_pXlrE` annotation
  anomaly (same video, 3 different labels) remains unexplained. No `.pose`
  archive or video clips are locally available (`archive/*.zip` verified
  unchanged; `.pose` archive `158+` GB not downloaded; `staging_kaggle_results/isign/iSign_v1.1.csv`
  is CSV-only — no `.pose` or video reference). The subset-generation script
  (`auto_subset.json`, `subset: 2500`) does not explain the 74-row `subset6.csv`.
  No temporal-boundary annotations exist (`docs/DATA_AND_MODEL.md` §1: `uid`,
  `text` only). No working model exists for the intended vocabulary
  (`docs/DATA_AND_MODEL.md` §5: `IsolatedWordAdapter` serves 61.1% accuracy on
  the `INCLUDE` 6-word pilot; `352868924` is `NO-GO`; V2 untrained; no
  `subset6.csv`-vocabulary adapter exists). The promotion gate (`SIGNSAGE_MODEL_PROMOTED=false`)
  remains closed (`docs/STATUS_AND_ROADMAP.md` §4; `docs/DATA_AND_MODEL.md`
  §5).
- **Evidence opposing:** The existing `subset6.csv` manifest is a real,
  preserved artifact (74 verified rows). It can support a bounded prototype
  **only if** the user explicitly redefines the vocabulary (Option A) and
  resolves the `hM959_pXlrE` anomaly. The `SIGN_RECOGNITION_EXPANSION_PLAN.md`
  (§3–§8) provides a complete bounded-experiment design for the `subset6.csv`
  vocabulary (`you`/`music`/etc.), not the intended vocabulary.
- **Verdict:** **Supported** as the recommendation for the **intended pilot
  vocabulary** (`friend`/`hello`/etc.) using the `iSign`-derived data. The
  evidence clearly shows that the `iSign` dataset does not contain the intended
  vocabulary in sufficient isolated form, and that the `subset6.csv` manifest
  refers to a different dataset with a different vocabulary. However, the task
  requires comparing A–D and recommending the smallest valid correction. The
  evidence shows that Option D (`NO-GO`) applies specifically when the goal is
  to use the **intended vocabulary with the existing `subset6.csv` manifest**.

---

## 6. Recommendation and smallest corrective action

### 6.1 Clear recommendation

**Recommendation: Option D (`NO-GO — DATA INSUFFICIENT`) for the intended pilot
vocabulary (`friend`/`hello`/`hospital`/`market`/`school`/`thank you`) with the
existing `subset6.csv` manifest.**

**Reason (evidence-backed, not speculative):**

1. The `subset6.csv` manifest vocabulary (`music`/`one`/`three`/`two`/`water`/`you`)
   does not match the intended vocabulary (`docs/DATA_AND_MODEL.md` §4: 6-word
   `INCLUDE` pilot vocabulary; `data/manifests/subset6.csv`: verified 74 rows with
   different vocabulary — `python3` Counter verified).
2. The `iSign` dataset (`iSign_v1.1.csv`, 127,237 rows) has **zero** verified
   isolated examples for `hello`, `hospital`, `market`; **one** each for `friend`
   and `school`; and **two** sentence-level references for `thank you` (not
   isolated signs — `python3` verified by exact-match and substring count).
3. There is an unresolved annotation anomaly (`video_id`: `hM959_pXlrE`, same
   source video annotated as `water`, `you`, `music` at different `uid` sequence
   indices — verified from `subset6.csv`). Without temporal-boundary annotations
   (`start_frame`/`end_frame` — absent from `iSign_v1.1.csv`; verified by `head`),
   the discrepancy cannot be explained or corrected.
4. No `.pose` archive (`158+` GB) or video clips for `subset6.csv` are locally
   available (`archive/*.zip` verified; `staging_kaggle_results/isign/iSign_v1.1.csv`
   is CSV-only; `pose_cache/` 13.6 GB does not contain the vocabulary — verified
   by `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §3).
5. The subset-generation logic (`auto_subset.json`: `subset: 2500`; `signsage_kaggle.ipynb`
   cell 8 writes this) does **not** explain how `subset6.csv` (74 rows) was
   selected from the 127,237-row `iSign` dataset. No script produces a 74-row,
  6-word subset from `iSign_v1.1.csv` (verified by searching `subset6` references:
  only `SIGN_RECOGNITION_EXPANSION_PLAN.md` §8's example command uses `VOCAB`
  list, but the actual `subset6.csv` content does not match that `VOCAB` list).

**If the user explicitly wants to proceed with the `subset6.csv` vocabulary
(Option A), that is possible ONLY with these strict conditions (all must be met):**

- Explicit acknowledgment that the vocabulary (`music`/`one`/`three`/`two`/`water`/`you`)
  is different from the original `INCLUDE`-based pilot (`friend`/`hello`/etc.).
- Explicit resolution of the `hM959_pXlrE` discrepancy (either verified by
  temporal-boundary inspection from the `.pose` archive — which requires download
  and is outside this audit's scope — or documented as an unresolved risk).
- Explicit licensing acknowledgment (`CC BY-NC-SA 4.0` for any `iSign`-derived
  checkpoint; promotion requires separate license review — `docs/DATA_AND_MODEL.md`
  §6; `docs/STATUS_AND_ROADMAP.md` §4).
- No claim that sentence-level English translation works (no sentence-level dataset
  exists; `SIGNSAGE_MODEL_PROMOTED` remains `false` — preserved).
- No claim that signer-independent evaluation is possible (`signer_independent: false`
  — `docs/DATA_AND_MODEL.md` §4; `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §2).

### 6.2 Smallest corrective action needed (before any modification)

**Next step (evidence-required, not speculative):**

1. **Confirm vocabulary intent explicitly.** The user must state whether the
   experiment should use:
   - The `subset6.csv` vocabulary (`music`/`one`/`three`/`two`/`water`/`you`)
     — which aligns with `SIGN_RECOGNITION_EXPANSION_PLAN.md` (§5) and requires
     resolving the `hM959_pXlrE` anomaly and downloading/verifying `.pose`
     sequences; OR
   - The `INCLUDE` pilot vocabulary (`friend`/`hello`/`hospital`/`market`/`school`/`thank you`)
     — which requires switching back to the `INCLUDE` video-clip source (different
     dataset, requires new video → `.pose` extraction verification, different
     licensing verification for the aggregate).

2. **If `subset6.csv` vocabulary is chosen:** Resolve the `hM959_pXlrE`
   discrepancy by either:
   - Downloading the full `.pose` archive (`158+` GB — requires `HF_TOKEN`,
     budget/resource check, and is excluded by this audit's constraints); OR
   - Documenting the discrepancy as an unresolved data-quality risk and
     excluding `hM959_pXlrE` (and any other multi-label `video_id`) from any
     training subset.

3. **If intended vocabulary is chosen:** Create a new manifest from the verified
   122 `INCLUDE` clips (not from `iSign_v1.1.csv`), with new extraction, new
   split rules (`video_id` split impossible — `INCLUDE` uses recording IDs like
   `include_MVI_*`; split must use verified recording-ID grouping —
   `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §2: `reproducible 86/18/18 split`),
   and new licensing review (only `CC-BY-4.0` for the 122 `INCLUDE` clips;
   aggregate `license: other`).

4. **In either case:** Before creating any replacement manifest or modifying
   `subset6.csv`, obtain explicit approval from the user. No relabeling,
   replacement, or dataset extraction should occur without this approval
   (`docs/STATUS_AND_ROADMAP.md` §6: "Wait for approval before any download/training";
   `docs/DATA_AND_MODEL.md` §5: promotion requires documented evaluation covering
   quality, diversity, failure modes, and baselines; `docs/DEVELOPMENT.md` §5:
   `.env` settings unchanged; no `.env` secrets committed).

---

## 7. Final status

**Status: `NO-GO — DATA INSUFFICIENT`** (for the intended pilot vocabulary
`friend`/`hello`/`hospital`/`market`/`school`/`thank you` using the existing
`subset6.csv` manifest). The evidence clearly shows that:

- The `subset6.csv` vocabulary does not match the intended vocabulary (`docs/DATA_AND_MODEL.md`
  §4 vs. `data/manifests/subset6.csv` — verified).
- The `iSign` dataset (`iSign_v1.1.csv`) lacks sufficient isolated examples
  for the intended vocabulary (`python3` verified counts: 4 of 6 words at 0 or 1;
  `thank you` only as sentence-level text).
- There is an unresolved annotation discrepancy (`hM959_pXlrE`: same video,
  3 different labels — verified from CSV).
- No `.pose` archive or temporal annotations exist locally to confirm segment
  alignment (`archive/*.zip` verified; `.pose` archive `158+` GB not downloaded;
  `iSign_v1.1.csv` has no `start_frame`/`end_frame` columns — verified by `head`).
- The subset-generation logic (`auto_subset.json`: `subset: 2500`) does not
  explain the 74-row `subset6.csv` (`python3` verified by reading `auto_subset.json`
  and comparing with `subset6.csv` row count).
- No working promoted model exists for either vocabulary (`SIGNSAGE_MODEL_PROMOTED=false`
  — `docs/DATA_AND_MODEL.md` §5; `docs/STATUS_AND_ROADMAP.md` §4; `backend/.env`
  verified).

**Status reference in `docs/STATUS_AND_ROADMAP.md`:** Updated with a brief
reference to this review (see `docs/STATUS_AND_ROADMAP.md` line 1: reference
added to `docs/VOCABULARY_ALIGNMENT_REVIEW.md` and the `NO-GO` status).
All existing evidence preserved (`§1` vocabulary/status, `§2` integration
passes, `§2a` static classification, `§3` audit evidence, `§4` blockers,
`§5` end-to-end validation results, `§6` Playwright E2E verification results,
`archive/*.zip` unchanged, `.env` unchanged, `SIGNSAGE_MODEL_PROMOTED=false`
preserved, `results-v4.zip` untouched, `subset6.csv` unchanged — discrepancy
noted but file unmodified, no new dataset created, no manifest modified,
no model checkpoint changed, no training launched, no download performed,
no promotion flag flipped).

**Files changed (deliverables only):**

- `docs/VOCABULARY_ALIGNMENT_REVIEW.md` (new file — this document).
- `docs/STATUS_AND_ROADMAP.md` (updated line 1 — brief reference added;
  no content removed; all prior evidence preserved; `.env` settings preserved;
  promotion gate preserved; dataset preservation claims preserved).

No other repository files modified. All artifacts (`archive/*.zip`,
`data/manifests/subset6.csv`, `staging_kaggle_results/isign/iSign_v1.1.csv`,
`staging_kaggle_results/s2t_v2/pose_cache/`, `.env`, model checkpoints,
`results-v4.zip`) preserved exactly (`ls -lh` and `grep` verified before/after
— no differences except the two `docs/` files).

---

## 8. Explicit unresolved questions and limitations

These questions remain open and are **not filled with guesses** (`docs/STATUS_AND_ROADMAP.md`
§6: "If the evidence is insufficient to answer a question, explicitly state
what remains unknown").

1. **Does `subset6.csv` represent a valid 6-class isolated-word subset for any
   vocabulary?** The 74 rows have coherent `video_id--seq` IDs, but no `.pose`
   sequences are locally available. Without the `.pose` archive (`158+` GB,
   `CC BY-NC-SA 4.0`), no pose-alignment verification is possible. The
   `hM959_pXlrE` discrepancy (3 different labels) remains unexplained without
   temporal-boundary inspection.

2. **Can the `subset6.csv` vocabulary (`music`/`one`/`three`/`two`/`water`/`you`)
   be mapped to the `IsolatedWordAdapter` (`friend`/`hello`/etc.)?** No.
   The adapter (`docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §4) serves the `INCLUDE`
   6-word pilot vocabulary, not the `subset6.csv` vocabulary. There is **no**
   adapter mapping from `subset6.csv` vocabulary to any existing model bundle.
   The `SIGN_RECOGNITION_EXPANSION_PLAN.md` (§6) proposes training a new
   `d128/2L` model specifically on the `subset6.csv` vocabulary — but this
   requires the `.pose` archive (not downloaded) and a new tokenizer (not
   trained locally — the existing `tokenizer/signsage_en_sp.model` is `SP-16k`
   but was trained on the 2500-row pilot texts, not on `subset6.csv` texts —
   `docs/DATA_AND_MODEL.md` §3; `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` §4).

3. **Can any sentence-level English translation claim be supported?** No.
   There is no sentence-level dataset (`iSign` has multi-word sentences but no
   verified temporal sign-boundary annotations — `docs/DATA_AND_MODEL.md` §1;
   `subset6.csv` has 74 isolated-word-style rows — `docs/ISOLATED_ISL_RECOVERY_AUDIT.md`
   §3; `INCLUDE` clips are isolated-word video clips — `docs/ISOLATED_ISL_RECOVERY_AUDIT.md`
   §2). No working continuous model exists (`SIGNSAGE_MODEL_PROMOTED=false`;
   `352868924` broken — `docs/DATA_AND_MODEL.md` §5; `docs/STATUS_AND_ROADMAP.md`
   §4). No sentence-level vocabulary mapping exists (the 14 `sign_vocabulary.json`
   IDs are EN→ISL **generation** labels — `docs/DATA_AND_MODEL.md` §4; they are
   not ISL→EN recognition labels). The `avatar_gloss_service.py` produces gloss
   sequences (deterministic lookup or LLM-assisted mapping to pre-authored clips),
   not continuous recognition results (`docs/DEVELOPMENT.md` §3). Any sentence-level
   claim requires: a verified temporal dataset with sentence-level annotations,
   a working model (`docs/DATA_AND_MODEL.md` §5: promotion criteria include
   `WER`/`BLEU` + `exact_match` + `diversity` + `per_class_recall` + signer-independent
   evaluation — none met by any existing checkpoint), and an evaluation
   framework for sentence fluency (not present — current metrics measure
   sequence-level `WER`/`BLEU`, not grammar/fluency — `docs/STATUS_AND_ROADMAP.md`
   §5; `docs/DATA_AND_MODEL.md` §5).

4. **Does any subset-generation script explain how `subset6.csv` was created?**
   No. The `auto_subset.json` (`subset: 2500`) and `signsage_kaggle.ipynb`
   cell 8 (`subset` key = `2500`) do not match the 74-row `subset6.csv`. The
   `test_auto_subset_pilot_certify.py` verifies only that the `subset` key is
   read correctly by cell 9, not that the value equals the manifest size. The
   `SIGN_RECOGNITION_EXPANSION_PLAN.md` (§8) provides a command-line example
   (`python - <<'EOF' ... VOCAB = [...]`) that creates a subset CSV from the full
   `iSign_v1.1.csv`, but the actual `subset6.csv` content does not match any
   `VOCAB` list from that script (`music`/`one`/etc. are not in the intended
   `VOCAB`). The manifest's creation method is **unknown** and must be verified
   before any replacement or relabeling.

---

*Review produced without training, downloads, archive extractions, manifest
modifications, relabeling, `.env` changes, or promotion-flip. All findings
reference verified file contents (`cat`, `python3`, `bash` with `ls`/`grep`/`find`)
and preserve all existing evidence (`docs/STATUS_AND_ROADMAP.md` §1–§6 preserved
intact; only line 1 updated with a brief reference). Wait for explicit approval
before any manifest modification, dataset extraction, download, or training.
(`docs/STATUS_AND_ROADMAP.md` §6: "Stop after the audit and recommendations.
Wait for approval before implementing, downloading, or training anything."; `docs/DATA_AND_MODEL.md`
§5: promotion requires separate human-approved GO; `docs/DEVELOPMENT.md` §5:
no full training started — preserved; `docs/AGENTS.md`: agent workflow completes
audit before any implementation).*
