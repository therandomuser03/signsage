# SignSage — ISL-to-English Sentence Dataset Readiness Report

**Status:** Candidate identified (CONDITIONAL GO). No dataset fully satisfies all criteria; see blockers below.
**Authored:** from repository evidence + verified external sources. No model retrained, no weights modified, no GPU training launched, no massive download performed.
**SIGNSAGE_MODEL_PROMOTED:** kept `false` (unchanged). No commit/push/deploy performed.

---

## 1. Objective and headline finding

SignSage intends **Indian Sign Language (ISL) → English sentence translation** (continuous pose→text, SignPose2Text). The repository's existing sentence-model run (checkpoint 352868924) is a documented **NO-GO** (WER 1.0046, exact match 0.0, output diversity 0.99%, top-pred 44.8%) and was trained on *synthetic* iSign text→pose data, not real ISL video. The task is therefore to find a *legally usable, technically accessible* real ISL sentence-level dataset.

**Best candidate: iSign v1.1 (Hugging Face `Exploration-Lab/iSign`).** It is the only candidate in evidence that is (a) genuinely ISL, (b) sentence/phrase-level, (c) pose/video-grounded, and (d) large enough to train a translation model. It is **CONDITIONAL GO**, not unconditional GO, because it lacks signer metadata (signer-independent evaluation cannot be established from the CSV) and is CC BY-NC-SA 4.0 (research use; not commercial) and gated on Hugging Face.

No other inspected candidate is sentence-level ISL; all others are isolated-word, alphabet/fingerspelling, caption-only, or non-ISL, and are presented as NO-GO for this task. They are **not interchangeable** with iSign.

---

## 2. Evidence sources and what was actually accessed

Distinguishing verified (accessed) from unverified (not accessed):

| Source | Accessed? | What it establishes |
|---|---|---|
| HF dataset card `Exploration-Lab/iSign` (web, HTTP 200) | YES | Title "iSign: A Benchmark for Indian Sign Language Processing"; tags `indian sign language`, `machine translation`, `sign language translation`, `task_categories:translation`; license `cc-by-nc-sa-4.0`; gated `auto`; size `100K<n<1M`; `numRows: 129322`; arXiv `2407.05404`; "free for research use but not for commercial purposes"; formats `csv`, modality `text`. |
| arXiv `2407.05404` abstract (web, HTTP 200) | YES | Paper title "iSign: A Benchmark for Indian Sign Language Processing"; "more than 118K video-sentence/phrase pairs"; "largest sign language dataset available for ISL"; tasks incl. `SignVideo2Text`, `SignPose2Text`, `Text2Pose`. |
| `kaggle_input_v4/isign/iSign_v1.1.csv` header + first rows (local) | YES | Columns: exactly `uid,text` (2 columns). Sample text = English sentences ("Make it shorter.", "One day, Akbar drew a line on the floor and ordered.") plus some one-word/caption rows ("Rub.", "Puzzle", "Page 111"). **No `signer_id`/`signer` column.** |
| `staging_kaggle_results/s2t_v2/dataset_audit.json` (local) | YES | raw 127,237 rows; usable 123,779 after cleaning; 3,797 unique videos; 120,221 unique targets; mean 10.86 / median 11 / p90 18 / max 66 words per target; 2,663 one-word rows (~2.1%); split train 98,485 / val 13,014 / test 12,280 (videos 3,037/380/380); **`signer_column: null`**; signer-independence "assumed from video-level split only". |
| `research/manifests/models/signsage-s2t_v352868924.yaml` (local) | YES | Confirms dataset source `iSign (Hugging Face)`, verified NO-GO metrics, `promoted_to_backend: false`. |
| `docs/DATA_AND_MODEL.md` (local) | YES | iSign = Joshi et al. ACL Findings 2024, CC BY-NC-SA 4.0; columns `uid,text`; 127,237 rows; 543-landmark pose contract; signer-independence NOT verified ("Only a small subset was certified-signer-verified"); license gate documented. |
| `research/training/READINESS.md` (local) | YES | Training resource estimates, cache modes, split policy, signer-independence requirement (`require_signer_independent`), diversity gate. |
| `research/experiments/FINAL_AUDIT_REPORT.md` (local) | YES | Archive/forensic findings; checkpoint resumability unverified; tokenizer/pose-cache fingerprints. |
| `research/image_classification/AUDIT.md` (local) | YES | RealSign (alphabet, CC0) and Real-life Words (20 words, Mendeley) — **isolated-word/alphabet, not sentence-level**. |
| `research/training/GESTURETALK_AUDIT.md` (local) | YES | GestureTalk = isolated alphabet+word recognition, NOT continuous sentence translation; signer-independence OPEN for iSign. |
| `docs/EXTERNAL_EVALUATION.md` (local) | YES | External `t5-small-motion` on 1,887 held-out iSign samples → exact match 0/1887 → NO-GO. |
| HF API search `indian sign language` (web, HTTP 200) | YES | ~10 small ISL uploads (Hemg/, akritRihal/, ajeet-123/, LIGHTscrn/, KRISH09bha/, silentone0725/, SandhyaPitchika/, Vignesh3816/, suneetpaul/): **no license, no paper, no size/rows, no sentence-level tags, no signer metadata**. Not credible as primary candidates without substantial new verification. |
| iSign full pose archive (~158 GiB, 4-part ZIP64), arXiv full paper, .pose payloads | NOT accessed | Gated on HF (requires HF_TOKEN + user agreement); too large to download without approval; not needed for this readiness determination. |
| iSign signer metadata | NOT available | Confirmed absent from CSV (`uid,text` only). Whether the paper or authors provide signer IDs was not verified. |

---

## 3. Candidate evaluation

### Candidate 1 — iSign v1.1 (`Exploration-Lab/iSign`) — CONDITIONAL GO

| Criterion | Finding | Evidence / source |
|---|---|---|
| Actually ISL? | **YES** | Paper title + abstract "iSign: A Benchmark for Indian Sign Language Processing"; HF tag `indian sign language`; arXiv 2407.05404. Verified via HF dataset card + arXiv abstract. |
| Video / temporally ordered pose data? | **YES** | "118K video-sentence/phrase pairs" (arXiv abstract); `.pose` files, one per UID/row, 543 landmarks/frame (POSE 33 + FACE 468 + LEFT_HAND 21 + RIGHT_HAND 21 = 1629 floats/frame). Verified via DATA_AND_MODEL.md + notebook cell 1a. |
| Aligned with English sentence? | **YES** | Columns `uid,text`; text = English sentences (mean 10.86, median 11, p90 18 words); arXiv "video-sentence/phrase pairs"; SignPose2Text task. Verified via CSV sample + dataset_audit.json + arXiv. |
| Size / usable samples | **Large** | 127,237 raw rows; 123,779 usable after cleaning; 3,797 unique videos; 120,221 unique targets; split train 98,485 / val 13,014 / test 12,280. Verified via dataset_audit.json. |
| Annotation quality | **Mostly good; ~1% junk** | Junk filtered: empty 391, numeric 559, single-letter 177, bracketed caption 31, punctuation 9 (total 1,167 removed). 2,663 one-word rows (~2.1%) retained as legitimate one-word targets. Some caption-ish targets ("blank", "dash", "Page 111") appear in top targets and should be filtered. Verified via dataset_audit.json. |
| License | **CC BY-NC-SA 4.0** | HF card `license:cc-by-nc-sa-4.0`; dataset card "free for research use but not for commercial purposes". SignSage README documents the NC/SA license gate. Verified via HF card + DATA_AND_MODEL.md. |
| Access restrictions | **Gated** | `gated: auto` on HF; requires HF_TOKEN and user agreement (username+email shared). Not openly downloadable without authentication. Verified via HF card. |
| Signer IDs for signer-independent split? | **NO — blocker** | CSV has exactly 2 columns (`uid,text`); `signer_column: null`; dataset_audit signer_note: "signer-independence assumed from video-level split only; confirm on the training host." No signer metadata accessible. Verified via CSV header + dataset_audit.json. |
| Fits existing pipeline? | **YES** | Pipeline (543-landmark contract, `pose_normalize_v1`, SP-16k tokenizer, ≤96 tokens, video_id split) was built for iSign; pose/tokenizer cache fingerprints verified. Verified via DATA_AND_MODEL.md, audit_dataset.py, FINAL_AUDIT_REPORT. |

**Decision: CONDITIONAL GO.** iSign is the only sentence-level ISL dataset with credible evidence. It can support SignSage's intended task *for research*. Conditions before committing to full training:
1. Resolve the signer-metadata gap (signer-independent evaluation cannot be established from `uid,text` alone).
2. Confirm CC BY-NC-SA 4.0 permits SignSage's intended use (research prototype; commercial deployment needs separate review/clearance).
3. Confirm gated HF access is obtainable (HF_TOKEN + user agreement).

---

### Candidate 2 — RealSign-Indian-Sign-Language-Dataset (alphabet, CC0) — NO-GO for sentence translation

- ISL fingerspelled alphabet A–Z, ~26,000 images (Train 18,198 / Val 2,579 / Test 5,200), CC0, separate Train/Val/Test splits.
- **Not sentence-level:** isolated alphabet letters; no English sentence targets; no video/pose temporal data (static images); no signer per image.
- Suitable only for an isolated alphabet classifier, which SignSage explicitly does not want for recognition (no static-sign classifier in the recognition flow).
- Source: `research/image_classification/AUDIT.md` (verified).

### Candidate 3 — Indian Sign Language — Real-life Words (Mendeley) — NO-GO for sentence translation

- 20 isolated word classes (afraid, become, bad, …, you), ~18,000 images, ~900/word, signer subdirectories (`User_1`…`User_6+`) enabling signer-independent splits.
- **Not sentence-level:** isolated words, static images, no English sentence targets, no video/pose.
- License to be confirmed from Mendeley page (no LICENSE inside zip).
- Source: `research/image_classification/AUDIT.md` (verified).

### Candidate 4 — `vidit031/isl-isolated-40words` — NO-GO for sentence translation

- Isolated-word ISL pilot (40 words), used for the 6-word supported recognition vocabulary (friend, hello, hospital, market, school, thank you). Test accuracy 61.1%, below the 80/80/60 promotion bar, not signer-independent.
- **Not sentence-level.** Separate reference/baseline dataset; explicitly not merged into Sign→English training (notebook policy).
- Source: `docs/DATA_AND_MODEL.md`, `research/training/GESTURETALK_AUDIT.md` (verified).

### Candidate 5 — External `t5-small-motion` (`manavdhamecha77/iSign-t5-pose-to-text`) — NO-GO (quality)

- Evaluated on 1,887 real held-out iSign samples: exact match 0/1887; top prediction "isn't that amazing!" 401/1887 (21.25%); 298 unique (0.158).
- Not a dataset per se, but an external model on iSign; confirms iSign is the underlying data but the external model is NO-GO.
- Source: `docs/EXTERNAL_EVALUATION.md` (verified).

### Candidate 6 — Other HF "Indian Sign Language" uploads — NO-GO (insufficient evidence)

HF API search returned ~10 small uploads (Hemg/, akritRihal/, ajeet-123/, LIGHTscrn/, KRISH09bha/, silentone0725/, SandhyaPitchika/, Vignesh3816/, suneetpaul/). None have a license, research paper, row count, sentence-level tags, or signer metadata. Sizes 3–1,162 downloads. **Do not present as interchangeable with iSign;** each would require a fresh, full documentation/license/alignment investigation before consideration. Source: HF API search (verified listing; contents not accessed).

---

## 4. Sentence-level vs non-sentence-level distinction (enforced)

| Category | Candidates | Sentence-level English target? | Video/pose temporal data? | ISL? | Verdict for SignSage |
|---|---|---|---|---|---|
| Sentence-level ISL translation | iSign | YES | YES | YES | CONDITIONAL GO |
| Isolated alphabet/fingerspelling | RealSign | NO | NO (static images) | YES | NO-GO |
| Isolated words (static) | Real-life Words, isl-isolated-40words | NO | NO (static images) | YES | NO-GO |
| Caption-only / single-word | (junk rows within iSign: "blank","dash","Page 111") | Partial/no | YES | YES | Filtered out in cleaning |
| Other sign language / non-ISL | (none credible found) | — | — | — | NO-GO |
| External translation model | t5-small-motion | — | — | — | NO-GO (quality) |

iSign's "blank"/"dash"/"Page N" rows are caption/artifact noise, removed by the audit's junk filter; they are **not** silently relabeled as translations.

---

## 5. Storage and compute estimates (clearly labeled estimates)

These are estimates drawn from local instrumented logs (READINESS.md, FINAL_AUDIT_REPORT, dataset_audit.json); none were re-measured in this session.

| Item | Estimate | Basis |
|---|---|---|
| iSign v1.1 CSV | ~9.8 MB | `iSign_v1.1.csv` = 9,808,778 bytes (local copy) |
| iSign pose archive (remote, 4-part ZIP64) | ~158+ GiB | streamed, never fully downloaded (READINESS.md) |
| Full pose cache (fp32, 123,779 × 160 × 1629) | ~120 GiB fp32 / ~60 GiB fp16 | READINESS.md; deliberately never cached |
| Subset cache (s2t_small6, 2,500/split = 7,500 uids) | ~9.1 GiB heuristic | `auto_subset.json` cache_gib_heuristic |
| Full-cache prepare time | ~29 h sequential (1.2 samples/s); ~4 h with 8 jobs | READINESS.md measured/local rates; does not fit one Kaggle session |
| Training per epoch (2×T4, batch 64, AMP) | ~3.5–4 h/epoch (v2, ~3× params) | READINESS.md |
| Full run (6–12 epochs) | ~2–3 Kaggle sessions | READINESS.md |
| Bounded diagnostic pilot (2,500-row subset) | ~4.99 h projected | `auto_subset.json` projected_total_hours |
| Tokenizer (SP-16k) | ~539 KB model + vocab | verified fingerprint `2d88a5de48318cb9` |

**Key estimate caveat:** full-cache prepare is the largest storage/time consumer; the recommended path is a deterministic subset pilot (2,500 rows/split) before any full-dataset training.

---

## 6. GO / CONDITIONAL GO / NO-GO decisions

| Candidate | Decision | Rationale (one line) |
|---|---|---|
| iSign v1.1 (`Exploration-Lab/iSign`) | **CONDITIONAL GO** | Only sentence-level ISL pose dataset with credible evidence; blocked by missing signer IDs, CC BY-NC-SA 4.0 (non-commercial), and gated HF access. |
| RealSign (alphabet) | **NO-GO** | Isolated alphabet, static images, no sentences; wrong task. |
| Real-life Words | **NO-GO** | Isolated words, static images, no sentences; wrong task. |
| isl-isolated-40words | **NO-GO** | Isolated words; below quality bar; not signer-independent. |
| t5-small-motion (external model) | **NO-GO** | 0/1887 exact match on iSign; quality failure, not a dataset. |
| Other HF ISL uploads | **NO-GO** | No license/paper/rows/signer/sentence evidence; not credible without new investigation. |

---

## 7. Precise blocker (if a fully suitable dataset is required)

**No inspected dataset fully satisfies all of SignSage's criteria.** The precise blocker for the best candidate (iSign):

> **Signer-independent evaluation cannot be established.** The iSign CSV contains exactly two columns (`uid`, `text`) and no signer identifier. `dataset_audit.json` records `signer_column: null` and notes signer-independence is "assumed from video-level split only." A video-level split does **not** guarantee signer independence (one signer can appear across multiple videos), which SignSage has explicitly flagged as an open item (GESTURETALK_AUDIT §9). Without signer IDs, held-out signer-independent splits—and thus a defensible claim that the model generalizes across signers—cannot be constructed from this dataset's public metadata.

Secondary blocker: **CC BY-NC-SA 4.0** permits research use but explicitly not commercial purposes; any public/commercial checkpoint deployment needs a separate license review (documented in DATA_AND_MODEL.md §6 and README).

These are **data limitations, not model limitations**—they cannot be fixed by retraining or by relabeling captions as translations.

---

## 8. Minimum validation experiment (before full training)

Do **not** launch full training or a Kaggle GPU session yet. Run this smallest credible validation first:

1. **Confirm signer-metadata ceiling definitively.**
   - Re-inspect the iSign CSV (already local: `kaggle_input_v4/isign/iSign_v1.1.csv`) to confirm no signer column (done: `uid,text` only).
   - Check the arXiv full paper / iSign website / dataset authors for any published signer-ID split or signer-verified subset (DATA_AND_MODEL.md notes "Only a small subset was certified-signer-verified" but this was not verified against the paper).
   - Decision point: if no signer metadata exists anywhere, accept video-level split as the ceiling and document signer-independence as an unresolved risk; do not claim signer-independent evaluation.

2. **Bounded alignment/diagnostics pilot (no full training).**
   - Use the existing verified s2t_small6 subset (2,500 rows/split, tokenizer fingerprint `2d88a5de48318cb9`, pose cache) with fresh weights, `--embed-init scaled`, `--stages overfit,train`, no `--resume`, external 24-minute timeout (per FINAL_AUDIT_REPORT §F and `research/experiments/kaggle_diagnostic.py`).
   - Success signal: overfit gate passes and diversity stays ≥ 0.05 floor — i.e., real pose→sentence alignment is learnable on a small held-out split. Collapse would indicate a data/alignment problem, not just an initialization problem.

3. **License/access confirmation.**
   - Obtain HF gated access (HF_TOKEN + user agreement) to confirm downloadability.
   - Record the CC BY-NC-SA 4.0 use determination (research vs commercial) before any bundle deployment.

Only after (1)–(3) succeed should full-dataset training be scheduled.

---

## 9. Files changed / repository impact

- **New file:** `research/reports/signsage_dataset_readiness.md` (this report). No existing files modified.
- **No changes to:** model weights, `SIGNSAGE_MODEL_PROMOTED` (kept `false`), frontend, backend, training scripts, notebooks, or any dataset. No commits, pushes, or deploys. No large downloads performed.

---

## 10. Exact next action required from you

**Approve (or reject) proceeding to the minimum validation experiment in §8**, specifically:

1. Confirm I may verify signer-metadata absence against the iSign paper/website (read-only web lookups only; no download).
2. Confirm I may run the bounded diagnostic pilot on the existing local s2t_small6 subset (CPU/local or a Kaggle GPU session you approve; no full training, no `--resume`, external timeout).
3. Confirm whether SignSage's intended deployment is research-only (CC BY-NC-SA 4.0-compatible) or commercial (requires separate license clearance before any checkpoint is promoted).

Until you answer, I will not launch training, download the gated pose archive, modify any code, or change `SIGNSAGE_MODEL_PROMOTED`.

---

## Sources checked (verified facts marked)

- HF dataset card `https://huggingface.co/datasets/Exploration-Lab/iSign` — **verified** (title, ISL tag, license CC BY-NC-SA 4.0, gated, 129,322 rows, arXiv 2407.05404).
- arXiv `https://arxiv.org/abs/2407.05404` — **verified** abstract (118K video-sentence/phrase pairs; largest ISL dataset; SignPose2Text task).
- `kaggle_input_v4/isign/iSign_v1.1.csv` — **verified** (columns `uid,text`; sample English sentences; no signer column).
- `staging_kaggle_results/s2t_v2/dataset_audit.json` — **verified** (123,779 usable; 3,797 videos; mean 10.86 words; `signer_column: null`).
- `docs/DATA_AND_MODEL.md`, `research/training/READINESS.md`, `research/experiments/FINAL_AUDIT_REPORT.md`, `research/manifests/models/signsage-s2t_v352868924.yaml`, `research/image_classification/AUDIT.md`, `research/training/GESTURETALK_AUDIT.md`, `docs/EXTERNAL_EVALUATION.md` — **verified** local documents.
- HF API search `indian sign language` — **verified listing** (small unlicensed uploads); contents not accessed.
- iSign pose archive, arXiv full paper, .pose payloads, signer metadata — **NOT accessed** (gated/too large; claims about them are not made).
