# Isolated ISL Recovery Audit

Initial audit date: 2026-10-26. The initial pass did not download model or
video artifacts. See the execution follow-up below for the bounded pilot.

## Decision

Do not integrate Sanket into the current backend. Its Hugging Face card
describes a video + 195-feature pose dual-stream model with TimeSformer-base
and T5-base. The current application sends 543 MediaPipe landmarks per frame
and its transformer expects the SignSage `pose_normalize_v1` representation.
No compatible Sanket inference source was located at the GitHub URL in its
model-card example (the URL returned 404). Hugging Face lists two checkpoint
files and about 5.1 GB total repository storage; the card reports training on
an A10G with 24 GB VRAM. Weight size, serving memory, and inference latency
were not measured. Downloading those files without source, exact preprocessing,
or measured compatibility would not be a bounded integration.

The bounded alternative is a trained, isolated-word recognizer from
`vidit031/isl-isolated-40words`, not continuous translation. The current Hub
card reports 642 H.264 clips, 40 glosses, and about 30 FPS. Its repository
contains `metadata.csv` (230,543 bytes) and MP4/JSON pairs. Metadata schema
includes word, source dataset, original label, path, signer, split, duration,
license, source URL, checksums, and review status. The inspected CISLR sample
has blank split, signer value `gesture` (a category, not an identifiable
signer), and license `research (see HF card)`. Therefore signer-independent
evaluation and a redistribution license cannot be assumed from this aggregate.

The aggregate declares `license: other` / `multi-source-aggregate`, not one
blanket license. Its card reports 405 ISL500, 143 INCLUDE, 81 CISLR, and 13
ISLRTC dictionary clips. It instructs users to respect each source license;
the CISLR row inspected does not state a usable license. Do not redistribute
or publish weights trained on a source until its terms are verified. See
https://huggingface.co/datasets/vidit031/isl-isolated-40words and its
`metadata.csv`.

## Existing backend and blocking evidence

- The live input contract is frontend MediaPipe Holistic landmarks:
  `(T, 543, 3)`, ordered pose, face, left hand, right hand. See
  `backend/app/ml/preprocessing/landmarks.py` and
  `frontend/src/features/translate/landmarks.ts`.
- The existing T5-style SignSage adapter is a continuous pose-to-English
  model, not a 40-class recognizer. The registry caches its loaded adapter
  once per process and keeps the promotion gate closed by default.
- The recorded SignSage checkpoint is NO-GO: WER 1.0046, output diversity
  0.0099, top output share 0.4484. Its documented bottleneck is a large remote
  pose cache (about 143 GiB for the full usable corpus) and a collapsed,
  under-trained model, not a missing way to start FastAPI.
- An existing five-shape heuristic is not a trained ISL model and is not
  evidence that the 40-word corpus can be recognized. Do not promote it as
  the requested feature or join its per-frame guesses into sentences.

## Initial audit limitations

At the initial audit, the MP4 corpus, class counts, model weights, and a
held-out prediction had not been inspected; Bubblewrap could not start because
`slirp4netns` was missing. The follow-up below used the normal VS Code task
terminal, so those data and execution findings are updated there. Sanket
remains unintegrated; the original 40-class aggregate was not trained.

The trained pilot bundle is outside this checkout. Promotion remains closed
because its measured accuracy is low and signer-independent evaluation is not
possible from supplied metadata.

## Execution follow-up

The normal VS Code task terminal worked despite the Bubblewrap shell being
unavailable because `slirp4netns` was absent. No system settings or packages
were changed. Dataset inspection, bounded training, tests, and API checks below
ran in that terminal.

Pinned dataset revision:
`255ce7fd10da5d05bc30d59668ba256e5f02531c`. Its metadata contains 642 clips
and 40 labels. The source mix is 405 ISL500, 143 INCLUDE, 81 CISLR, and 13
ISLRTC. Only the 122 accepted INCLUDE clips were downloaded and used; all
declare CC-BY-4.0 and point to AI4Bharat INCLUDE / Zenodo record 4010759. The
21 `okay` clips marked for manual review were excluded. No iSign or ISL500
videos were downloaded.

The 40 metadata label counts range from 1 (`home`) to 39 (`thank you`), so
the aggregate is strongly imbalanced. The accepted INCLUDE-only training
subset has friend 20, hello 21, hospital 20, market 20, school 20, and thank
you 21. All 642 metadata rows were reviewed for label/source/license counts;
only the selected 122 INCLUDE clips were downloaded and decoded.

INCLUDE metadata has blank split values and `signer` values like
`include_MVI_3290`, which identify a recording but do not verify a human
signer. One recording ID appears twice. The reproducible 86/18/18 split groups
matching recording IDs together and has no recording-ID overlap across
partitions, but cannot establish signer-independent generalization.

Added a separate six-word pose-and-hand Conv1D pilot. Training and live
inference both select pose landmarks 0–32 and hand landmarks 501–542 from the
existing `(T, 543, 3)` API representation, root-center and scale them, then
resample to 64 frames × 225 features. Face points are ignored. Training uses
the same MediaPipe Tasks pose/hand assets referenced by the frontend. The
bundle is separate from the continuous T5 model.

Measured held-out metrics on CPU:

- 18 test clips; accuracy 61.1%, macro recall 61.1%.
- Per-class recall: friend 0%, hello 100%, hospital 66.7%, market 100%,
  school 33.3%, thank you 66.7%.
- Model accepted 7/18 held-out clips; 6 were correct and 1 was a false
  accept. The other 11 were rejected as low confidence.
- A held-out `hello` clip (`hello__INCLUDE__00566__MVI_0029.mp4`, 38 frames)
  produced `hello`, accepted, with uncalibrated softmax score 0.7959.
- Another held-out `friend` clip was rejected; the API returned empty text,
  `accepted=false`, and `reason=low_model_confidence`.
- CPU model load measured 11.5 ms; median inference was 2.05 ms per clip,
  with a 239 ms cold first pass. Offline video landmark extraction took a
  median 1.83 seconds per clip on this CPU. The API consumes frontend
  landmarks, not video.
- Backend server RSS after loading and one WebSocket request was 574,184 KiB. No GPU was
  available. Full training runtime was not separately timed.

The backend loaded this bundle with `SIGNSAGE_ALLOW_UNPROMOTED_DEV=true`.
`/health` reported `model_loaded=true`; `/capabilities` reported
`sign_to_text_source=isolated-word`. REST inference returned HTTP 200 for a
real held-out landmark sequence. An empty `frames` request returned HTTP 422.
The frontend's WebSocket protocol returned a final accepted `hello` result for
a real held-out sequence, and returned empty text with
`reason=low_model_confidence` for a held-out `school` sequence. The unpromoted
bundle failed the promotion check as intended.

The live UI labels this as a limited isolated-word pilot, not sentence
translation, and labels the score uncalibrated. Production promotion remains
closed. Isolated-word promotion requires a documented GO, passing tests,
verified signer-independent evaluation, test accuracy and macro recall of at
least 0.80, and every class recall of at least 0.60. This run fails measured
accuracy and signer-identity requirements.

The generated grouped-split bundle and 35 MB data/assets are outside Git at
`/tmp/signsage-isl-bundle-grouped` and `/tmp/signsage-isl-data`. These are
session-local artifacts; rerun the documented command to recreate them.
MediaPipe task assets were fetched from Google's official storage URLs also
used by the frontend. Their separate redistribution terms were not
independently audited; do not redistribute those assets without confirming
their terms. Dataset clips remain under the source's CC-BY-4.0 attribution
requirements.

Verification commands and results:

- `backend/.venv/bin/python -m pytest -q -o pythonpath=backend backend/tests`
  — 234 passed, 5 warnings.
- `npm --prefix frontend test` — 20 passed.
- `npm --prefix frontend run build` — passed; existing large-chunk warning.
- `backend/.venv/bin/python backend/scripts/train_isolated_words.py --data-dir
  /tmp/signsage-isl-data --output /tmp/signsage-isl-bundle-grouped --epochs
  50 --seed 42` — completed on CPU.
- `backend/.venv/bin/python backend/scripts/smoke_isolated_words.py
  --data-dir /tmp/signsage-isl-data --bundle
  /tmp/signsage-isl-bundle-grouped --all-test` — ran all 18 held-out videos.
- A real Uvicorn process loaded the generated bundle; the tested held-out
  landmark request returned HTTP 200 with `text=hello`. Empty frames returned
  HTTP 422. The final real inference was made with the unpromoted dev override;
  it does not establish production readiness.
