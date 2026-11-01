# Storing and deploying backend models on Render

## Recommended setup

Keep model artifacts in a private, versioned artifact store. At startup,
Render downloads the selected bundle to its persistent disk at `/models`.
Do not commit weights to Git or bake them into the Docker image.

The repository already implements this flow:

- `render.yaml` mounts a 2 GB persistent disk at `/models`.
- `backend/scripts/fetch_bundle.sh` downloads `SIGNSAGE_BUNDLE_URL`, checks
  `SIGNSAGE_BUNDLE_SHA256` when set, and extracts the ZIP into
  `SIGNSAGE_MODEL_BUNDLE_DIR`.
- A complete bundle already on disk is reused at startup.

The artifact store is the source of truth; Render's disk is a runtime copy,
not the only backup.

## Hugging Face Hub or Spaces?

Use a **private Hugging Face Hub Model repository** to store a model bundle
that Render downloads. The Hub is the artifact repository; pin the download
to a commit so a later upload cannot silently change what Render receives.

Use a **Space** only if you want Hugging Face to host and run a separate
interactive demo or inference app. It is not the right place to park files
for Render to download, and it does not replace this project's Render
backend. Keep one serving backend to avoid duplicating runtime, secrets, and
deployment configuration.

You do not need to upload a model just to deploy the current backend. Its
checkpoint-free heuristic fallback can recognize only five static hand
shapes. The general Sign-to-Text transformer is not production-ready, and
the other downloaded checkpoints are not compatible with this backend.
See [Current project state](#current-project-state) before choosing an
artifact.

## Important: choose the right model artifact

The current Render configuration expects the Sign-to-Text **exported
deployment bundle**, not an arbitrary `.pt` or `.pth` file. Its ZIP must have
these entries at the archive root:

```text
checkpoint/
tokenizer/
landmark_contract.json
sign_to_text_deployment.json
```

Use the bundle produced by the SignSage export pipeline. Do not substitute
the ISLiM or Sanket checkpoints, or the mobile TorchScript files: repository
audits say those candidates are not integrated with the backend adapter.
The current Sign-to-Text evaluations are NO-GO, so storing or loading a
checkpoint does not make it appropriate to serve. Keep
`SIGNSAGE_MODEL_PROMOTED=false` unless a future model passes the documented
evaluation gate and receives explicit approval.

## Deployment steps

1. **Choose a model that is actually eligible to deploy.** It must be
   compatible with the SignSage adapter, pass the documented held-out quality
   and diversity gates, and have redistribution rights cleared. Do not use
   ISLiM or Sanket for the current Sign-to-Text adapter, and do not treat a
   successful load test as a quality pass.

2. **Export and validate a backend bundle.** From the backend environment,
   point the local settings at the exported directory and run the existing
   loader check:

   ```bash
   cd backend
   export SIGNSAGE_MODEL_BUNDLE_DIR=/absolute/path/to/signsage_deployment_bundle
   python -m scripts.model_check
   ```

   This checks loading and the activity-gate self-test; it does not prove
   translation quality or authorize production promotion.

3. **Package the bundle with its contents at the ZIP root.** Use an absolute
   ZIP path outside the bundle directory:

   ```bash
   BUNDLE_DIR=/absolute/path/to/signsage_deployment_bundle
   ZIP_PATH=/absolute/path/to/signsage_deployment_bundle-v1.zip
   (cd "$BUNDLE_DIR" && zip -0 -r "$ZIP_PATH" *)
   unzip -l "$ZIP_PATH" | head -30
   sha256sum "$ZIP_PATH"
   ```

   Check that `checkpoint/`, `tokenizer/`, and the JSON contract files are
   archive-root entries—not nested under a
   `signsage_deployment_bundle/` directory. Save the SHA-256 value for Render.

4. **Upload to Hugging Face Hub.**
   - On Hugging Face, create a **Model** repository, set it to **Private**,
     and upload the versioned ZIP from step 3. Do not create a Space for this.
   - Open the uploaded file's history and copy the commit SHA. Use that
     immutable revision in the file URL:

     ```text
     https://huggingface.co/USERNAME/REPOSITORY/resolve/COMMIT_SHA/signsage_deployment_bundle-v1.zip?download=true
     ```

   - For a private repository, create a read-only access token and add it in
     Render as the secret `SIGNSAGE_HF_TOKEN`. For a public repository, the
     token is not needed, but the checksum is still required below.
   - Never upload datasets or weights whose redistribution rights are
     unverified. Keep tokens out of URLs, Git, and logs.

   A private S3-compatible object store is also suitable if you already use
   one. The artifact must be downloadable by Render over HTTPS; its URL must
   remain valid when a fresh download is needed.

5. **Configure Render.** In the `signsage-backend` service's dashboard, add
   or update these environment variables:

   | Variable | Value |
   |---|---|
   | `SIGNSAGE_BUNDLE_URL` | HTTPS URL pinned to the Hugging Face commit SHA |
   | `SIGNSAGE_BUNDLE_SHA256` | SHA-256 from step 3 |
   | `SIGNSAGE_MODEL_BUNDLE_DIR` | A new path, e.g. `/models/signsage_deployment_bundle_v1` |
   | `SIGNSAGE_HF_TOKEN` | Read token, private Hub repository only; store as a secret |

   Keep the persistent disk mounted at `/models`. Its capacity must fit the
   downloaded ZIP, extracted bundle, and any retained older versions; increase
   the disk size in `render.yaml` and redeploy if needed. If using a
   time-limited signed URL, it must still be valid whenever Render needs to
   download the bundle again (for example, after disk loss or when using a
   new bundle path).

6. **Deploy and verify.** Check Render logs for checksum verification and
   `bundle ready`. Then check:

   ```bash
   curl -fsS https://YOUR-RENDER-SERVICE.onrender.com/api/v1/health
   curl -fsS https://YOUR-RENDER-SERVICE.onrender.com/api/v1/capabilities
   ```

   Confirm `model_bundle_found` is true in health and that capabilities
   honestly report the model's availability and promotion status. A healthy
   HTTP service alone does not mean inference is enabled or the model is
   validated. With the current `SIGNSAGE_MODEL_PROMOTED=false`, production
   Sign-to-Text requests remain gated.

## Updating or rolling back a bundle

Do not overwrite a bundle in place. The startup script skips downloading when
the configured directory already looks complete, even if the URL or checksum
has changed. Upload each release to a new immutable storage path, then set
Render's `SIGNSAGE_MODEL_BUNDLE_DIR` to a new directory such as
`/models/signsage_deployment_bundle_v2` along with its URL and checksum.
Deploy and verify; rollback by restoring the previous three values. Keep the
old directory until the new release is verified.

## Security notes

- Treat `.pt` and `.pth` files as trusted-code inputs: PyTorch checkpoint
  loading can involve pickle deserialization. Only deploy artifacts from a
  trusted, access-controlled source, and verify the bundle checksum against a
  value obtained through a trusted channel.
- Do not enable `SIGNSAGE_ALLOW_UNPROMOTED_DEV` on Render. It is a local
  development escape hatch, not a deployment fix.
- ISLiM and Sanket weight redistribution rights are documented as unresolved
  in `research/pretrained_models/HF_STORAGE_POLICY.md`; do not rehost those
  weights until the rights are verified.

## Current project state

This is the latest state documented in the repository (evidence through
2026-10-27), not a live check of your Render dashboard:

- The app has an implemented frontend/backend path, browser landmark capture,
  WebSocket transport, health/capability reporting, and Render deployment
  configuration. Render can start without a model bundle, but then
  `model_bundle_found` is false and the transformer path is unavailable.
- A checkpoint-free heuristic recognizes five static hand shapes only. It
  does not translate continuous signing or sentences.
- The main Sign-to-Text transformer is NO-GO because of output collapse. The
  isolated-word pilot scored 61.1% test accuracy and is also unpromoted; neither
  should be presented as production translation.
- A separate static alphabet classifier scored 87.1% test accuracy on its
  documented held-out set, with weak classes. This is image classification,
  not sentence translation, and requires its own bundle/configuration.
- Text-to-sign has a working gloss/procedural preview, but its procedural
  motions are explicitly not authored ISL.
- The latest recorded local checks include 240 backend tests, a passing
  frontend build, 27 frontend tests, and 7 Playwright checks. These establish
  software behavior only—not translation quality.
- Live Render deployment, real-camera operation, and real Supabase/database
  integration are not verified in those records.

**Practical next move:** check `/api/v1/health` and `/api/v1/capabilities` on
your Render URL. If the goal is to host the existing app, fix deployment
configuration only if those checks fail; do not upload the experimental
checkpoints. Use the Hub workflow above only after you have a compatible,
evaluated, license-cleared deployment bundle. For the evidence details, see
[`STATUS_AND_ROADMAP.md`](../STATUS_AND_ROADMAP.md),
[`PROTOTYPE_COMPLETION_REPORT.md`](../reports/PROTOTYPE_COMPLETION_REPORT.md),
and [`PRODUCT_READINESS_AUDIT.md`](../reports/PRODUCT_READINESS_AUDIT.md).
