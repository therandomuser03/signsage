#!/usr/bin/env bash
# Fetch the Sign->Text deployment bundle at container start (Render).
#
# Weights are NOT baked into the image (~100+ MB per run). This script
# downloads SIGNSAGE_BUNDLE_URL (a signsage_deployment_bundle.zip, see
# research/manifests/models/*.yaml) into SIGNSAGE_MODEL_BUNDLE_DIR and
# verifies the entries the backend requires:
#   checkpoint/, tokenizer/, landmark_contract.json, sign_to_text_deployment.json
# (see app/infrastructure/storage/model_storage.py).
#
# Behaviour:
#   - bundle already complete on disk (persistent Render disk) -> skip download.
#   - SIGNSAGE_BUNDLE_URL unset -> warn and exit 0. The server still starts;
#     /api/v1/health reports model_bundle_found=false and translation
#     endpoints refuse with MODEL_UNAVAILABLE. Never fake readiness.
#   - checksum mismatch / missing entries -> fail loudly (non-zero exit) so
#     the deploy visibly fails instead of serving a corrupt model.
#
# Only dependency is python3 stdlib (urllib/zipfile/hashlib).
set -euo pipefail

DEST="${SIGNSAGE_MODEL_BUNDLE_DIR:-/models/signsage_deployment_bundle}"

required_present() {
  [ -f "$DEST/sign_to_text_deployment.json" ] \
    && [ -f "$DEST/landmark_contract.json" ] \
    && [ -d "$DEST/checkpoint" ] \
    && [ -d "$DEST/tokenizer" ]
}

if required_present; then
  echo "[MODEL] bundle already present at $DEST, skipping download."
  exit 0
fi

if [ -z "${SIGNSAGE_BUNDLE_URL:-}" ]; then
  echo "[MODEL] WARNING: no bundle at $DEST and SIGNSAGE_BUNDLE_URL is unset." >&2
  echo "[MODEL] Server will still start; /api/v1/health reports model_bundle_found=false" >&2
  echo "[MODEL] and translation endpoints refuse with MODEL_UNAVAILABLE." >&2
  exit 0
fi

mkdir -p "$DEST"
python3 - "$DEST" <<'EOF'
import hashlib
import os
import sys
import urllib.request
import zipfile

dest = sys.argv[1]
url = os.environ["SIGNSAGE_BUNDLE_URL"]
expected_sha = os.environ.get("SIGNSAGE_BUNDLE_SHA256", "").strip().lower()
tmp = os.path.join(dest, "bundle.download")

print(f"[MODEL] downloading bundle ...", flush=True)
# shortcut: 5-minute timeout; upgrade to a streaming download with progress
# reporting if bundles grow larger.
with urllib.request.urlopen(url, timeout=300) as response:
    with open(tmp, "wb") as out_file:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out_file.write(chunk)

if expected_sha:
    digest = hashlib.sha256()
    with open(tmp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        os.remove(tmp)
        raise SystemExit(
            f"[MODEL] checksum mismatch for downloaded bundle "
            f"(got {digest.hexdigest()}, want {expected_sha})"
        )
    print("[MODEL] checksum OK.", flush=True)

if not zipfile.is_zipfile(tmp):
    os.remove(tmp)
    raise SystemExit("[MODEL] downloaded bundle is not a zip archive; refusing to extract.")

with zipfile.ZipFile(tmp) as zf:
    zf.extractall(dest)
os.remove(tmp)

required = ["sign_to_text_deployment.json", "landmark_contract.json", "checkpoint", "tokenizer"]
missing = [n for n in required if not os.path.exists(os.path.join(dest, n))]
if missing:
    raise SystemExit(f"[MODEL] bundle extracted but missing required entries: {missing}")
print(f"[MODEL] bundle ready at {dest}.", flush=True)
EOF
