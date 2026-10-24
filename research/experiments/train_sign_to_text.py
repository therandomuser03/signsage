#!/usr/bin/env python3
"""SignSage v2 sign-to-text training — corrected post-352868924 configuration.

What this is: a reproducible, resumable training job for continuous ISL
pose-sequence -> English translation on iSign v1.1. It reuses the backend's
 OWN model, preprocessing, and masks (single canonical implementation), keeps
every training mechanic the 352868924 audit verified as correct
(shift / masks / ignore_index / video-level split / tokenizer procedure),
and uses the configuration described in research/training/READINESS.md:
pre-norm, larger capacity, label smoothing,
longer schedule, per-epoch diversity collapse gate, hard overfit gate.

Stages (select with --stages subset, default all):
    audit -> tokenizer -> prepare -> overfit-gate -> train -> evaluate -> export
    measure is a separate opt-in dry-run (zstd projection, creates no cache).

    python train_sign_to_text.py --csv /path/iSign_v1.1.csv \\
        --pose-source /path/to/poses.zip --out /kaggle/working/s2t_v2
    python train_sign_to_text.py --smoke --out /tmp/s2t_smoke   # CPU plumbing check

Resumable: a per-epoch recovery snapshot (<out>/checkpoints/
signsage_transformer_recovery.pt holds model + optimizer + scheduler +
epoch/best/bad/history + RNG states) plus the best checkpoint are written
after accepted epochs; re-run with --resume to continue. Epoch 1 is provisional:
the diversity guard is enforced from epoch 2, not proof of model quality.
Never overwrites run 352868924.

Multi-session (Kaggle): the prepare stage materializes a resumable local
pose cache (<out>/pose_cache) once; later sessions reuse it plus the token
cache (<out>/tokenizer/tokens.npz) and --resume continues from the recovery
checkpoint. /kaggle/working is ephemeral -- persist <out> via Save Version
and re-attach it as an input dataset next session (see signsage_kaggle.ipynb).

EXPOSURE-BIAS MITIGATION (added after the 2-epoch diagnostic collapsed):
    --decoder-input-dropout P replaces each non-BOS decoder input token with a
    random vocab id (in [4, vocab_size)) with probability P during training.
    This is an experimental prefix-perturbation control, not a demonstrated
    fix for autoregressive collapse. Default 0.0 preserves
    the original behavior. Gate and validation paths are unaffected.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
import zipfile
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from app.ml.models.sign_to_text import SignTransformer, causal_mask, padding_mask  # noqa: E402
from app.ml.postprocessing.decoding import ban_repeated_ngram  # noqa: E402
from app.ml.preprocessing.landmarks import (  # noqa: E402
    COMPONENT_ORDER,
    COORDINATES_PER_LANDMARK,
    TOTAL_LANDMARKS,
)
from app.ml.preprocessing.normalization import normalize_pose_sequence  # noqa: E402
from app.ml.preprocessing.sequence import resize_sequence  # noqa: E402
from audit_dataset import clean_dataframe, normalize_text, video_split  # noqa: E402

SEED = 42
MAX_FRAMES = 160
MAX_TOKENS = 96
LANDMARK_DIM = TOTAL_LANDMARKS * COORDINATES_PER_LANDMARK  # 1629
MODEL_INPUT_DIM = LANDMARK_DIM * 2  # model concatenates frame deltas
FULL_FEATURE_MASK = list(range(LANDMARK_DIM))

# v2 model capacity; effectiveness requires held-out validation.
D_MODEL, NHEAD, NLAYERS, FF = 256, 4, 4, 1024
DROPOUT = 0.15
NORM_FIRST = True  # pre-norm: stable from step 0 on short budgets
LABEL_SMOOTHING = 0.1
PEAK_LR = 1e-4
WARMUP_FRACTION = 0.15
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Decoding constraints (identical to the serving contract).
MAX_DECODE_TOKENS = 94
MIN_OUTPUT_TOKENS = 1
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.15

# Collapse gate: unique-fraction on a val decode sample; STOP below this.
DIVERSITY_FLOOR = 0.05
DIVERSITY_SAMPLE = 512

# Pose materialization cache (Phase A). Processed tensors are NOT cached:
# 123,779 samples x 160 x 1629 float32 ~= 120 GiB (~60 GiB in fp16) -- both
# exceed practical Kaggle disk. Instead the cache stores the compressed
# `.pose` member bytes verbatim in sharded ZIPs (ZIP_STORED, no recompress),
# which is the smallest representation that preserves bit-exact preprocessing
# (same bytes -> same load_pose_item path as remote). Estimated from the
# 158 GiB archive: a fraction of it (only needed UIDs), projected at runtime
# after the first 200 samples; --max-cache-gb stops gracefully if exceeded.
POSE_CACHE_VERSION = 1
POSE_CACHE_MANIFEST = "manifest.json"
POSE_CACHE_SHARD_PREFIX = "shard-"
POSE_CACHE_SHARD_SUFFIX = ".zip"
DEFAULT_SHARD_SAMPLES = 2000
DEFAULT_VERIFY_EVERY = 200
TOKEN_CACHE_NAME = "tokens.npz"
TOKEN_CACHE_META = "tokens_meta.json"


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    """object_pairs_hook that rejects JSON objects containing duplicate keys.

    Used when loading the pose-cache manifest so duplicate UID keys can never
    be silently overwritten by stdlib json (which keeps the last value).
    """
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_strict_manifest(path: str | Path) -> dict:
    """Load a pose-cache manifest JSON, rejecting duplicate keys.

    Fails closed: duplicate object keys raise ValueError instead of being
    collapsed to the last value by stdlib json.
    """
    return json.loads(Path(path).read_text(), object_pairs_hook=_reject_duplicate_json_keys)


def check_cache_identity(uid_to_shard: dict, expected_ids: set[str]) -> tuple[list[str], list[str]]:
    """Compare cached identities against expected pilot identities.

    Returns (missing, extra) sorted lists of identities. Both empty == exact
    set equality (no expected sample missing, no unexpected sample present).
    """
    manifest_ids = set(uid_to_shard.keys())
    missing = sorted(expected_ids - manifest_ids)
    extra = sorted(manifest_ids - expected_ids)
    return missing, extra


def log(msg: str) -> None:
    print(f"[s2t-v2 {datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


HTTP_RETRY_ATTEMPTS = 5      # bounded: total attempts for one transient remote failure
RETRY_AFTER_CAP_S = 60.0     # never sleep longer than this on a server-supplied Retry-After


class RemoteTransportError(ConnectionError):
    """Remote pose access still failing after bounded retries (transport reset / 5xx).

    A subclass of ConnectionError so existing handlers keep working. It signals a DATA-ACCESS
    failure: never a model-quality verdict, and never a reason to switch to another cache."""


def _atomic_torch_save(obj, path) -> None:
    """torch.save to `path` atomically (tmp file + fsync + os.replace).

    A Kaggle session killed mid-write leaves the previous good file intact instead of a
    truncated checkpoint that would break --resume and lose all progress."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _retry_after_delay(retry_after: str | None, attempt: int) -> float:
    """Delay before retrying a transient remote server error.

    Honors a Retry-After header (seconds) when present and valid; otherwise
    uses bounded exponential backoff plus jitter so concurrent workers do not
    hammer the server in lockstep. Never exceeds the per-attempt timeout budget.
    """
    if retry_after is not None:
        try:
            return min(RETRY_AFTER_CAP_S, max(0.0, float(retry_after.strip())))
        except ValueError:
            pass
    return 1.5 * (2 ** attempt) + random.random()


class PhaseTimer:
    """Tiny perf probe: split data-wait vs compute vs misc; periodic logs only."""

    def __init__(self, label: str, report_every: int = 100):
        self.label = label
        self.report_every = report_every
        self.t_data = 0.0
        self.t_compute = 0.0
        self.t_misc = 0.0
        self.n = 0
        self._t0 = time.perf_counter()

    def add_batch(self, d_data: float, d_compute: float) -> None:
        self.t_data += d_data
        self.t_compute += d_compute
        self.n += 1
        if self.n % self.report_every == 0:
            total = self.t_data + self.t_compute + self.t_misc
            log(f"{self.label}: batch {self.n} "
                f"data={self.t_data:.1f}s compute={self.t_compute:.1f}s misc={self.t_misc:.1f}s "
                f"(data share {self.t_data / max(1e-9, total):.0%})")

    def summary(self) -> str:
        total = self.t_data + self.t_compute + self.t_misc
        return (f"{self.label}: {self.n} batches in {total:.1f}s "
                f"(data {self.t_data:.1f}s / compute {self.t_compute:.1f}s / misc {self.t_misc:.1f}s)")


def estimate_processed_cache_gib(n_samples: int, dtype_bytes: int = 4) -> float:
    """Upper-bound size if processed (160,1629) tensors were cached.
    Kept as documentation for why Phase A caches compressed member bytes."""
    return n_samples * MAX_FRAMES * LANDMARK_DIM * dtype_bytes / (1024 ** 3)


# --------------------------------------------------------------------------
# Pose sources
# --------------------------------------------------------------------------

def get_hf_token() -> str | None:
    """Safe retrieval of HF_TOKEN from environment or Kaggle Secrets without printing."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]
        sec = UserSecretsClient().get_secret("HF_TOKEN")
        if sec:
            token = str(sec).strip()
            os.environ["HF_TOKEN"] = token  # populate for huggingface_hub without printing
            return token
    except Exception:
        pass
    return None


class PoseSource:
    def read_member_bytes(self, uid: str) -> bytes:
        raise NotImplementedError


class DirPoseSource(PoseSource):
    def __init__(self, root: str):
        self.root = Path(root)

    def read_member_bytes(self, uid: str) -> bytes:
        path = self.root / f"{uid}.pose"
        if not path.is_file():
            raise FileNotFoundError(f"UID not found in pose dir: {uid}")
        return path.read_bytes()


class ZipPoseSource(PoseSource):
    def __init__(self, path: str):
        self.zf = zipfile.ZipFile(path, mode="r", allowZip64=True)
        self.names = set(self.zf.namelist())

    def read_member_bytes(self, uid: str) -> bytes:
        for cand in (f"iSign-poses_v1.1/{uid}.pose", f"{uid}.pose", f"poses/{uid}.pose"):
            if cand in self.names:
                with self.zf.open(cand, "r") as fh:
                    return fh.read()
        raise FileNotFoundError(f"UID not found in pose archive: {uid}")


class RemoteConcatFile(io.RawIOBase):
    """Seekable view over the 4-part iSign pose ZIP on Hugging Face.

    The four parts are consecutive byte ranges of ONE ZIP archive; this
    presents them as one file so zipfile can read the ZIP64 central
    directory, then fetch only requested members via bounded HTTP ranges.
    Nothing is ever fully downloaded (at most 8 cached 32 MiB chunks)."""

    PARTS = [
        "iSign-poses_v1.1_part_aa",
        "iSign-poses_v1.1_part_ab",
        "iSign-poses_v1.1_part_ac",
        "iSign-poses_v1.1_part_ad",
    ]

    def __init__(self, token: str, chunk_mb: int = 32):
        import requests
        from huggingface_hub import hf_hub_url

        super().__init__()
        self._requests = requests
        self.urls = [hf_hub_url("Exploration-Lab/iSign", p, repo_type="dataset") for p in self.PARTS]
        self.sizes = [self._probe(u, token) for u in self.urls]
        self.starts = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.total_size = int(sum(self.sizes))
        self.pos = 0
        self.token = token
        self.chunk_size = int(chunk_mb * 1024 * 1024)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}", "User-Agent": "SignSage-v2/1.0"})
        from collections import OrderedDict
        self.cache: OrderedDict = OrderedDict()
        self.resolved: dict = {}  # part idx -> post-redirect CDN URL (re-resolved on expiry)

    def _probe(self, url: str, token: str) -> int:
        """Size probe (GET bytes=0-0) with BOUNDED retries on transient failures.

        Runs when a store is opened -- including inside every DataLoader worker that re-opens
        it -- so one connection reset here must not kill the run. Retried: timeouts, connection
        errors, chunked-encoding errors and HTTP 5xx. NOT retried: 401/403 and any other status
        (those are configuration errors). Exhaustion raises RemoteTransportError."""
        headers = {"Authorization": f"Bearer {token}", "Range": "bytes=0-0", "User-Agent": "SignSage-v2/1.0"}
        last = None
        for attempt in range(HTTP_RETRY_ATTEMPTS):
            retry_after = None
            try:
                r = self._requests.get(url, headers=headers, timeout=(20, 60), allow_redirects=True)
            except (self._requests.exceptions.Timeout, self._requests.exceptions.ConnectionError,
                    self._requests.exceptions.ChunkedEncodingError) as exc:
                last = exc
            else:
                if not 500 <= r.status_code < 600:
                    break
                last = RuntimeError(f"probe failed: HTTP {r.status_code}")
                retry_after = r.headers.get("Retry-After")
            if attempt < HTTP_RETRY_ATTEMPTS - 1:
                time.sleep(_retry_after_delay(retry_after, attempt))
        else:
            raise RemoteTransportError(
                f"remote size probe failed after {HTTP_RETRY_ATTEMPTS} attempts "
                f"({type(last).__name__}); data-access failure, not a model verdict") from last
        if r.status_code in (401, 403):
            raise PermissionError(
                f"Hugging Face dataset access denied (HTTP {r.status_code}) for Exploration-Lab/iSign. "
                "Ensure HF_TOKEN is valid and has accepted terms for the gated repository."
            )
        if r.status_code != 206:
            raise RuntimeError(f"Could not probe remote part size: HTTP {r.status_code}")
        total = r.headers.get("Content-Range", "0/0").rsplit("/", 1)[-1]
        if not total.isdigit():
            raise RuntimeError(f"Could not parse total size from Content-Range: {r.headers.get('Content-Range')!r}")
        return int(total)

    def readable(self): return True

    def seekable(self): return True

    def tell(self): return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            new = offset
        elif whence == os.SEEK_CUR:
            new = self.pos + offset
        elif whence == os.SEEK_END:
            new = self.total_size + offset
        else:
            raise ValueError("invalid whence")
        if new < 0:
            raise ValueError("negative seek position")
        self.pos = min(int(new), self.total_size)
        return self.pos

    def _part_for(self, pos):
        idx = max(0, min(int(np.searchsorted(self.starts, pos, side="right") - 1), len(self.urls) - 1))
        return idx, pos - self.starts[idx]

    def _fetch(self, part_idx, local_start, n):
        key = (part_idx, (local_start // self.chunk_size) * self.chunk_size)
        if key in self.cache:
            self.cache.move_to_end(key)
            returned_start, content = self.cache[key]
            piece = content[local_start - returned_start:local_start - returned_start + n]
            if len(piece) == n:
                return piece
            self.cache.pop(key, None)  # never serve a short/corrupt entry
        chunk_start = key[1]
        chunk_end = min(self.sizes[part_idx] - 1, chunk_start + self.chunk_size - 1)
        last = None
        for attempt in range(5):
            try:
                request_url = self.resolved.get(part_idx, self.urls[part_idx])
                r = self.session.get(request_url, headers={"Range": f"bytes={chunk_start}-{chunk_end}"},
                                     timeout=(20, 180), allow_redirects=True)
                if r.status_code != 206 and request_url != self.urls[part_idx]:
                    self.resolved.pop(part_idx, None)  # cached CDN link expired; re-resolve once
                    r = self.session.get(self.urls[part_idx], headers={"Range": f"bytes={chunk_start}-{chunk_end}"},
                                         timeout=(20, 180), allow_redirects=True)
                if r.status_code != 206:
                    if 500 <= r.status_code < 600 and attempt < 4:
                        # Transient server error (e.g. HTTP 503): bounded retry with
                        # backoff, honoring Retry-After when the server supplies it.
                        delay = _retry_after_delay(r.headers.get("Retry-After"), attempt)
                        last = RuntimeError(f"range request failed: HTTP {r.status_code}")
                        self.resolved.pop(part_idx, None)
                        time.sleep(delay)
                        continue
                    raise RuntimeError(f"range request failed: HTTP {r.status_code}")
                m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("Content-Range", "").strip())
                if not m:
                    raise RuntimeError(f"Invalid Content-Range: {r.headers.get('Content-Range')!r}")
                rs, re_, total = map(int, m.groups())
                if total != self.sizes[part_idx]:
                    raise RuntimeError(f"part {part_idx} size changed: {total} != {self.sizes[part_idx]}")
                if not (rs <= chunk_start and chunk_end <= re_):
                    raise RuntimeError(f"range {rs}-{re_} does not cover {chunk_start}-{chunk_end}")
                if len(r.content) != re_ - rs + 1:
                    raise RuntimeError(f"incomplete range body for part {part_idx}")
                self.resolved[part_idx] = r.url
                self.cache[key] = (rs, r.content)
                self.cache.move_to_end(key)
                while len(self.cache) > 8:
                    self.cache.popitem(last=False)
                piece = r.content[local_start - rs:local_start - rs + n]
                if len(piece) != n:
                    raise RuntimeError(f"short range slice for part {part_idx}")
                return piece
            except (self._requests.exceptions.Timeout, self._requests.exceptions.ConnectionError,
                    self._requests.exceptions.ChunkedEncodingError) as exc:
                last = exc
                self.resolved.pop(part_idx, None)
                time.sleep(1.5 * 2 ** attempt)
        raise RemoteTransportError(f"range fetch failed after retries: {key}") from last

    def read(self, n=-1):
        if self.pos >= self.total_size:
            return b""
        if n is None or n < 0:
            n = self.total_size - self.pos
        n = min(int(n), self.total_size - self.pos)
        out = bytearray()
        remaining = n
        while remaining:
            part_idx, local = self._part_for(self.pos)
            take = min(remaining, self.sizes[part_idx] - local, self.chunk_size - (local % self.chunk_size))
            out.extend(self._fetch(part_idx, local, take))
            self.pos += take
            remaining -= take
        return bytes(out)


class RemotePoseSource(PoseSource):
    def __init__(self, token: str, chunk_mb: int = 32):
        self.zf = zipfile.ZipFile(RemoteConcatFile(token, chunk_mb), mode="r", allowZip64=True)
        self.names = set(self.zf.namelist())

    def read_member_bytes(self, uid: str) -> bytes:
        for cand in (f"iSign-poses_v1.1/{uid}.pose", f"{uid}.pose"):
            if cand in self.names:
                with self.zf.open(cand, "r") as fh:
                    return fh.read()
        raise FileNotFoundError(f"UID not found in remote pose archive: {uid}")


class ShardPoseSource(PoseSource):
    """Local disk cache of verbatim `.pose` member bytes in sharded ZIPs.

    Created once by materialize_pose_cache() from any other PoseSource
    (normally remote). Reads are plain local-disk random access -- no HTTP.
    Each shard is a ZIP_STORED archive of `<uid>.pose` members, so a shard
    is readable by the existing ZipPoseSource with zero new decode code.
    Shard files are opened lazily (per process), hence DataLoader-fork-safe.
    """

    def __init__(self, cache_dir: str, max_open: int = 8):
        from collections import OrderedDict

        self.cache_dir = Path(cache_dir)
        self.manifest = load_strict_manifest(self.cache_dir / POSE_CACHE_MANIFEST)
        if self.manifest.get("version") != POSE_CACHE_VERSION:
            raise RuntimeError(f"pose cache version {self.manifest.get('version')} != {POSE_CACHE_VERSION}")
        self.index: dict = self.manifest["uid_to_shard"]
        self._open: OrderedDict = OrderedDict()
        self._max_open = max(1, int(max_open))

    def _shard_store(self, shard: str) -> ZipPoseSource:
        store = self._open.get(shard)
        if store is None:
            store = ZipPoseSource(str(self.cache_dir / shard))
            self._open[shard] = store
            while len(self._open) > self._max_open:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(shard)
        return store

    def read_member_bytes(self, uid: str) -> bytes:
        shard = self.index.get(uid)
        if shard is None:
            raise FileNotFoundError(f"UID not in local pose cache: {uid}")
        return self._shard_store(shard).read_member_bytes(uid)


def cache_coverage(cache_dir: Path, uids: list[str]) -> tuple[int, int]:
    """(present, missing) counts for uids against an existing cache dir."""
    manifest_path = cache_dir / POSE_CACHE_MANIFEST
    if not manifest_path.is_file():
        return 0, len(uids)
    try:
        manifest = load_strict_manifest(manifest_path)
    except (OSError, ValueError):
        return 0, len(uids)
    ver = manifest.get("version")
    if ver != POSE_CACHE_VERSION:
        raise RuntimeError(f"incompatible cache version: {ver} != {POSE_CACHE_VERSION}")
    index = manifest.get("uid_to_shard", {})
    physical = _scan_shard_members(cache_dir)
    if physical and set(index.keys()) != set(physical.keys()):
        raise RuntimeError(f"manifest does not match physical shard members ({len(index)} vs {len(physical)})")
    present = sum(1 for u in uids if u in index)
    return present, len(uids) - present


def _scan_shard_members(cache_dir: Path) -> dict:
    """Source of truth on resume: actual members present in shard files."""
    found: dict = {}
    for shard_path in sorted(cache_dir.glob(f"{POSE_CACHE_SHARD_PREFIX}*{POSE_CACHE_SHARD_SUFFIX}")):
        try:
            with zipfile.ZipFile(shard_path, mode="r", allowZip64=True) as zf:
                for name in zf.namelist():
                    if name.endswith(".pose"):
                        found[name[:-len(".pose")]] = shard_path.name
        except zipfile.BadZipFile:
            log(f"cache: ignoring corrupt partial shard {shard_path.name} (will be rebuilt)")
    return found


def _write_manifest(cache_dir: Path, uid_to_shard: dict, complete: bool) -> None:
    tmp = cache_dir / (POSE_CACHE_MANIFEST + ".tmp")
    tmp.write_text(json.dumps({"version": POSE_CACHE_VERSION, "uid_to_shard": uid_to_shard,
                               "complete": complete, "n": len(uid_to_shard)}))
    os.replace(tmp, cache_dir / POSE_CACHE_MANIFEST)


def require_local_cache(data_mode: str, stages: set, present: int, missing: int, total: int) -> None:
    """Fail closed: in --data-mode cache, model stages must read a COMPLETE local pose cache.

    Without this, an incomplete or mismatched cache (e.g. prepare ran with another --seed/--subset)
    silently falls back to remote streaming -- slow, fragile, and not what was requested."""
    if data_mode == "cache" and missing and stages & {"overfit", "train", "evaluate", "export"}:
        raise SystemExit(
            f"LOCAL POSE CACHE INCOMPLETE: {present:,}/{total:,} cached ({missing:,} missing) in "
            f"--data-mode cache. Refusing to silently stream the rest remotely. Most likely prepare "
            f"ran with a different --seed/--subset than this run (the seed picks the pilot subset), "
            f"or prepare was interrupted/budget-refused. Re-run prepare with IDENTICAL "
            f"--seed/--subset, or choose --data-mode stream explicitly.")


def materialize_pose_cache(uids: list[str], remote: PoseSource, cache_dir: Path,
                           shard_samples: int = DEFAULT_SHARD_SAMPLES,
                           verify_every: int = DEFAULT_VERIFY_EVERY,
                           max_cache_gb: float = 0.0, jobs: int = 1,
                           source_factory=None, budget_gib: float = 0.0) -> Path:
    """Download each uid's `.pose` bytes ONCE into sharded local ZIPs (resumable).

    Skips uids already present in existing shard files (verified from the
    shard central directories, not just the manifest, so interrupted sessions
    resume correctly). Writes progress (manifest) after every shard and every
    200 samples within a shard. Verifies bytes non-empty for every sample and
    full shape/dtype via the real load_pose_item() path every `verify_every`
    samples. Never retains more than one sample's arrays in RAM. Never logs
    tokens (the `remote` store already holds auth; nothing is printed here).

    jobs>1 fetches concurrently (one private PoseSource per thread via
    source_factory; required because HTTP sessions are not shared across
    threads). Shard writing stays single-threaded and deterministic, so the
    cache layout is identical to a sequential run. Memory stays bounded: at
    most 4*jobs blobs are in flight.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    have = _scan_shard_members(cache_dir)
    todo = [u for u in dict.fromkeys(uids) if u not in have]
    log(f"pose cache: {len(have):,} present on disk, {len(todo):,} to fetch "
        f"(total {len(uids):,} unique) fp32-tensor equivalent "
        f"~{estimate_processed_cache_gib(len(uids)):.1f} GiB NOT written)")
    if not todo:
        _write_manifest(cache_dir, have, complete=True)
        log("pose cache: already complete, nothing to fetch.")
        return cache_dir
    bytes_written = sum(
        (cache_dir / s).stat().st_size for s in set(have.values()) if (cache_dir / s).is_file())
    n_done, n_fail = 0, 0
    failures: list = []
    projected = False
    current_shard_idx = 0
    existing = sorted(cache_dir.glob(f"{POSE_CACHE_SHARD_PREFIX}*{POSE_CACHE_SHARD_SUFFIX}"))
    if existing:
        try:
            current_shard_idx = max(int(p.name[len(POSE_CACHE_SHARD_PREFIX):-len(POSE_CACHE_SHARD_SUFFIX)])
                                    for p in existing) + 1
        except ValueError:
            current_shard_idx = len(existing)
    zf = None
    zf_name = ""
    zf_count = 0

    def _close_shard() -> None:
        nonlocal zf
        if zf is not None:
            zf.close()
            zf = None
            _write_manifest(cache_dir, have, complete=False)

    def _results():
        """Yield (uid, blob, error) in todo order; bounded parallel fetch."""
        if jobs is None or jobs <= 1 or source_factory is None:
            for uid in todo:
                try:
                    blob = remote.read_member_bytes(uid)
                    yield uid, blob, None
                except Exception as exc:  # noqa: BLE001
                    yield uid, None, exc
            return
        import concurrent.futures
        import threading

        local = threading.local()

        def _get_source():
            src = getattr(local, "src", None)
            if src is None:
                src = source_factory()
                local.src = src
            return src

        def _fetch(uid):
            try:
                return uid, _get_source().read_member_bytes(uid), None
            except Exception as exc:  # noqa: BLE001
                return uid, None, exc

        width = max(1, 4 * jobs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            for s in range(0, len(todo), width):
                # chunked map: deterministic order, bounded in-flight blobs
                yield from pool.map(_fetch, todo[s:s + width])

    try:
        for i, (uid, blob, err) in enumerate(_results()):
            if zf is None:
                zf_name = f"{POSE_CACHE_SHARD_PREFIX}{current_shard_idx:05d}{POSE_CACHE_SHARD_SUFFIX}"
                zf = zipfile.ZipFile(cache_dir / zf_name, mode="a", allowZip64=True,
                                     compression=zipfile.ZIP_STORED)
                # actual members (not a reset counter): correct after flush-reopen
                # and when resuming into a partial shard from a prior session.
                zf_count = sum(1 for n in zf.namelist() if n.endswith(".pose"))
            try:
                if err is not None:
                    raise err
                if not blob:
                    raise ValueError("empty pose payload")
                zf.writestr(f"{uid}.pose", blob)
                bytes_written += len(blob)
                have[uid] = zf_name
                n_done += 1
                zf_count += 1
                del blob
            except Exception as exc:  # noqa: BLE001 -- recorded, run continues
                n_fail += 1
                failures.append(uid)
                log(f"cache: fetch failed for {uid}: {type(exc).__name__}; continuing")
            if (n_done + n_fail) % 200 == 0:
                _close_shard()  # flush central directory before recording progress
            if n_done == 200 and not projected:
                per_sample = bytes_written / max(1, len(have))
                proj_gib = per_sample * len(uids) / (1024 ** 3)
                el = time.perf_counter() - t0
                rate = 200 / max(1e-9, el)
                eta_h = (len(todo) - 200) / max(1e-9, rate) / 3600
                log(f"cache: projection after 200 samples: ~{per_sample / 1024:.0f} KiB/sample "
                    f"-> ~{proj_gib:.1f} GiB total, rate {rate:.1f} samples/s, ETA ~{eta_h:.1f}h")
                check_cache_budget(proj_gib, budget_gib, cache_dir)  # refuse BEFORE burning hours
                projected = True
            if (n_done + n_fail) % verify_every == 0 and n_done:
                _close_shard()  # flush: reader needs the central directory on disk
                probe_uid = uid if uid in have else todo[max(0, i - 1)]
                if probe_uid in have:
                    frames, true_len = load_pose_item(
                        probe_uid, ShardPoseSource(str(cache_dir)))
                    assert frames.shape == (MAX_FRAMES, LANDMARK_DIM) and frames.dtype == np.float32, \
                        f"cache verify shape {frames.shape} {frames.dtype}"
                    del frames
            if max_cache_gb and bytes_written / (1024 ** 3) > max_cache_gb:
                raise SystemExit(f"pose cache exceeded --max-cache-gb={max_cache_gb} "
                                 f"({bytes_written / (1024 ** 3):.1f} GiB); partial cache kept, resume later.")
            if zf_count >= shard_samples:
                _close_shard()
                current_shard_idx += 1
            if (n_done + n_fail) % 1000 == 0:
                el = time.perf_counter() - t0
                log(f"cache: {n_done + n_fail:,}/{len(todo):,} fetched "
                    f"({bytes_written / (1024 ** 3):.2f} GiB on disk, {el / 60:.1f} min, fail={n_fail})")
        _close_shard()
    finally:
        try:
            if zf is not None:
                zf.close()
        except Exception:  # noqa: BLE001
            pass
    el = time.perf_counter() - t0
    log(f"cache: done {n_done:,}/{len(todo):,} in {el / 60:.1f} min "
        f"({bytes_written / (1024 ** 3):.2f} GiB on disk, fail={n_fail})")
    if n_fail:
        (cache_dir / "fetch_failures.json").write_text(json.dumps(failures))
        raise SystemExit(f"pose cache incomplete: {n_fail} fetches failed (see fetch_failures.json).")
    _write_manifest(cache_dir, have, complete=True)
    return cache_dir


# D-cache measurement (dry-run only): is fixed-160x1629-fp32 + lossless zstd
# small enough to persist? Raw bytes measured ~143 GiB and fixed fp32 is
# exactly 120.2 GiB -- both presumed too big. This stage measures the actual
# zstd ratio on ~2000 deterministically sampled items through the EXACT V2
# preprocessing path, verifies byte-exact round-trips, and projects. It
# creates NO cache. Full D-cache creation stays blocked until the Kaggle
# measurement verdict is reviewed.
ZSTD_LEVEL = 3
MEASURE_REPORT = "measure_report.json"
FULL_DATASET_N = 123779  # usable iSign rows the projection targets


def _zstd():
    try:
        import zstandard
        return zstandard
    except ImportError:
        # ponytail: fallback compressor using stdlib zlib when zstandard is not installed; upgrade path: pip install zstandard on network-enabled hosts
        import zlib
        class _ZstdCompressor:
            def __init__(self, level=3): self.level = level
            def compress(self, data): return zlib.compress(data, level=self.level)
        class _ZstdDecompressor:
            def decompress(self, data): return zlib.decompress(data)
        class _ZstdFallback:
            ZstdCompressor = _ZstdCompressor
            ZstdDecompressor = _ZstdDecompressor
        return _ZstdFallback()


def measure_compression(split_uids: dict, store: PoseSource, out: Path,
                        n_samples: int = 2000, target_gib: float = 0.0) -> dict:
    """Dry-run: sample deterministically, measure zstd(fp32) sizes, report.

    split_uids: {"train": [...], "val": [...], "test": [...]} in authoritative
    row order. Stride-samples ~n_samples across the concatenated order so all
    splits are represented proportionally. Per item: exact load_pose_item(),
    true_len, fixed-fp32 bytes, zstd compress, decompress-and-byte-compare,
    plus a true_len-only compression to attribute padding vs data savings.
    Only running totals + the true_len list are retained, never the population.
    """
    zstandard = _zstd()
    t0 = time.perf_counter()
    ordered = [(split, uid) for split in ("train", "val", "test")
               for uid in split_uids.get(split, [])]
    step = max(1, len(ordered) // max(1, n_samples))
    picked = ordered[::step][:max(1, n_samples)]
    cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    tls: list = []
    per_split: dict = {}
    n_bytes = c_bytes = v_bytes = vc_bytes = 0
    fetch_t = 0.0
    for i, (split, uid) in enumerate(picked):
        t_f = time.perf_counter()
        frames, true_len = load_pose_item(uid, store)
        fetch_t += time.perf_counter() - t_f
        assert frames.shape == (MAX_FRAMES, LANDMARK_DIM) and frames.dtype == np.float32
        full = np.ascontiguousarray(frames).tobytes()
        comp = cctx.compress(full)
        if zstandard.ZstdDecompressor().decompress(comp) != full:
            raise RuntimeError(f"zstd round-trip mismatch for {uid} -- refusing")
        var = np.ascontiguousarray(frames[:true_len]).tobytes()
        vcomp = cctx.compress(var)
        tls.append(int(true_len))
        n_bytes += len(full)
        c_bytes += len(comp)
        v_bytes += len(var)
        vc_bytes += len(vcomp)
        st = per_split.setdefault(split, {"n": 0, "compressed": 0})
        st["n"] += 1
        st["compressed"] += len(comp)
        del frames, full, comp, var, vcomp
        if (i + 1) % 500 == 0:
            log(f"measure: {i + 1:,}/{len(picked):,} "
                f"(ratio so far {n_bytes / max(1, c_bytes):.2f}x)")
    n = len(picked)
    tls_arr = np.asarray(tls, dtype=np.int64)
    ratio = n_bytes / max(1, c_bytes)
    avg_c = c_bytes / max(1, n)
    proj = {}
    for split in ("train", "val", "test"):
        # a tiny split may contribute zero stride samples (e.g. subset pilots);
        # project 0 bytes for it rather than KeyError.
        st = per_split.get(split, {"n": 0, "compressed": 0})
        proj[split] = (st["compressed"] / max(1, st["n"])
                       * len(split_uids.get(split, [])))
    proj_full = avg_c * FULL_DATASET_N
    verdict, margin_gib = "UNKNOWN-no-target", None
    if target_gib and target_gib > 0:
        margin_gib = target_gib - proj_full / (1024 ** 3)
        frac = proj_full / (1024 ** 3) / target_gib
        verdict = "FIT" if frac < 0.8 else ("BORDERLINE" if frac < 1.0 else "DOES_NOT_FIT")
    report = {
        "n_sampled": n, "requested": n_samples, "stride": step,
        "per_split_sampled": {s: per_split.get(s, {"n": 0})["n"] for s in ("train", "val", "test")},
        "true_len": {"min": int(tls_arr.min()), "median": float(np.median(tls_arr)),
                     "mean": float(tls_arr.mean()), "p90": float(np.quantile(tls_arr, 0.90)),
                     "p95": float(np.quantile(tls_arr, 0.95)), "max": int(tls_arr.max())},
        "mean_zero_padding_fraction": float(1.0 - tls_arr.mean() / MAX_FRAMES),
        "uncompressed_bytes": n_bytes, "compressed_bytes": c_bytes,
        "compression_ratio": ratio, "avg_compressed_bytes_per_sample": avg_c,
        # padding vs data attribution (exact byte counts, both lossless):
        "unpadded_bytes": v_bytes, "unpadded_compressed_bytes": vc_bytes,
        "padding_bytes": n_bytes - v_bytes,
        "dense_data_compression_ratio": v_bytes / max(1, vc_bytes),
        "projected_compressed_gib_full_123779": proj_full / (1024 ** 3),
        "projected_compressed_gib_by_split": {s: v / (1024 ** 3) for s, v in proj.items()},
        "split_counts": {s: len(split_uids.get(s, [])) for s in ("train", "val", "test")},
        "persistence_target_gib": target_gib if target_gib > 0 else None,
        "safety_margin_gib": margin_gib, "verdict": verdict,
        "fetch_prep_per_sample_s": fetch_t / max(1, n),
        "zstd_level": ZSTD_LEVEL, "representation": "fixed-160x1629-fp32+zstd-lossless",
        "elapsed_s": time.perf_counter() - t0,
    }
    tmp = out / (MEASURE_REPORT + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out / MEASURE_REPORT)
    log(f"measure: n={n} true_len median={report['true_len']['median']:.0f} mean={report['true_len']['mean']:.1f} "
        f"ratio={ratio:.2f}x avg={avg_c / 1024:.0f} KiB/sample "
        f"-> projected {proj_full / (1024 ** 3):.1f} GiB / 123,779 samples; verdict={verdict}")
    return report


def check_cache_budget(projected_gib: float, budget_gib: float, cache_dir: Path) -> None:
    """Refuse unsafe materialization BEFORE hours burn. Raises SystemExit when
    the projection exceeds the configured budget or 90% of free disk.
    Pure function of (projection, budget, free-bytes) -- unit-tested, and the
    nearly-full-disk case is exercised by monkeypatching shutil.disk_usage."""
    free_gib = shutil.disk_usage(cache_dir if cache_dir.exists()
                                 else cache_dir.parent).free / (1024 ** 3)
    if budget_gib and projected_gib > budget_gib:
        raise SystemExit(f"cache REFUSED: projected {projected_gib:.1f} GiB > "
                         f"--cache-budget-gib={budget_gib:.1f} (partial cache kept; "
                         f"use --data-mode stream or a smaller --subset)")
    if projected_gib > 0.9 * free_gib:
        raise SystemExit(f"cache REFUSED: projected {projected_gib:.1f} GiB > 90% of "
                         f"free disk ({free_gib:.1f} GiB free)")
    log(f"cache budget: projected {projected_gib:.1f} GiB within budget "
        f"{budget_gib:.1f} GiB and free disk {free_gib:.1f} GiB")


def update_best_state(best: float, bad: int, val_loss: float, patience: int,
                      tol: float = 1e-4) -> tuple[float, int, bool, bool]:
    """Pure best-loss/patience transition for one validation epoch.

    Returns (new_best, new_bad, improved, should_stop). Non-finite val_loss
    never improves (best preserved, patience increments). A finite first
    loss replaces non-finite best (e.g. initial inf). `tol` matches the
    training loop's improvement margin so unit tests pin the exact behavior.
    """
    improved = (
        math.isfinite(val_loss)
        and (not math.isfinite(best) or val_loss < best - tol)
    )
    if improved:
        return float(val_loss), 0, True, False
    new_bad = bad + 1
    return best, new_bad, False, new_bad >= patience


def check_checkpoint_not_collapsed(payload: dict, path) -> None:
    """Reject explicit collapse evidence, including older pre-flag checkpoint history.

    An unchecked epoch (notably epoch 1) remains loadable, not certified healthy.
    Only the saved epoch matters; a directory marker may describe a later failure.
    """
    epoch = payload.get("epoch", 0)
    row = next((r for r in reversed(payload.get("history") or [])
                if r.get("epoch") == epoch), {})
    old_guard_failed = ("diversity_checked" not in row and epoch >= 2
                        and row.get("unique_fraction", 1.0) < DIVERSITY_FLOOR)
    if (payload.get("collapsed") or row.get("collapsed") or old_guard_failed
            or (row.get("diversity_checked") and row.get("diversity_pass") is False)):
        raise SystemExit(f"refusing explicitly collapsed checkpoint {path}; "
                         "use a recovery/best checkpoint from an earlier accepted epoch")


def init_token_embedding(model, mode: str) -> None:
    """Initialize decoder embeddings for a bounded scale-control experiment.

    `scaled` draws N(0, d_model**-0.5), with the padding row zero, to offset
    decode's sqrt(d_model) multiplier. `legacy` leaves nn.Embedding's default
    initialization intact. This changes initial scale, not proven collapse causality
    or translation quality. Keys/shapes and the untied output head are unchanged.
    Apply before loading any checkpoint so saved weights always take precedence.
    """
    if mode == "legacy":
        return
    if mode != "scaled":
        raise ValueError(f"unknown embed init mode {mode!r}")
    with torch.no_grad():
        nn.init.normal_(model.tok_embed.weight, mean=0.0, std=model.tok_embed.embedding_dim ** -0.5)
        if model.tok_embed.padding_idx is not None:
            model.tok_embed.weight[model.tok_embed.padding_idx].zero_()


def unigram_baseline_nll(train_ids, val_ids, vocab_size: int,
                         add_k=(1.0, 0.3, 0.1, 0.03, 0.01)) -> float:
    """Unsmoothed val NLL (nats/token) of the best add-k unigram model fit on train targets.

    Beating this pose-independent reference is diagnostic, not proof of pose use or
    translation quality. Inputs are token-id sequences WITH BOS/EOS; targets are
    ids[1:] (same shift as collate_sign_text). Return the minimum over add_k."""
    def targets(seqs):
        return np.concatenate([np.asarray(s, dtype=np.int64)[1:] for s in seqs])
    tr, va = targets(train_ids), targets(val_ids)
    cnt = np.bincount(tr, minlength=vocab_size).astype(np.float64)
    best = float("inf")
    for k in add_k:
        q = (cnt + k) / (cnt.sum() + k * vocab_size)
        best = min(best, float(-np.log(q[va]).mean()))
    return best


# Throughput micro-benchmarks + quota feasibility gate (all bounded, all labeled).
BENCH_TRAIN_STEPS = 20
BENCH_DECODE_SAMPLES = 256
# Overfit-gate step cap. The gate loop breaks early on pass, so this only
# prices the worst case in compute_feasibility (gate upper bound).
GATE_STEPS = 1200
FEASIBILITY_REPORT = "feasibility.json"


def benchmark_training_throughput(device, vocab_size: int, batch: int,
                                  embed_init: str = "scaled") -> dict:
    """Temporary real-config model; preserve training RNG and run_epoch quality stats."""
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=devices):
        t0 = time.perf_counter()
        model = SignTransformer(MODEL_INPUT_DIM, vocab_size, MAX_FRAMES, MAX_TOKENS,
                                pad_idx=0, d_model=D_MODEL, nhead=NHEAD, layers=NLAYERS,
                                ff=FF, dropout=DROPOUT, norm_first=NORM_FIRST)
        init_token_embedding(model, embed_init)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
        crit = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=LABEL_SMOOTHING)
        frames = torch.randn(batch, MAX_FRAMES, LANDMARK_DIM, device=device)
        lengths = torch.full((batch,), MAX_FRAMES, device=device, dtype=torch.long)
        tin = torch.randint(4, vocab_size, (batch, 32), device=device)
        tout = torch.randint(4, vocab_size, (batch, 32), device=device)
        amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        model.train()
        for _ in range(BENCH_TRAIN_STEPS):
            opt.zero_grad(set_to_none=True)
            with amp:
                loss = crit(model(frames, padding_mask(lengths, frames.size(1)), tin,
                                  causal_mask(tin.size(1), device)).reshape(-1, vocab_size),
                            tout.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        dt = time.perf_counter() - t0
    return {"train_steps_per_sec": BENCH_TRAIN_STEPS / dt, "device": str(device),
            "embed_init": embed_init,
            "note": "synthetic batch, temporary model; excludes data loading"}


# Local-cache read rate for cache-mode feasibility. Measured 2026-10-08 on
# 160-frame worst-case blobs through the full V2 path (shard ZIP read +
# pose parse + normalize + resize): ~158 samples/sec cold (6.3 ms/sample).
# The default below is ~3x slower (conservative upper bound); refresh_*
# replaces it with a real probe of the complete pilot cache whenever one
# exists. Never use the remote 1.18 sec/sample rate for local cached reads.
DEFAULT_LOCAL_FETCH_S = 0.02


def measure_local_fetch_s(store: PoseSource, uids: list, n: int = 32) -> float:
    """Tiny local benchmark: mean load_pose_item() sec/sample over a stride
    sample of at most n uids from an already-materialized cache. No network;
    sub-second for n=32. Returns DEFAULT_LOCAL_FETCH_S when empty/timed-out."""
    if not uids:
        return DEFAULT_LOCAL_FETCH_S
    step = max(1, len(uids) // max(1, n))
    picked = list(uids[::step])[:max(1, n)]
    t0 = time.perf_counter()
    for u in picked:
        load_pose_item(str(u), store)
    dt = time.perf_counter() - t0
    return dt / max(1, len(picked)) if dt > 0 else DEFAULT_LOCAL_FETCH_S


def compute_feasibility(meas: dict, n_train: int, n_val: int, n_test: int,
                        batch: int, epochs: int, prep_h: float, tok_h: float,
                        quota_h: float, target_h: float,
                        data_mode: str = "stream",
                        local_fetch_s: float | None = None) -> dict:
    """Project the full experiment from MEASURED rates. Conservative: data and
    compute are summed (no overlap assumed); gate is an upper bound (assumes
    never breaks early within GATE_STEPS); val forward-only assumed 3x
    train-step rate. Pure function -- unit-tested.

    data_mode="stream": every epoch re-reads remotely at fetch_prep_per_sample_s.
    data_mode="cache": prepare pays remote fetch ONCE; gate/train/val/diversity/
    eval read from the local shard cache at local_fetch_s (probe-measured, else
    DEFAULT_LOCAL_FETCH_S). The overfit gate loads its 64 samples ONCE
    (SignToTextDataset memoize=True) in both modes -- never 600x64 fetches."""
    fetch_s = meas["fetch_prep_per_sample_s"]
    steps_s = meas["benchmarks"]["train_steps_per_sec"]
    dec_s = meas["benchmarks"]["decode_hyps_per_sec"]
    if data_mode == "cache":
        try:
            local_s = float(local_fetch_s) if local_fetch_s is not None else DEFAULT_LOCAL_FETCH_S
            if not (math.isfinite(local_s) and local_s > 0):
                raise ValueError
        except (TypeError, ValueError):
            local_s = DEFAULT_LOCAL_FETCH_S
        data_s = local_s
    else:
        data_mode = "stream"
        local_s = None
        data_s = fetch_s
    train_steps = math.ceil(n_train / batch)
    val_steps = math.ceil(n_val / batch)
    epoch_data_h = (n_train + n_val) * data_s / 3600
    epoch_comp_h = (train_steps + val_steps / 3) / max(1e-9, steps_s) / 3600
    epoch_h = epoch_data_h + epoch_comp_h
    gate_h = (64 * data_s + GATE_STEPS * 4 / max(1e-9, steps_s)) / 3600
    div_h = 512 / max(1e-9, dec_s) / 3600
    eval_h = n_test / max(1e-9, dec_s) / 3600 + 0.2
    total_h = prep_h + tok_h + gate_h + epochs * (epoch_h + div_h) + eval_h + 0.5
    verdict = "PASS" if total_h <= target_h else "FAIL"
    measured = {"fetch_prep_per_sample_s": fetch_s,
                "train_steps_per_sec": steps_s,
                "decode_hyps_per_sec": dec_s,
                "data_mode": data_mode}
    assumptions = ["no data/compute overlap (conservative sum)",
                   "val forward-only = 3x train-step rate",
                   "gate upper bound: full GATE_STEPS checks, no early break; "
                   "gate/diversity loaders memoize (64/512 fetches once total)",
                   "decode rate measured on random weights (optimistic for eval)"]
    if data_mode == "cache":
        measured["local_fetch_per_sample_s"] = local_s
        assumptions.append(
            "cache mode: remote fetch paid ONCE in prepare; post-prepare "
            "gate/train/val/diversity/eval read local shards at "
            "local_fetch_per_sample_s (probed, else conservative default)")
    return {
        "quota_hours": quota_h, "target_hours": target_h, "verdict": verdict,
        "projected_total_hours": total_h,
        "breakdown_hours": {"prepare": prep_h, "tokenizer": tok_h, "overfit_gate_ub": gate_h,
                            "per_epoch": epoch_h, "per_epoch_diversity": div_h,
                            "epochs": epochs, "eval": eval_h, "slack": 0.5},
        "measured_inputs": measured,
        "assumptions": assumptions,
    }


def feasibility_banner(feas: dict, n_usable: int, data_mode: str) -> str:
    b = feas["breakdown_hours"]
    mi = feas["measured_inputs"]
    lines = ("================================",
             "V2 RESOURCE FEASIBILITY",
             "================================",
             f"Remaining Kaggle budget: {feas['quota_hours']:.2f}h",
             f"Target maximum runtime: {feas['target_hours']:.2f}h",
             f"Dataset: {n_usable:,} usable samples",
             f"Cache mode: {'FULL' if data_mode == 'cache' else 'STREAMING (FULL CACHE DISABLED)'}",
             f"Preprocessing: {1 / max(1e-9, mi['fetch_prep_per_sample_s']):.1f} samples/sec",
             f"Fetch+prep: {mi['fetch_prep_per_sample_s']:.3f} sec/sample",
             f"Training: {mi['train_steps_per_sec']:.2f} steps/sec")
    if data_mode == "cache" and mi.get("local_fetch_per_sample_s") is not None:
        lines += (f"Local cache read: {mi['local_fetch_per_sample_s'] * 1000:.1f} ms/sample "
                  f"(gate/train/val read local shards, no HTTP)",)
    return ("\n".join(lines) +
            f"\nEstimated epoch: {b['per_epoch']:.2f} hours\n"
            f"Estimated total ({b['epochs']} epochs): {feas['projected_total_hours']:.2f} hours\n"
            f"FEASIBILITY: {feas['verdict']}\n"
            "================================")


def refresh_feasibility_after_cache(out: Path, n_train: int, n_val: int, n_test: int,
                                      batch: int, epochs: int,
                                      quota_h: float, target_h: float,
                                      pose_cache: Path | None = None) -> dict | None:
    """Cache-mode measure writes projected_total_hours=NaN (prepare cost unknowable
    before the 200-sample projection). Once the cache is COMPLETE the prepare cost
    is sunk, and gate/train/val/diversity/eval read LOCAL shards (no HTTP), so
    recompute with prep_h=0 and a probed local-cache read rate, then overwrite
    feasibility.json. Returns the new feas, or None when no usable measurement exists."""
    try:
        rep = json.loads((out / MEASURE_REPORT).read_text())
        bench = rep["benchmarks"]
        float(rep["fetch_prep_per_sample_s"])
        float(bench["train_steps_per_sec"])
        float(bench["decode_hyps_per_sec"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    tok_h = float(rep.get("tokenizer_hours", 0.0) or 0.0)
    cache_dir = Path(pose_cache) if pose_cache else (out / "pose_cache")
    local_s = DEFAULT_LOCAL_FETCH_S
    local_note = f"local-cache default {DEFAULT_LOCAL_FETCH_S:.3f}s/sample"
    try:
        probe_store = ShardPoseSource(str(cache_dir))
        probe_uids = list(probe_store.index)[:64]
        if probe_uids:
            local_s = measure_local_fetch_s(probe_store, probe_uids, n=32)
            local_note = (f"probed local cache {local_s * 1000:.1f} ms/sample "
                          f"({len(probe_uids)} uids, {cache_dir.name})")
    except (OSError, ValueError, RuntimeError):
        pass
    feas = compute_feasibility(rep, n_train, n_val, n_test, batch, epochs,
                               0.0, tok_h, quota_h, target_h,
                               data_mode="cache", local_fetch_s=local_s)
    feas["note"] = ("recomputed after complete local cache: prepare cost sunk (0h); "
                    + local_note)
    tmp = out / (FEASIBILITY_REPORT + ".tmp")
    tmp.write_text(json.dumps(feas, indent=1))
    os.replace(tmp, out / FEASIBILITY_REPORT)
    return feas


def apply_subset(df_tr, df_va, df_te, subset: int, seed: int):
    """Deterministic pilot subset: uniform random rows within each split.

    The splits themselves are untouched (same video sets from the seed-42
    video_split; no video crosses splits); only row counts shrink to at most
    `subset` per split. Draws are without replacement from a seeded RNG, so
    identical inputs always yield identical subsets. Pure function --
    unit-tested for determinism, exact counts, and split containment."""
    out = []
    rng = np.random.default_rng(seed)
    for name, frame in (("train", df_tr), ("val", df_va), ("test", df_te)):
        n = min(subset, len(frame))
        if n < len(frame):
            idx = np.sort(rng.choice(len(frame), size=n, replace=False))
            frame = frame.iloc[idx].reset_index(drop=True)
        log(f"SUBSET MODE: {name} capped to {n} rows via deterministic random sampling "
            f"(pilot run only -- NOT the full V2 experiment; representation unchanged)")
        out.append(frame)
    return out[0], out[1], out[2]


def _open_pose_store(kind: str, arg: str) -> PoseSource:
    """Open a fresh pose store; each DataLoader worker gets its own (fork-safe)."""
    if kind == "remote":
        return RemotePoseSource(arg)
    if kind == "zip":
        return ZipPoseSource(arg)
    if kind == "shards":
        return ShardPoseSource(arg)
    return DirPoseSource(arg)


def preflight_pose_source(store: PoseSource, uid: str) -> dict:
    """Cheap accessibility probe: EOCD + central directory already read on open;
    here read ONE real member end-to-end (bytes -> normalized frames)."""
    names = sorted(store.names) if hasattr(store, "names") else []
    raw = store.read_member_bytes(uid)
    if not raw:
        raise RuntimeError(f"preflight: empty pose payload for {uid}")
    frames, true_len = load_pose_item(uid, store)
    report = {"members": len(names), "sample_members": names[:5], "probe_uid": uid,
              "probe_bytes": len(raw), "frames": [int(x) for x in frames.shape],
              "true_len": int(true_len)}
    log(f"pose preflight OK: {len(names):,} members; probe {uid}: {len(raw):,} bytes, frames={frames.shape}")
    return report


def verify_pose_layout(pose, uid: str) -> None:
    """Fail loudly if a .pose file's header components do not start with the
    canonical COMPONENT_ORDER. The reader below slices the first 543 points
    positionally, so a reordered header would silently mistrain -- rejected
    here instead."""
    names = [c.name for c in pose.header.components]
    if names[: len(COMPONENT_ORDER)] != list(COMPONENT_ORDER):
        raise ValueError(
            f"Pose header component order {names} for {uid} does not start with "
            f"expected {list(COMPONENT_ORDER)} -- refusing to slice positionally"
        )
    total = sum(len(c.points) for c in pose.header.components)
    if total < TOTAL_LANDMARKS:
        raise ValueError(
            f"Pose header has only {total} points for {uid}, need {TOTAL_LANDMARKS}"
        )


def read_normalized_pose(uid: str, store: PoseSource) -> np.ndarray:
    """Raw .pose -> (frames, 1629) via the canonical pose_normalize_v1 path."""
    from pose_format import Pose

    pose = Pose.read(store.read_member_bytes(uid))
    verify_pose_layout(pose, uid)
    data = np.asarray(np.ma.filled(pose.body.data, 0.0), dtype=np.float32)
    if data.ndim != 4 or data.shape[1] < 1:
        raise ValueError(f"Unexpected pose tensor {data.shape} for {uid}")
    flat = data[:, 0, :TOTAL_LANDMARKS, :].reshape(data.shape[0], -1).astype(np.float32)
    return normalize_pose_sequence(flat)


def load_pose_item(uid: str, store: PoseSource) -> tuple[np.ndarray, int]:
    raw = read_normalized_pose(uid, store)[:, FULL_FEATURE_MASK]
    true_len = min(len(raw), MAX_FRAMES)
    return resize_sequence(raw, MAX_FRAMES).astype(np.float32), true_len


# --------------------------------------------------------------------------
# Tokenizer (same procedure as 352868924: SP unigram 16k on train text only)
# --------------------------------------------------------------------------

def train_tokenizer(train_texts: list[str], out_dir: Path, vocab_size: int = 16000):
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "train_text.txt"
    with open(corpus, "w", encoding="utf-8") as fh:
        for text in train_texts:
            fh.write(normalize_text(text).replace("\n", " ") + "\n")
    prefix = out_dir / "signsage_en_sp"
    model_path = prefix.with_suffix(".model")
    if not model_path.exists():
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(prefix), vocab_size=vocab_size,
            model_type="unigram", character_coverage=1.0,
            normalization_rule_name="nmt_nfkc_cf",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
            hard_vocab_limit=False, shuffle_input_sentence=False, input_sentence_size=0,
            seed_sentencepiece_size=1000000, num_threads=max(1, min(8, os.cpu_count() or 1)),
        )
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    assert (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) == (0, 1, 2, 3)
    return sp


def encode_text(sp, text: str, max_len: int = MAX_TOKENS) -> list[int]:
    ids = sp.encode(normalize_text(text), out_type=int, add_bos=True, add_eos=True)
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [sp.eos_id()]
    return ids


def decode_ids(sp, ids) -> str:
    clean = []
    for idx in ids:
        idx = int(idx)
        if idx == sp.eos_id():
            break
        if idx not in (sp.bos_id(), sp.pad_id()):
            clean.append(idx)
    return normalize_text(sp.decode(clean)) if clean else ""


def _sp_fingerprint(sp, model_path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(model_path).read_bytes())
    h.update(str(sp.get_piece_size()).encode())
    h.update(f"{sp.pad_id()},{sp.unk_id()},{sp.bos_id()},{sp.eos_id()}".encode())
    return h.hexdigest()[:16]


def _uids_fingerprint(uids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(u) for u in uids).encode()).hexdigest()[:16]


def _texts_fingerprint(texts: np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(t) for t in texts).encode()).hexdigest()[:16]


def load_or_build_token_cache(sp, model_path: Path, tok_dir: Path,
                              splits: dict) -> dict:
    """Tokenize each split once; reuse tokens.npz on later sessions.

    splits: {name: (uids, texts)}. Returns {name: [np int64 arrays]} with the
    exact same ids encode_text() would produce (verified by fingerprint of
    the SP model + vocab + per-split uid order). Atomic write; corrupt cache
    is discarded and rebuilt, never trusted.
    """
    t0 = time.perf_counter()
    fp = _sp_fingerprint(sp, model_path)
    meta_path = tok_dir / TOKEN_CACHE_META
    npz_path = tok_dir / TOKEN_CACHE_NAME
    want = {name: {"sp": fp, "max_len": MAX_TOKENS, "n": len(uids),
                   "uids": _uids_fingerprint(uids), "texts": _texts_fingerprint(texts)}
            for name, (uids, texts) in splits.items()}
    try:
        if npz_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            if meta.get("splits") == want:
                z = np.load(npz_path, allow_pickle=False)
                out = {}
                for name, (uids, _) in splits.items():
                    mat, lens = z[f"{name}_ids"], z[f"{name}_len"]
                    assert len(mat) == len(uids) and len(lens) == len(uids)
                    out[name] = [np.asarray(mat[i, : lens[i]], dtype=np.int64) for i in range(len(uids))]
                log(f"token cache: reused {sum(len(v) for v in out.values()):,} ids "
                    f"from {npz_path.name} in {time.perf_counter() - t0:.1f}s (no re-tokenize)")
                return out
            log("token cache: fingerprint mismatch (tokenizer or splits changed) -- rebuilding")
    except Exception as exc:  # noqa: BLE001 -- corrupt cache is never trusted
        log(f"token cache: unreadable ({type(exc).__name__}) -- rebuilding")
    out, mats = {}, {}
    for name, (uids, texts) in splits.items():
        log(f"tokenizing {len(texts):,} {name} captions (one-time; cached to disk)...")
        ids = [np.asarray(encode_text(sp, t), dtype=np.int64) for t in texts]
        out[name] = ids
        width = max(len(a) for a in ids)
        mat = np.zeros((len(ids), width), dtype=np.int64)
        lens = np.zeros(len(ids), dtype=np.int64)
        for i, a in enumerate(ids):
            mat[i, : len(a)] = a
            lens[i] = len(a)
        mats[f"{name}_ids"], mats[f"{name}_len"] = mat, lens
    tmp = tok_dir / (TOKEN_CACHE_NAME + ".tmp")
    with open(tmp, "wb") as fh:  # file object: savez must not append .npz
        np.savez_compressed(fh, **mats)
    os.replace(tmp, npz_path)
    meta_tmp = tok_dir / (TOKEN_CACHE_META + ".tmp")
    meta_tmp.write_text(json.dumps({"splits": want}))
    os.replace(meta_tmp, meta_path)
    log(f"token cache: built {sum(len(v) for v in out.values()):,} ids in "
        f"{time.perf_counter() - t0:.1f}s -> {npz_path.stat().st_size / 1024:.0f} KiB")
    return out


# --------------------------------------------------------------------------
# Dataset / collate (same shift semantics as the verified baseline)
# --------------------------------------------------------------------------

class SignToTextDataset(Dataset):
    def __init__(self, uids: np.ndarray, texts: np.ndarray, sp, store: PoseSource | None,
                 store_spec: tuple | None = None, token_ids=None, memoize: bool = False):
        self.uids = uids
        self.sp = sp
        self.store = store
        self.store_spec = store_spec  # (kind, arg); workers open their own (fork-safe)
        self._worker_store = None
        # memoize=True pins per-uid frames in RAM: ONLY for small fixed loaders
        # (overfit gate: 64 samples; diversity sample: 512). Values are a pure
        # function of uid (no augmentation anywhere), so repeats return
        # bit-identical tensors while skipping remote re-fetch. NEVER enable
        # for full train/val/test loaders (would pin tens of GiB).
        self._memo: dict | None = {} if memoize else None
        if memoize and len(uids) > 1024:
            raise ValueError(f"memoize refused for {len(uids)} samples (RAM bound)")
        if token_ids is None:
            log(f"Pre-tokenizing {len(texts):,} captions...")
            self.token_ids = [np.asarray(encode_text(sp, t), dtype=np.int64) for t in texts]
        else:
            assert len(token_ids) == len(uids)
            self.token_ids = [np.asarray(a, dtype=np.int64) for a in token_ids]

    def __len__(self): return len(self.uids)

    def _active_store(self):
        if self.store_spec is None:
            return self.store
        try:
            from torch.utils.data import get_worker_info
            in_worker = get_worker_info() is not None
        except Exception:
            in_worker = False
        if not in_worker:
            if self.store is None:
                self.store = _open_pose_store(*self.store_spec)
            return self.store
        if self._worker_store is None:
            self._worker_store = _open_pose_store(*self.store_spec)
        return self._worker_store

    def __getitem__(self, idx):
        uid = str(self.uids[idx])
        if self._memo is not None and uid in self._memo:
            frames, length = self._memo[uid]
            return (torch.from_numpy(np.ascontiguousarray(frames)), torch.tensor(length),
                    torch.from_numpy(self.token_ids[idx].copy()).long())
        if self.store is None and self.store_spec is None:  # smoke mode: deterministic synthetic poses
            rng = np.random.default_rng(abs(hash(uid)) % (2 ** 31))
            frames = rng.normal(size=(MAX_FRAMES, LANDMARK_DIM)).astype(np.float32)
            length = MAX_FRAMES
        else:
            frames, length = load_pose_item(uid, self._active_store())
        if self._memo is not None:
            self._memo[uid] = (np.ascontiguousarray(frames), int(length))
        return (torch.from_numpy(np.ascontiguousarray(frames)), torch.tensor(length),
                torch.from_numpy(self.token_ids[idx].copy()).long())


def collate_sign_text(batch):
    frames, lengths, tokens = zip(*batch)
    frames = torch.stack(frames)
    lengths = torch.stack(lengths)
    max_tok = max(len(t) for t in tokens)
    tok_pad = torch.full((len(tokens), max_tok), 0, dtype=torch.long)
    for i, t in enumerate(tokens):
        tok_pad[i, :len(t)] = t
    return frames, lengths, tok_pad[:, :-1], tok_pad[:, 1:]


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def build_schedule(opt, total_steps: int):
    warmup = max(1, int(WARMUP_FRACTION * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class PoseRolledLoader:
    """Eval-only within-batch pose-roll control; targets stay in their original order.

    A positive NLL gap is evidence of pose sensitivity, not a quality verdict.
    Singleton batches are unchanged, and duplicate poses can weaken the control.
    """

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for frames, lengths, tin, tout in self.loader:
            yield frames.roll(1, 0), lengths.roll(1, 0), tin, tout


def run_epoch(model, loader, opt, crit, device, scheduler=None, train: bool = True,
              decoder_input_dropout: float = 0.0, vocab_size: int = 0,
              token_weighted: bool = False) -> float:
    """One pass over `loader`.

    decoder_input_dropout: probability of replacing a non-BOS decoder input
    token with a random vocab id in [4, vocab_size) during TRAINING only.
    This is a token-level exposure-bias mitigation (a simple form of
    scheduled sampling). Position 0 (BOS) is never perturbed. Validation
    and gate calls leave it at 0.0 so metric semantics are unchanged.

    token_weighted=True reports mean CE per non-ignored target token (including
    EOS), for comparison with unigram NLL. It requires an unweighted mean CE.
    Default False preserves the historical mean-of-batch-means loss reporting;
    neither mode changes the loss used for backpropagation.
    """
    if token_weighted and (not isinstance(crit, nn.CrossEntropyLoss)
                           or crit.reduction != "mean" or crit.weight is not None):
        raise ValueError("token_weighted requires unweighted mean CrossEntropyLoss")
    model.train(train)
    total, token_count = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    scaler = getattr(run_epoch, "_scaler", None)
    if train and scaler is None:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        run_epoch._scaler = scaler  # type: ignore[attr-defined]
    if train:
        opt.zero_grad(set_to_none=True)
    timer = PhaseTimer("train" if train else "val")
    gnorms: list = []
    with ctx:
        batch_iter = iter(loader)
        while True:
            t_data0 = time.perf_counter()
            try:
                frames, lengths, tin, tout = next(batch_iter)
            except StopIteration:
                break
            t_comp0 = time.perf_counter()
            frames, lengths, tin, tout = (t.to(device, non_blocking=True) for t in (frames, lengths, tin, tout))
            if (train and decoder_input_dropout > 0.0 and vocab_size > 4
                    and tin.size(1) > 1):
                # Exposure-bias mitigation. Replace each non-BOS decoder input
                # token with a uniform random vocab id in [4, vocab_size) with
                # probability `decoder_input_dropout`. Position 0 (BOS) is
                # pinned. Special ids 0..3 (PAD/UNK/BOS/EOS) are never sampled.
                _dm = torch.rand(tin.size(), device=tin.device) < decoder_input_dropout
                _dm[:, 0] = False
                _rand = torch.randint(4, vocab_size, tin.size(),
                                       device=tin.device, dtype=tin.dtype)
                tin = torch.where(_dm, _rand, tin)
            src_pad = padding_mask(lengths, frames.size(1))
            tgt_mask = causal_mask(tin.size(1), frames.device)
            with amp:
                logits = model(frames, src_pad, tin, tgt_mask)
                loss = crit(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            if train:
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                _gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
                gnorms.append(_gnorm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            if token_weighted:
                n_tokens = int(tout.ne(crit.ignore_index).sum().item())
                if n_tokens:
                    total += loss.item() * n_tokens
                    token_count += n_tokens
            else:
                total += loss.item()
            timer.add_batch(t_comp0 - t_data0, time.perf_counter() - t_comp0)
    log(timer.summary())
    if train and gnorms:
        with torch.no_grad():
            _pnorm = float(torch.sqrt(sum(p.detach().pow(2).sum()
                                          for p in model.parameters())))
        run_epoch.last_train_stats = {  # type: ignore[attr-defined]
            "grad_norm_mean": float(sum(gnorms) / len(gnorms)),
            "grad_norm_max": float(max(gnorms)),
            "grad_clip_frac": float(sum(g > GRAD_CLIP for g in gnorms) / len(gnorms)),
            "param_norm": _pnorm}
    if token_weighted and not token_count:
        raise ValueError("token-weighted NLL requires at least one non-ignored target")
    return total / max(1, token_count if token_weighted else len(loader))


@torch.no_grad()
def batch_greedy(model, loader, sp, device, max_decode: int = MAX_DECODE_TOKENS,
                 constrained: bool = True, return_ids: bool = False):
    """Constrained batched greedy decode (same bans as serving).

    Parity contract with backend/app/ml/inference/sign_to_text.py
    greedy_decode_ids(): per generated row, ban PAD/BOS/UNK, ban EOS for the
    first MIN_OUTPUT_TOKENS steps, apply ban_repeated_ngram with
    NO_REPEAT_NGRAM_SIZE, then the REPETITION_PENALTY divide-if->=0-else-multiply
    over set(ids[1:]), then log_softmax + argmax. Serving is single-sample;
    this is the batched equivalent -- covered by test_decode_parity in
    test_v2_data_workflow.py.

    constrained=False skips the EOS-minimum, no-repeat-ngram and repetition
    penalties (PAD/BOS/UNK bans stay: structural vocab hygiene). Used ONLY as
    a gate diagnostic alongside the authoritative constrained score, to tell
    constraint-blocked failures apart from optimization failures.

    return_ids=True additionally returns per-sample (ref_ids, gen_ids):
    ref_ids are the unpadded tout rows (no BOS, with EOS); gen_ids are the
    raw greedy rows (with EOS when emitted). Default False: existing
    3-tuple callers are untouched.
    """
    t0 = time.perf_counter()
    model.eval()
    refs, hyps, confs = [], [], []
    ref_ids_all, gen_ids_all = [], []
    pad, unk, bos, eos = sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()
    n_batches = 0
    for frames, lengths, tin, tout in loader:
        frames, lengths = frames.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
        src_pad = padding_mask(lengths, frames.size(1))
        memory = model.encode(frames, src_pad)
        batch_size = frames.size(0)
        ids = torch.full((batch_size, 1), bos, dtype=torch.long, device=device)
        histories = [[bos] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sum_logp = torch.zeros(batch_size, device=device)
        steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        gen = []
        for step in range(max_decode):
            logits = model.decode(ids, memory, causal_mask(ids.size(1), device), src_pad)[:, -1].float()
            nxt = torch.empty(batch_size, dtype=torch.long, device=device)
            prob = torch.empty(batch_size, device=device)
            for r in range(batch_size):
                if finished[r]:
                    nxt[r] = eos
                    prob[r] = 0.0
                    continue
                lg = logits[r].clone()
                lg[[pad, bos, unk]] = -torch.inf
                if constrained:
                    if step < MIN_OUTPUT_TOKENS:
                        lg[eos] = -torch.inf
                    lg = ban_repeated_ngram(lg, histories[r], NO_REPEAT_NGRAM_SIZE)
                    for tid in set(histories[r][1:]):
                        lg[tid] = (lg[tid] / REPETITION_PENALTY
                                   if lg[tid] >= 0 else lg[tid] * REPETITION_PENALTY)
                lp = torch.log_softmax(lg, dim=-1)
                p, t = torch.max(lp, dim=-1)
                prob[r], nxt[r] = p, t
            active = ~finished
            sum_logp += torch.where(active, prob, torch.zeros_like(prob))
            steps += active.long()
            nxt = torch.where(finished, torch.full_like(nxt, eos), nxt)
            gen.append(nxt.detach().cpu().numpy())
            for r in range(batch_size):
                if not finished[r]:
                    histories[r].append(int(nxt[r]))
            finished |= nxt.eq(eos)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            if finished.all():
                break
        gen = np.stack(gen, axis=1)
        conf = torch.exp(sum_logp / steps.clamp(min=1)).detach().cpu().numpy()
        for r, row in enumerate(tout.cpu().tolist()):
            refs.append(decode_ids(sp, row))
            hyps.append(decode_ids(sp, gen[r].tolist()))
            confs.append(float(conf[r]))
            if return_ids:
                ref_ids_all.append([int(x) for x in row if int(x) != pad])
                gen_ids_all.append([int(x) for x in gen[r].tolist()])
        n_batches += 1
    log(f"decode: {len(hyps):,} hyps in {time.perf_counter() - t0:.1f}s ({n_batches} batches)")
    if return_ids:
        return refs, hyps, confs, ref_ids_all, gen_ids_all
    return refs, hyps, confs


from evaluation_metrics import (
    compute_metrics,
    diversity_metrics,
    per_class_recall,
    build_trivial_baseline,
    evaluate_gate,
)


# --------------------------------------------------------------------------
# Export (same bundle layout the backend already loads)
# --------------------------------------------------------------------------

def export_bundle(model, sp, out_dir: Path, run_id: str, train_cfg: dict,
                  eval_report: dict, contract_src: Path | None) -> Path:
    bundle = out_dir / "signsage_deployment_bundle"
    (bundle / "checkpoint").mkdir(parents=True, exist_ok=True)
    (bundle / "tokenizer").mkdir(parents=True, exist_ok=True)
    is_pilot = bool(train_cfg.get("is_pilot", False))
    is_smoke = bool(train_cfg.get("is_smoke", False))
    mc = {
        "input_dim": MODEL_INPUT_DIM, "raw_feature_dim": LANDMARK_DIM,
        "vocab_size": sp.get_piece_size(),
        "d_model": D_MODEL, "nhead": NHEAD, "layers": NLAYERS, "ff": FF,
        "dropout": DROPOUT, "norm_first": NORM_FIRST,
        "max_frames": MAX_FRAMES, "max_tokens": MAX_TOKENS,
    }
    if is_pilot:
        mc["is_pilot"] = True
        mc["subset"] = train_cfg.get("subset")
        mc["subset_seed"] = train_cfg.get("subset_seed")
        mc["note"] = "PILOT / PIPELINE VALIDATION ONLY -- NOT FOR PRODUCTION"
    mc["training_scope"] = (train_cfg.get("data_counts") or {}).get("scope", "unknown")
    if is_smoke:
        mc["is_smoke"] = True
        mc["note"] = "SMOKE TEST ON SYNTHETIC DATA -- NOT A TRAINED MODEL"
    _atomic_torch_save({"state_dict": model.state_dict(), "model_config": mc,
                        "initialization_config": train_cfg.get("initialization_config"),
                        "vocab": {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}},
                       bundle / "checkpoint" / "signsage_transformer_full.pt")
    shutil.copy(sp_train_artifacts["model"], bundle / "tokenizer" / "signsage_en_sp.model")
    if contract_src and contract_src.is_file():
        shutil.copy(contract_src, bundle / "landmark_contract.json")
    else:
        bundle.joinpath("landmark_contract.json").write_text(json.dumps({
            "total_landmark_count": TOTAL_LANDMARKS, "coordinates_per_landmark": COORDINATES_PER_LANDMARK,
            "raw_feature_dim": LANDMARK_DIM,
            "component_order": ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
            "component_ranges": {"POSE_LANDMARKS": [0, 33], "FACE_LANDMARKS": [33, 501],
                                 "LEFT_HAND_LANDMARKS": [501, 522], "RIGHT_HAND_LANDMARKS": [522, 543]},
            "normalization_version": "pose_normalize_v1", "max_frames": MAX_FRAMES,
            "final_model_input_dim": MODEL_INPUT_DIM, "full_feature_mask": FULL_FEATURE_MASK,
        }, indent=1))
    tokenizer_desc = ("SentencePiece unigram trained on FULL training corpus (subset mode: tokenizer on full, "
                      "pose data on subset)" if is_pilot else "SentencePiece unigram trained on training targets only")
    bundle.joinpath("sign_to_text_deployment.json").write_text(json.dumps({
        "project": "SignSage", "task": "ISL_to_English", "dataset": "iSign v1.1",
        "feature_dim": LANDMARK_DIM, "model_input_dim": MODEL_INPUT_DIM,
        "max_frames": MAX_FRAMES, "max_tokens": MAX_TOKENS, "vocab_size": sp.get_piece_size(),
        "tokenizer": tokenizer_desc,
        "tokenizer_model": "tokenizer/signsage_en_sp.model",
        "special_token_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        "normalization_version": "pose_normalize_v1",
        "model_architecture": {"input_dim": MODEL_INPUT_DIM, "raw_feature_dim": LANDMARK_DIM,
                               "vocab_size": sp.get_piece_size(), **{k: mc[k] for k in
                               ("d_model", "nhead", "layers", "ff", "dropout", "norm_first",
                                "max_frames", "max_tokens")}},
        "decoding": {"decoder_runtime": "constrained_greedy", "max_decode_tokens": MAX_DECODE_TOKENS,
                     "min_output_tokens": MIN_OUTPUT_TOKENS,
                     "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE, "repetition_penalty": REPETITION_PENALTY},
        "training_config": train_cfg,
        "checkpoint": "checkpoint/signsage_transformer_full.pt",
    }, indent=1))
    bundle.joinpath("evaluation_results.json").write_text(json.dumps(eval_report, indent=1))
    manifest = {
        "model_id": "signsage-s2t", "run_id": run_id, "architecture": "spatial-temporal-transformer",
        "predecessor": "352868924 (NO-GO baseline; results preserved, not overwritten)",
        "dataset": {"source": "iSign v1.1 (Hugging Face)", **train_cfg.get("data_counts", {})},
        "metrics": eval_report.get("test_metrics", {}),
        "diversity": eval_report.get("diversity", {}),
        "gate": eval_report.get("gate", {}),
        "status": ("SMOKE TEST ON SYNTHETIC DATA -- NOT A TRAINED MODEL"
                   if train_cfg.get("is_smoke") else
                   "PILOT / PIPELINE VALIDATION ONLY -- NOT FOR PRODUCTION"
                   if train_cfg.get("is_pilot") else
                   "unpromoted — set SIGNSAGE_MODEL_PROMOTED=true only after the quality gate passes"),
    }
    if train_cfg.get("is_pilot"):
        manifest["subset"] = train_cfg.get("subset")
        manifest["subset_seed"] = train_cfg.get("subset_seed")
        manifest["note"] = train_cfg.get("note")
    out_dir.joinpath(f"signsage-s2t_v{run_id}.yaml").write_text(
        "# SignSage run manifest (generated by train_sign_to_text.py)\n"
        + json.dumps(manifest, indent=1) + "\n")
    return bundle


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

sp_train_artifacts: dict = {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SignSage v2 sign-to-text training")
    ap.add_argument("--csv", default=None, help="path to iSign_v1.1.csv")
    ap.add_argument("--pose-source", default=None,
                    help="local .pose dir, local .zip, or 'remote' (HF streaming, needs HF_TOKEN)")
    ap.add_argument("--pose-cache", default=None,
                    help="local materialized pose cache dir (default: <out>/pose_cache). "
                         "Built once from --pose-source, then reused; training reads from it.")
    ap.add_argument("--shard-samples", type=int, default=DEFAULT_SHARD_SAMPLES,
                    help="pose samples per cache shard file")
    ap.add_argument("--verify-every", type=int, default=DEFAULT_VERIFY_EVERY,
                    help="full shape/dtype verify every N materialized samples")
    ap.add_argument("--max-cache-gb", type=float, default=0.0,
                    help="graceful stop if the pose cache would exceed this size (0 = no cap)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel pose fetch workers for the prepare stage (1 = sequential; "
                         "measured 1.2 samples/s sequential on Kaggle, ~jobs x with 8)")
    ap.add_argument("--subset", type=int, default=0,
                    help="pilot mode: deterministic random sample of at most N rows within each "
                         "of train/val/test (0 = full data). "
                         "Loudly logged and recorded in dataset_audit.json; splits, videos, "
                         "representation unchanged, just fewer rows.")
    ap.add_argument("--measure-samples", type=int, default=2000,
                    help="items sampled by the measure stage (deterministic stride; default 2000)")
    ap.add_argument("--persistence-target-gib", type=float, default=0.0,
                    help="persistence budget the measure verdict is computed against. "
                         "0 = unknown: verdict stays UNKNOWN-no-target. Never hardcoded: "
                         "pass a value measured on the Kaggle host (df -h, trial saves).")
    ap.add_argument("--data-mode", choices=("stream", "cache"), default="stream",
                    help="stream: train directly from --pose-source, no materialization "
                         "(FULL CACHE DISABLED; default). cache: materialize pose_cache/ "
                         "once under a disk budget, then train from disk.")
    ap.add_argument("--cache-budget-gib", type=float, default=10.0,
                    help="max pose-cache size in cache mode; projected overruns refuse "
                         "before fetching (subset pilots fit, full 143 GiB does not).")
    ap.add_argument("--quota-hours", type=float, default=13.0 + 34.0 / 60.0,
                    help="remaining Kaggle quota the feasibility gate budgets against.")
    ap.add_argument("--target-hours", type=float, default=11.5,
                    help="max projected runtime for a PASS verdict (quota minus safety margin).")
    ap.add_argument("--force", action="store_true",
                    help="override a FAIL feasibility verdict (loudly logged; quota WILL burn).")
    ap.add_argument("--contract", default="research/evaluation/reports/352868924/landmark_contract.json",
                    help="landmark contract to copy into the bundle (representation unchanged)")
    ap.add_argument("--out", required=True, help="output dir for checkpoints/bundle/report")
    ap.add_argument("--stages", default="all",
                    help="comma subset of: audit,tokenizer,prepare,measure,overfit,train,evaluate,export "
                         "(measure is opt-in only, never in 'all')")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=PEAK_LR)
    ap.add_argument("--decoder-input-dropout", type=float, default=0.0,
                    help="Token-level dropout on decoder inputs during training: with this "
                         "probability, each non-BOS decoder input token is replaced by a "
                         "uniform random vocab id in [4, vocab_size). Experimental prefix "
                         "perturbation, not a proven collapse fix. 0.0 disables (original behavior). "
                         "Validation and gate paths are unaffected.")
    ap.add_argument("--embed-init", choices=("scaled", "legacy"), default="scaled",
                    help="decoder token-embedding init. scaled (default): N(0, d_model**-0.5), matching "
                         "the model's *sqrt(d_model) scaling. legacy: nn.Embedding default N(0,1) "
                         "(for initialization-scale A/B comparisons).")
    ap.add_argument("--max-hours", type=float, default=8.0, help="graceful stop budget")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--preflight-only", action="store_true",
                    help="verify pose-source accessibility (one real member) then exit before expensive stages")
    ap.add_argument("--smoke", action="store_true", help="CPU plumbing check on synthetic data")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    import pandas as pd

    args = parse_args(argv)
    stages = set(args.stages.split(",")) if args.stages != "all" else {"audit", "tokenizer", "prepare", "overfit", "train", "evaluate", "export"}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "train" in stages:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"run {run_id} on {device}; stages={sorted(stages)}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log(f"GPU detected: {gpu_name} ({gpu_mem_gb:.1f} GB VRAM); AMP mixed-precision: float16 enabled")
    else:
        log("GPU: None detected (running on CPU; AMP disabled)")
    if args.decoder_input_dropout > 0:
        log(f"decoder input dropout: {args.decoder_input_dropout:.3f} "
            f"(exposure-bias mitigation; training only)")

    # ---- data ----
    if args.smoke:
        log("SMOKE mode: synthetic poses + tiny vocab targets (no dataset).")
        vocab_texts = [f"hello world {i}" for i in range(8)]
        df_tr = pd.DataFrame({"uid": [f"smoke-{i}" for i in range(8)], "text": vocab_texts})
        df_va = df_tr.iloc[:4].reset_index(drop=True)
        df_te = df_tr.iloc[:4].reset_index(drop=True)
        _rows_available = {"train": len(df_tr), "val": len(df_va), "test": len(df_te)}
        _usable_rows = None
        _scope = "smoke_synthetic"
        store = None
        store_spec = None
    else:
        if not args.csv:
            raise SystemExit("--csv is required (or use --smoke)")
        import pandas as pd  # noqa: F811

        df = pd.read_csv(args.csv)
        cleaned, cleaning = clean_dataframe(df)
        if cleaning["duplicate_uid"]:
            raise SystemExit("duplicate UIDs after cleaning -- refusing.")
        df_tr_all, df_va_all, df_te_all, split = video_split(cleaned, args.seed)
        df_tr, df_va, df_te = df_tr_all, df_va_all, df_te_all
        _rows_available = {"train": len(df_tr_all), "val": len(df_va_all), "test": len(df_te_all)}
        _usable_rows = int(cleaning["usable_rows"])
        _scope = "pilot_subset" if (args.subset and args.subset > 0) else "full_dataset"
        if args.subset and args.subset > 0:
            df_tr, df_va, df_te = apply_subset(df_tr, df_va, df_te, args.subset, args.seed)
        log(f"data: {cleaning['usable_rows']:,} usable; "
            f"train={len(df_tr):,} val={len(df_va):,} test={len(df_te):,}")
        if args.subset and args.subset > 0:
            split = dict(split, subset=args.subset,
                         subset_sampling="deterministic_random_without_replacement",
                         subset_seed=args.seed,
                         train_rows=len(df_tr), val_rows=len(df_va), test_rows=len(df_te),
                         note="pilot subset: deterministic random sampling within each seeded split")
        if "audit" in stages:
            (out / "dataset_audit.json").write_text(json.dumps(
                {"cleaning": cleaning, "split": split,
                 "note": "run audit_dataset.py --pose-source ... for pose-level checks"}, indent=1))
        if args.pose_source == "remote":
            token = get_hf_token()
            if not token:
                raise SystemExit(
                    "AUTHENTICATION REQUIRED: HF_TOKEN not found in environment or Kaggle Secrets. "
                    "For remote pose streaming from Exploration-Lab/iSign (gated repository), "
                    "add HF_TOKEN under Kaggle Add-ons > Secrets or set HF_TOKEN environment variable."
                )
            log("remote pose authentication verified: HF_TOKEN is configured (redacted)")
            store_spec = ("remote", token)
        elif args.pose_source and zipfile.is_zipfile(args.pose_source):
            store_spec = ("zip", args.pose_source)
        elif args.pose_source:
            store_spec = ("dir", args.pose_source)
        else:
            raise SystemExit("--pose-source is required (dir | zip | remote)")
        store = _open_pose_store(*store_spec)
        # Cheap preflight BEFORE any expensive stage: proves the pose source
        # can seek/read and decode ONE real member. Writes pose_preflight.json.
        probe_uid = str(df_tr["uid"].iloc[0])
        preflight = preflight_pose_source(store, probe_uid)
        (out / "pose_preflight.json").write_text(json.dumps(preflight, indent=1))
        if args.preflight_only:
            log("preflight-only: pose source verified, exiting before expensive stages.")
            return 0
        # ---- measure (dry-run): zstd projection through the exact V2 path ----
        # Creates NO cache. Full D-cache creation stays blocked until the
        # Kaggle measurement verdict is reviewed.
        split_uids = {"train": [str(u) for u in list(df_tr["uid"])],
                      "val": [str(u) for u in list(df_va["uid"])],
                      "test": [str(u) for u in list(df_te["uid"])]}
        split_uids_order = split_uids  # authoritative row order; gate/div loaders map back into it
        if "measure" in stages:
            t_measure = time.perf_counter()
            rep = measure_compression(split_uids, store, out,
                                      n_samples=args.measure_samples,
                                      target_gib=args.persistence_target_gib)
            # Tokenizer, timed and persistent: later stages reuse the artifacts.
            # Always trained on the FULL training corpus (subset mode subsets pose
            # data only); otherwise a pilot measure would persist a subset vocab.
            t_tok = time.perf_counter()
            train_texts = df_tr_all["text"].astype(str).tolist() if not args.smoke else df_tr["text"].astype(str).tolist()
            sp_m = train_tokenizer(train_texts, out / "tokenizer", 16000)
            tok_h = (time.perf_counter() - t_tok) / 3600
            log(f"measure: tokenizer ready in {tok_h * 3600:.0f}s (reused by later stages)")
            # Compute benches on the real config/shapes (random weights noted).
            bench = benchmark_training_throughput(device, sp_m.get_piece_size(), args.batch,
                                                               embed_init=args.embed_init)
            t_ckpt = time.perf_counter()
            tmp_ckpt = out / "bench_ckpt.pt"
            torch.save({"probe": torch.randn(16)}, tmp_ckpt)
            ckpt_s = time.perf_counter() - t_ckpt
            tmp_ckpt.unlink(missing_ok=True)
            bench["ckpt_probe_save_s"] = ckpt_s
            # Decode rate on REAL val/test samples through the real path.
            # CPU bench uses a smaller pool (same code path, fewer hyps): the
            # autoregressive loop has no KV cache, so 256 full decodes on CPU
            # would take longer than the measurement is worth. GPU measures all.
            bench_n = BENCH_DECODE_SAMPLES if device.type == "cuda" else 32
            text_of = dict(zip(list(df_va["uid"]) + list(df_te["uid"]),
                               list(df_va["text"]) + list(df_te["text"])))
            bench_pool = [str(u) for u in list(text_of)[:bench_n]]
            bench_texts = [str(text_of[u]) for u in bench_pool]
            bmodel = SignTransformer(MODEL_INPUT_DIM, sp_m.get_piece_size(), MAX_FRAMES, MAX_TOKENS,
                                     pad_idx=sp_m.pad_id(), d_model=D_MODEL, nhead=NHEAD,
                                     layers=NLAYERS, ff=FF, dropout=DROPOUT,
                                     norm_first=NORM_FIRST)
            init_token_embedding(bmodel, args.embed_init)
            bmodel = bmodel.to(device)
            btids = [np.asarray(encode_text(sp_m, t), dtype=np.int64) for t in bench_texts]
            bloader = DataLoader(
                SignToTextDataset(np.array(bench_pool), np.array(bench_texts), sp_m,
                                  store, store_spec, token_ids=btids),
                batch_size=min(args.batch, len(bench_pool)), shuffle=False,
                collate_fn=collate_sign_text, num_workers=0)
            t_dec = time.perf_counter()
            _, dhyps, _ = batch_greedy(bmodel, bloader, sp_m, device)
            bench["decode_hyps_per_sec"] = len(dhyps) / max(1e-9, time.perf_counter() - t_dec)
            bench["decode_note"] = ("random weights, real config/sp/samples; "
                                    "optimistic for eval (fewer early-EOS breaks)")
            del bmodel
            rep["benchmarks"] = bench
            rep["tokenizer_hours"] = tok_h
            (out / MEASURE_REPORT).write_text(json.dumps(rep, indent=1))
            prep_h = 0.0 if args.data_mode == "stream" else float("nan")
            feas = compute_feasibility(rep, len(df_tr), len(df_va), len(df_te),
                                       args.batch, args.epochs, prep_h, tok_h,
                                       args.quota_hours, args.target_hours,
                                       data_mode=args.data_mode)
            if args.data_mode == "cache":
                feas["note"] = ("prepare excluded: cache-mode prepare cost unknown "
                                "until the 200-sample projection; re-run feasibility after prepare")
            (out / FEASIBILITY_REPORT).write_text(json.dumps(feas, indent=1))
            print(feasibility_banner(feas, len(df_tr) + len(df_va) + len(df_te),
                                     args.data_mode), flush=True)
            log(f"measure: done in {(time.perf_counter() - t_measure) / 60:.1f} min "
                f"(no cache created)")
        # ---- prepare: materialize the local pose cache once, train from it ----
        # data-mode=stream (default): FULL CACHE DISABLED, no materialization.
        pose_cache_dir = Path(args.pose_cache) if args.pose_cache else (out / "pose_cache")
        need_uids = [str(u) for u in
                     list(df_tr["uid"]) + list(df_va["uid"]) + list(df_te["uid"])]
        need_uids = list(dict.fromkeys(need_uids))
        log(f"data mode: {args.data_mode}" +
            (f" (cache budget {args.cache_budget_gib:.1f} GiB)" if args.data_mode == "cache" else
             " -- FULL CACHE DISABLED, on-demand remote reads"))
        if "prepare" in stages and args.data_mode == "cache":
            free_gib = shutil.disk_usage(pose_cache_dir if pose_cache_dir.exists()
                                         else pose_cache_dir.parent).free / (1024 ** 3)
            log(f"prepare: disk free {free_gib:.1f} GiB; budget {args.cache_budget_gib:.1f} GiB")
            t_prep = time.perf_counter()
            materialize_pose_cache(need_uids, store, pose_cache_dir,
                                   shard_samples=args.shard_samples,
                                   verify_every=args.verify_every,
                                   max_cache_gb=args.max_cache_gb,
                                   budget_gib=args.cache_budget_gib,
                                   jobs=args.jobs,
                                   source_factory=lambda: _open_pose_store(*store_spec))
            # Fail closed: preparation is only complete when every required
            # pilot UID is actually present in the cache. Never claim prepare
            # complete on partial coverage; the partial cache is kept and the
            # cell can be re-run to resume.
            _prep_present, _prep_missing = cache_coverage(pose_cache_dir, need_uids)
            if _prep_missing != 0:
                raise SystemExit(
                    f"pose cache incomplete after prepare: {_prep_missing:,}/{len(need_uids):,} "
                    f"samples not cached; partial cache kept, re-run the prepare cell to resume."
                )
            log(f"prepare: materialization took {(time.perf_counter() - t_prep) / 60:.1f} min")
        elif "prepare" in stages:
            log("prepare: skipped (data-mode=stream; nothing to materialize)")
        present, missing = cache_coverage(pose_cache_dir, need_uids)
        log(f"pose cache: {present:,}/{len(need_uids):,} samples cached "
            f"(tensor equivalent ~{estimate_processed_cache_gib(len(need_uids)):.1f} GiB fp32 NOT written)")
        require_local_cache(args.data_mode, stages, present, missing, len(need_uids))
        if missing == 0 and present:
            store_spec = ("shards", str(pose_cache_dir))
            store = _open_pose_store(*store_spec)
            # Prove the LOCAL cache decodes through the identical path.
            preflight_pose_source(store, probe_uid)
            log("training pose source: LOCAL CACHE (no remote HTTP during epochs)")
        else:
            log(f"training pose source: {args.pose_source} DIRECT "
                f"({missing:,} samples not cached -- run the prepare stage; "
                f"every epoch will re-read remotely)")

    # ---- tokenizer ----
    # Measure-only runs skip all tokenizer work (no model file needed, none built).
    needs_nlp = args.smoke or bool(stages & {"tokenizer", "overfit", "train", "evaluate", "export"})
    tok_dir = out / "tokenizer"
    if not needs_nlp:
        sp, split_tokens = None, None
        log("tokenizer skipped (measure-only run)")
    elif "tokenizer" in stages or args.smoke:
        t_tok = time.perf_counter()
        # In subset mode, train tokenizer on FULL training corpus for vocab consistency
        # with full runs; only pose data is subset. The token cache fingerprints
        # will differ (subset uids/texts) but the SP model is identical.
        tokenizer_train_texts = df_tr_all["text"].astype(str).tolist() if (not args.smoke and args.subset and args.subset > 0) else df_tr["text"].astype(str).tolist()
        vocab_size = 40 if args.smoke else 16000
        sp = train_tokenizer(tokenizer_train_texts, tok_dir, vocab_size)
        sp_train_artifacts["model"] = str(tok_dir / "signsage_en_sp.model")
        log(f"tokenizer vocab={sp.get_piece_size()} (trained on {len(tokenizer_train_texts):,} texts "
            f"{'[FULL CORPUS for subset mode]' if (not args.smoke and args.subset and args.subset > 0) else ''}) "
            f"in {time.perf_counter() - t_tok:.1f}s)")
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(tok_dir / "signsage_en_sp.model"))
        sp_train_artifacts["model"] = str(tok_dir / "signsage_en_sp.model")
        log(f"tokenizer reused: vocab={sp.get_piece_size()} (no retraining)")

    # ---- token cache (Phase B): tokenize each split once, reuse across sessions ----
    if args.smoke or not needs_nlp:
        split_tokens = None
    else:
        splits = {"train": (df_tr["uid"].to_numpy(), df_tr["text"].to_numpy()),
                  "val": (df_va["uid"].to_numpy(), df_va["text"].to_numpy()),
                  "test": (df_te["uid"].to_numpy(), df_te["text"].to_numpy())}
        split_tokens = load_or_build_token_cache(
            sp, tok_dir / "signsage_en_sp.model", tok_dir, splits)

    def make_loader(frame, train: bool, batch: int | None = None, token_ids=None,
                      memoize: bool = False):
        ds = SignToTextDataset(frame["uid"].to_numpy(), frame["text"].to_numpy(), sp, store, store_spec,
                               token_ids=token_ids, memoize=memoize)
        return DataLoader(ds, batch_size=batch or args.batch, shuffle=train,
                          collate_fn=collate_sign_text,
                          num_workers=2 if device.type == "cuda" else 0,
                          pin_memory=device.type == "cuda")

    def cached_ids_for(frame, split_name: str):
        """Token ids already built by load_or_build_token_cache, in frame order.

        The gate/diversity loaders sample rows of df_tr/df_va, so map uids back
        to split_tokens instead of re-running SentencePiece on the same text."""
        if args.smoke or split_tokens is None:
            return None
        pos = {str(u): i for i, u in enumerate(split_uids_order[split_name])}
        return [split_tokens[split_name][pos[str(u)]] for u in frame["uid"]]

    # ---- model ----
    def build_model():
        _m = SignTransformer(MODEL_INPUT_DIM, sp.get_piece_size(), MAX_FRAMES, MAX_TOKENS,
                             pad_idx=sp.pad_id(), d_model=D_MODEL, nhead=NHEAD, layers=NLAYERS,
                             ff=FF, dropout=DROPOUT, norm_first=NORM_FIRST)
        init_token_embedding(_m, args.embed_init)  # checkpoint loads below override initialization
        return _m.to(device)

    # ---- overfit gate (mandatory) ----
    if "overfit" in stages:
        log("overfit gate: 64 memorization samples must reach >=95% exact.")
        import pandas as pd  # noqa: F811

        # Reachability-filtered selection: the serving decoder bans repeated
        # n-grams (NO_REPEAT_NGRAM_SIZE) + EOS/min-length + special tokens, so
        # random targets can be un-emittable even when fully memorized
        # (unconstrained 98% vs constrained 92% on the same weights). Sample
        # ONLY from reachable examples; the constrained score then measures
        # optimization, not the bans. Held-out evaluation is untouched: this
        # filters the 64 memorization probes only, never val/test.
        train_uids_all = [str(u) for u in list(df_tr["uid"])]
        if split_tokens is not None and len(split_tokens.get("train", [])) == len(train_uids_all):
            train_ids_all = [np.asarray(a, dtype=np.int64).tolist()
                             for a in split_tokens["train"]]
        else:
            train_ids_all = [encode_text(sp, str(t))
                             for t in df_tr["text"].astype(str).tolist()]
        selected, gate_excluded = select_gate_examples(
            train_uids_all, train_ids_all, min(64, len(train_uids_all)), args.seed)
        if gate_excluded:
            reasons = {}
            for ex in gate_excluded:
                reasons[ex.get("reason", "?")] = reasons.get(ex.get("reason", "?"), 0) + 1
            log(f"overfit gate: excluded {len(gate_excluded)} unreachable "
                f"targets from selection {reasons}")
        sub = df_tr.iloc[selected].reset_index(drop=True)
        gate_uids = [str(u) for u in sub["uid"]]
        # shuffle=False: batch_greedy yields refs in loader order, which must
        # match gate_uids for the failure diagnostics below. (Batch order is
        # immaterial to 64-sample memorization.)
        gate_loader = make_loader(sub, False, batch=16, memoize=True,
                                  token_ids=cached_ids_for(sub, "train"))
        gate = build_model()
        gate_opt = torch.optim.AdamW(gate.parameters(), lr=1e-3)
        gate_crit = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
        exact = 0.0
        # Cosine decay 1e-3 -> 1e-5 over the first 600 steps, then hold the
        # floor: a fixed 1e-3 oscillates and diverges past ~step 400 (local
        # 64-sample replica with this exact loop: 87.5% -> 51.6% exact, loss
        # 0.07 -> 0.45; 0/31 failures constraint-blocked), so the gate can
        # fail while the implementation is sound. Decay settles into the
        # sharp memorization minimum instead of bouncing around it (replica:
        # loss ~0.01 stable, no collapse). The 1200-step tail at 1e-5 gives
        # the final sharpening room greedy exact needs (95% exact on ~15-token
        # targets needs teacher-forced loss ~0.003); the >=95% bar itself is
        # unchanged and the loop still breaks early on pass.
        _gate_checks: list = []
        refs, hyps, uhyps = [], [], []
        _ref_ids, _gen_c, _gen_u = [], [], []
        _gate_error: str | None = None
        try:
            for step in range(GATE_STEPS):
                gloss = run_epoch(gate, gate_loader, gate_opt, gate_crit, device)
                frac = min(1.0, (step + 1) / 600)
                lr = 1e-5 + (1e-3 - 1e-5) * 0.5 * (1 + math.cos(math.pi * frac))
                for pg in gate_opt.param_groups:
                    pg["lr"] = lr
                if (step + 1) % 100 == 0:
                    refs, hyps, _, _ref_ids, _gen_c = batch_greedy(
                        gate, gate_loader, sp, device, return_ids=True)
                    exact = float(np.mean([normalize_text(r) == normalize_text(h) for r, h in zip(refs, hyps)]))
                    # Diagnostic second opinion (never authoritative): the same
                    # greedy pass WITHOUT serving constraints. unconstrained >>
                    # constrained with equal loss means the bans are blocking
                    # memorized targets; both failing means optimization/data.
                    _, uhyps, _, _, _gen_u = batch_greedy(
                        gate, gate_loader, sp, device,
                        constrained=False, return_ids=True)
                    uexact = float(np.mean([normalize_text(r) == normalize_text(h) for r, h in zip(refs, uhyps)]))
                    log(f"  gate step {step + 1}: exact={exact:.1%} (unconstrained={uexact:.1%}) loss={gloss:.4f} lr={lr:.1e}")
                    _gate_checks.append({"step": step + 1, "loss": gloss, "lr": lr,
                                         "exact_constrained": exact,
                                         "exact_unconstrained": uexact})
                    if exact >= 0.95:
                        break
        except Exception as exc:  # persist partial diagnostics before surfacing
            _gate_error = f"{type(exc).__name__}: {exc}"
            log(f"overfit gate interrupted: {_gate_error}")
            raise
        finally:
            def _eos_pos(ids: list, eos_id: int):
                try:
                    return ids.index(eos_id)
                except ValueError:
                    return None
            _failures = []
            if refs and hyps:
                _uh = uhyps if len(uhyps) == len(refs) else [""] * len(refs)
                _gu = _gen_u if len(_gen_u) == len(refs) else [[] for _ in refs]
                for i, (r, h) in enumerate(zip(refs, hyps)):
                    if normalize_text(r) == normalize_text(h):
                        continue
                    rid = list(_ref_ids[i]) if i < len(_ref_ids) else []
                    cid = list(_gen_c[i]) if i < len(_gen_c) else []
                    uid_ = list(_gu[i]) if i < len(_gu) else []
                    _failures.append({
                        "uid": gate_uids[i] if i < len(gate_uids) else None,
                        "ref": r, "hyp_constrained": h, "hyp_unconstrained": _uh[i],
                        "ref_ids": rid, "hyp_constrained_ids": cid,
                        "hyp_unconstrained_ids": uid_,
                        "ref_len_ids": len(rid),
                        "hyp_constrained_len_ids": len(cid),
                        "hyp_unconstrained_len_ids": len(uid_),
                        "ref_eos_pos": _eos_pos(rid, sp.eos_id()),
                        "hyp_constrained_eos_pos": _eos_pos(cid, sp.eos_id()),
                        "hyp_unconstrained_eos_pos": _eos_pos(uid_, sp.eos_id()),
                        "ref_has_unk": sp.unk_id() in rid,
                    })
            _gate_report = {"threshold": 0.95, "passed": exact >= 0.95 and _gate_error is None,
                            "status": ("interrupted" if _gate_error
                                       else ("passed" if exact >= 0.95 else "failed")),
                            "quality_verdict": (None if _gate_error else bool(exact >= 0.95)),
                            "pose_source": (store_spec[0] if store_spec else "synthetic"),
                            "seed": args.seed, "gate_uids": gate_uids,
                            "n_gate": len(gate_uids),
                            "n_excluded_unreachable": len(gate_excluded),
                            "final": _gate_checks[-1] if _gate_checks else None,
                            "checks": _gate_checks,
                            "error": _gate_error,
                            "failures": _failures if exact < 0.95 else []}
            _tmp = out / "gate_report.json.tmp"
            _tmp.write_text(json.dumps(_gate_report, indent=1))
            os.replace(_tmp, out / "gate_report.json")
            log(f"gate diagnostics: {len(_failures)} failures "
                f"(ref/hyp/ids/lengths/EOS) -> {out / 'gate_report.json'}")
            for _f in _failures[:5]:
                log(f"  miss uid={_f['uid']} ref={_f['ref']!r} "
                    f"hyp={_f['hyp_constrained']!r} "
                    f"(unconstr={_f['hyp_unconstrained']!r}) "
                    f"ref_len={_f['ref_len_ids']} hyp_len={_f['hyp_constrained_len_ids']} "
                    f"eos_c={_f['hyp_constrained_eos_pos']} eos_u={_f['hyp_unconstrained_eos_pos']}")
        if exact < 0.95:
            raise SystemExit(f"OVERFIT GATE FAILED (exact={exact:.1%}) -- fix implementation, do not train. "
                             f"See {out / 'gate_report.json'}.")
        log("overfit gate PASSED.")

    # ---- train ----
    best_path = out / "checkpoints" / "signsage_transformer_full.pt"
    history: list = []
    if "train" in stages:
        if not args.smoke:
            feas_path = out / FEASIBILITY_REPORT
            if feas_path.is_file():
                try:
                    feas = json.loads(feas_path.read_text())
                except ValueError:
                    feas = {"verdict": "UNKNOWN-corrupt"}
                try:
                    proj = float(feas.get("projected_total_hours", float("nan")))
                except (TypeError, ValueError):
                    proj = float("nan")
                if (feas.get("verdict") == "FAIL" and not math.isfinite(proj)
                        and missing == 0 and present):
                    # Cache-mode measure could not price prepare; now that the cache
                    # is complete the cost is sunk -- recompute instead of refusing.
                    log("feasibility was FAIL-unknown-prepare but the pose cache is "
                        "COMPLETE; recomputing with sunk prepare cost...")
                    new = refresh_feasibility_after_cache(
                        out, len(df_tr), len(df_va), len(df_te),
                        args.batch, args.epochs, args.quota_hours, args.target_hours,
                        pose_cache=pose_cache_dir)
                    if new is not None:
                        feas = new
                        print(feasibility_banner(
                            feas, len(df_tr) + len(df_va) + len(df_te),
                            args.data_mode), flush=True)
                log(f"feasibility verdict: {feas.get('verdict')} "
                    f"(projected {feas.get('projected_total_hours', float('nan')):.2f}h "
                    f"vs target {feas.get('target_hours', float('nan')):.2f}h)")
                if feas.get("verdict") == "FAIL" and not args.force:
                    raise SystemExit(
                        "FEASIBILITY FAIL: refusing to burn quota. Re-run the measure cell, "
                        "reduce scope (subset/epochs via explicit flags), or pass --force "
                        "to override (quota WILL burn).")
                if feas.get("verdict") == "FAIL":
                    log("WARNING: --force override of FAIL feasibility; quota WILL burn")
            else:
                log("WARNING: no feasibility.json -- run the measure cell first; "
                    "proceeding WITHOUT a measured quota gate")
        train_loader = make_loader(df_tr, True, token_ids=None if args.smoke else split_tokens["train"])
        val_loader = make_loader(df_va, False, token_ids=None if args.smoke else split_tokens["val"])
        model = build_model()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = math.ceil(len(train_loader))
        scheduler = build_schedule(opt, max(1, steps_per_epoch * args.epochs))
        crit = nn.CrossEntropyLoss(ignore_index=sp.pad_id(), label_smoothing=LABEL_SMOOTHING)
        if device.type == "cuda":
            log(f"pre-train GPU memory: alloc={torch.cuda.memory_allocated() / (1024**2):.1f}MB, "
                f"res={torch.cuda.memory_reserved() / (1024**2):.1f}MB")
            bench = benchmark_training_throughput(device, sp.get_piece_size(), args.batch,
                                                               embed_init=args.embed_init)
            log(f"pre-train benchmark throughput: {bench['train_steps_per_sec']:.1f} steps/s "
                f"(~{bench['train_steps_per_sec'] * args.batch:.1f} samples/s compute-only)")
        start_epoch, best, bad = 1, float("inf"), 0
        log(f"embed init: {args.embed_init}")
        unigram_nll = None
        if split_tokens is not None:
            unigram_nll = unigram_baseline_nll(split_tokens["train"], split_tokens["val"], sp.get_piece_size())
            log(f"baselines (val NLL nats/token): uniform={math.log(sp.get_piece_size()):.3f} "
                f"best-unigram={unigram_nll:.3f}; diagnostic reference, not proof of pose use")
        crit_plain = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
        rec_path = out / "checkpoints" / "signsage_transformer_recovery.pt"
        initialization_config = {"seed": args.seed, "embed_init": args.embed_init}
        if args.resume and rec_path.exists():
            try:
                rec = torch.load(rec_path, map_location=device, weights_only=False)
            except Exception as exc:  # noqa: BLE001 -- surface an actionable message, not a raw traceback
                raise SystemExit(
                    f"recovery checkpoint {rec_path} is unreadable ({type(exc).__name__}: {exc}). "
                    f"It was probably truncated by a killed session. The best checkpoint "
                    f"({out / 'checkpoints' / 'signsage_transformer_full.pt'}) is untouched. Move the "
                    f"recovery file aside and re-run, or restore the last saved output version.") from exc
            check_checkpoint_not_collapsed(rec, rec_path)
            initialization_config = rec.get("initialization_config") or {"seed": None, "embed_init": None}
            model.load_state_dict(rec["state_dict"])
            opt.load_state_dict(rec["optimizer_state_dict"])
            if rec.get("scheduler_state_dict"):
                scheduler.load_state_dict(rec["scheduler_state_dict"])
            start_epoch = int(rec.get("epoch", 0)) + 1
            best = float(rec.get("best", best))
            bad = int(rec.get("bad", bad))
            history = list(rec.get("history", history))
            rng = rec.get("rng") or {}
            try:
                if rng.get("python") is not None:
                    random.setstate(rng["python"])
                if rng.get("numpy") is not None:
                    np.random.set_state(rng["numpy"])
                if rng.get("torch") is not None:
                    torch.set_rng_state(rng["torch"].cpu())
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as exc:  # noqa: BLE001 -- RNG restore is best-effort
                log(f"resume: RNG state not restored ({type(exc).__name__}); training continues")
            log(f"resumed at epoch {start_epoch} (best val {best:.4f}, bad={bad}, "
                f"history={len(history)} epochs, opt+sched+rng restored)")
        elif rec_path.exists():
            log(f"recovery checkpoint present at {rec_path} but --resume not given: "
                f"starting from epoch 1 (pass --resume to continue)")
        else:
            log("no recovery checkpoint: starting from epoch 1")
        div_frame = df_va.sample(min(DIVERSITY_SAMPLE, len(df_va)),
                                   random_state=args.seed).reset_index(drop=True)
        div_loader_sample = make_loader(div_frame, False,
                                        token_ids=cached_ids_for(div_frame, "val"),
                                        memoize=True)
        ckpt_model_config = {
            "input_dim": MODEL_INPUT_DIM, "raw_feature_dim": LANDMARK_DIM,
            "vocab_size": sp.get_piece_size(),
            "d_model": D_MODEL, "nhead": NHEAD, "layers": NLAYERS, "ff": FF,
            "dropout": DROPOUT, "norm_first": NORM_FIRST,
            "max_frames": MAX_FRAMES, "max_tokens": MAX_TOKENS,
        }
        try:
            ckpt_tokenizer_fp = _sp_fingerprint(
                sp, Path(sp_train_artifacts.get("model", tok_dir / "signsage_en_sp.model")))
        except OSError:
            ckpt_tokenizer_fp = ""
        for epoch in range(start_epoch, args.epochs + 1):
            tr_loss = run_epoch(model, train_loader, opt, crit, device, scheduler, train=True,
                                 decoder_input_dropout=args.decoder_input_dropout,
                                 vocab_size=sp.get_piece_size())
            _gstats = getattr(run_epoch, "last_train_stats", None) or {}
            va_loss = run_epoch(model, val_loader, opt, crit, device, train=False)
            # Unsmoothed val NLL: comparable to the unigram baseline (val_loss carries label smoothing).
            va_nll = run_epoch(model, val_loader, opt, crit_plain, device, train=False,
                               token_weighted=True)
            va_nll_shuf = run_epoch(model, PoseRolledLoader(val_loader), opt, crit_plain, device,
                                    train=False, token_weighted=True)
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                            "val_nll": va_nll, "val_nll_shuffled_pose": va_nll_shuf,
                            "pose_gap": va_nll_shuf - va_nll, "unigram_val_nll": unigram_nll,
                            "embed_init": initialization_config["embed_init"],
                            "epoch_completed": True, "diversity_checked": False,
                            "diversity_pass": None, "accepted_for_checkpoint": False,
                            "checkpoint_written": False})
            log(f"epoch {epoch:02d}/{args.epochs}: train={tr_loss:.4f} val={va_loss:.4f} "
                f"val_nll={va_nll:.4f} (unigram={'n/a' if unigram_nll is None else f'{unigram_nll:.4f}'}) "
                f"shuffled-pose nll={va_nll_shuf:.4f} gap={va_nll_shuf - va_nll:+.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}")
            if _gstats:
                log(f"  grads: norm mean={_gstats['grad_norm_mean']:.4f} "
                    f"max={_gstats['grad_norm_max']:.4f} "
                    f"clipped={_gstats['grad_clip_frac']:.1%} "
                    f"param_norm={_gstats['param_norm']:.2f}")
            # Reuse this bounded held-out decode for diagnostics; no extra decode/evaluation.
            drefs, dhyps, dconfs = batch_greedy(model, div_loader_sample, sp, device)
            div = diversity_metrics(dhyps)
            dominant = build_trivial_baseline(drefs, df_tr["text"].astype(str).tolist())
            diversity_checked = epoch >= 2
            diversity_pass = div["unique_fraction"] >= DIVERSITY_FLOOR if diversity_checked else None
            history[-1].update({
                "diversity_checked": diversity_checked, "diversity_pass": diversity_pass,
                "accepted_for_checkpoint": not diversity_checked or diversity_pass,
                "diagnostic_only": True, "quality_verdict": None,
                "diagnostic_metrics": compute_metrics(drefs, dhyps),
                "dominant_sentence_baseline": {"prediction": dominant[0],
                                               "metrics": compute_metrics(drefs, dominant)},
                "validation_samples": [{"uid": str(uid), "ref": r, "hyp": h, "conf": c}
                                       for uid, r, h, c in zip(div_frame["uid"].iloc[:16],
                                                               drefs[:16], dhyps[:16], dconfs[:16])],
            })
            if not diversity_checked:
                log("  epoch 1: diversity guard unchecked (warmup); checkpoint provisional, not proven healthy")
            log("  held-out sample metrics vs dominant training sentence: diagnostic only, not a quality PASS")
            log(f"  diversity: unique={div['unique']}/{len(dhyps)} ({div['unique_fraction']:.2%}) "
                f"top={div['top_prediction']!r} ({div['top_prediction_freq']:.1%})")
            history[-1].update({"unique": div["unique"], "unique_fraction": div["unique_fraction"],
                                "top_prediction": div["top_prediction"],
                                "top_prediction_freq": div["top_prediction_freq"],
                                "grad_norm_mean": _gstats.get("grad_norm_mean"),
                                "grad_norm_max": _gstats.get("grad_norm_max"),
                                "grad_clip_frac": _gstats.get("grad_clip_frac"),
                                "param_norm": _gstats.get("param_norm")})
            _htmp = out / "training_history.json.tmp"
            _htmp.write_text(json.dumps(history, indent=1))
            os.replace(_htmp, out / "training_history.json")
            if diversity_checked and not diversity_pass:
                # Keep previous accepted recovery/best files (epoch 1 may be unchecked).
                # checkpoint_written tracks accepted checkpoints, not this forensic artifact.
                _cpath = out / "checkpoints" / f"signsage_transformer_collapsed_epoch{epoch:02d}.pt"
                _atomic_torch_save({"state_dict": model.state_dict(), "epoch": epoch, "val_loss": va_loss,
                                    "history": history, "collapsed": True,
                                    "initialization_config": initialization_config,
                                    "training_args": vars(args).copy(),
                                    "model_config": ckpt_model_config,
                                    "tokenizer_fingerprint": ckpt_tokenizer_fp,
                                    "optimizer_state_dict": opt.state_dict(),
                                    "scheduler_state_dict": scheduler.state_dict()}, _cpath)
                (out / "COLLAPSED.txt").write_text(
                    f"epoch {epoch}: unique_fraction={div['unique_fraction']:.4f} < {DIVERSITY_FLOOR}; "
                    f"val_loss={va_loss:.4f}; "
                    f"top={div['top_prediction']!r} ({div['top_prediction_freq']:.1%}); "
                    f"forensic snapshot: {_cpath.name}\n")
                raise SystemExit("COLLAPSE GATE FIRED -- stopped instead of burning more GPU. See COLLAPSED.txt.")
            # Best/patience transition BEFORE building the payload so the
            # saved recovery metadata reflects this epoch's outcome. The best
            # file is only overwritten on a genuine improvement, with the same
            # consistent payload (its val_loss == best at that point).
            best, bad, improved, should_stop = update_best_state(
                best, bad, va_loss, args.patience)
            payload = {"state_dict": model.state_dict(), "epoch": epoch, "val_loss": va_loss, "best": best,
                       "bad": bad,
                       # This history is visible only if the atomic checkpoint write succeeds.
                       "history": [*history[:-1], {**history[-1], "checkpoint_written": True}],
                       "initialization_config": initialization_config,
                       "training_args": vars(args).copy(),
                       "model_config": ckpt_model_config,
                       "tokenizer_fingerprint": ckpt_tokenizer_fp,
                       "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                       "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                               "torch": torch.get_rng_state().cpu(),
                               "cuda": (torch.cuda.get_rng_state_all()
                                        if torch.cuda.is_available() else None)}}
            t_ckpt = time.perf_counter()
            _atomic_torch_save(payload, rec_path)
            history[-1]["checkpoint_written"] = True
            _htmp.write_text(json.dumps(history, indent=1))
            os.replace(_htmp, out / "training_history.json")
            log(f"  recovery checkpoint saved ({time.perf_counter() - t_ckpt:.1f}s)")
            if improved:
                _atomic_torch_save(payload, best_path)
                log(f"  new best (val {best:.4f})")
            elif should_stop:
                log("early stopping: val loss stopped improving.")
                break
            if (time.time() - t_start) / 3600 > args.max_hours:
                log("wall-clock budget reached -- stopping gracefully (resume with --resume).")
                break
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not best_path.exists():
            raise SystemExit("no best checkpoint produced.")
    else:
        if not best_path.exists():
            if {"evaluate", "export"} & stages:
                raise SystemExit("train stage skipped but no checkpoint found; nothing to evaluate/export.")
            log("train stage skipped and no checkpoint present; requested stages complete.")
            return 0

    # ---- evaluate ----
    eval_report: dict = {"history": history, "scope": _scope,
                         "rows_evaluated": {"test": len(df_te)}, "rows_available": _rows_available}
    if "evaluate" in stages:
        model = build_model()
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        check_checkpoint_not_collapsed(checkpoint, best_path)
        model.load_state_dict(checkpoint["state_dict"])
        test_loader = make_loader(df_te, False, token_ids=None if args.smoke else split_tokens["test"])
        refs, hyps, confs = batch_greedy(model, test_loader, sp, device)
        test_metrics = compute_metrics(refs, hyps)
        div = diversity_metrics(hyps, refs)
        train_texts = df_tr["text"].astype(str).tolist() if not args.smoke else None
        trivial = build_trivial_baseline(refs, train_texts)
        trivial_metrics = compute_metrics(refs, trivial)
        samples = [{"ref": r, "hyp": h, "conf": c}
                   for r, h, c in zip(refs[:30], hyps[:30], confs[:30])]
        lengths = {"gen_mean": float(test_metrics["gen_len_mean"]),
                   "ref_mean": float(test_metrics["ref_len_mean"])}
        gate = evaluate_gate(test_metrics, trivial_metrics, div)
        eval_report.update({"test_metrics": test_metrics, "trivial_baseline": trivial_metrics,
                            "diversity": div, "lengths": lengths, "samples": samples, "gate": gate,
                            "mean_confidence_proxy": float(np.mean(confs))})
        log(f"test: WER={test_metrics['WER']:.2f}% BLEU={test_metrics['BLEU']:.2f} "
            f"chrF={test_metrics['chrF']:.2f} TER={test_metrics['TER']:.2f}% "
            f"exact_match={test_metrics['exact_match']:.2%} unique={div['unique_fraction']:.2%} GO={gate['GO']}")
        (out / "evaluation_report.json").write_text(json.dumps(eval_report, indent=1))

    # ---- export ----
    if "export" in stages:
        model = build_model()
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        check_checkpoint_not_collapsed(checkpoint, best_path)
        model.load_state_dict(checkpoint["state_dict"])
        is_pilot = bool(args.subset and args.subset > 0)
        train_cfg = {"architecture": {"d_model": D_MODEL, "nhead": NHEAD, "layers": NLAYERS,
                                      "ff": FF, "dropout": DROPOUT, "norm_first": NORM_FIRST},
                     "label_smoothing": LABEL_SMOOTHING, "peak_lr": args.lr,
                     "batch": args.batch, "seed": args.seed, "history": history,
                     "decoder_input_dropout": args.decoder_input_dropout,
                     "initialization_config": checkpoint.get("initialization_config"),
                     "is_pilot": is_pilot, "is_smoke": bool(args.smoke),
                     "data_counts": {"scope": _scope, "usable_rows": _usable_rows,
                                     "rows_available": _rows_available,
                                     "rows_used": {"train": len(df_tr), "val": len(df_va),
                                                   "test": len(df_te)},
                                     "seed": args.seed, "data_mode": args.data_mode}}
        if is_pilot:
            train_cfg["subset"] = args.subset
            train_cfg["subset_seed"] = args.seed
            train_cfg["note"] = "PILOT / PIPELINE VALIDATION ONLY -- NOT FOR PRODUCTION"
        bundle = export_bundle(model, sp, out, run_id, train_cfg, eval_report,
                               Path(args.contract) if args.contract else None)
        log(f"bundle exported: {bundle}")
    log("done.")
    return 0


def reference_reachability(token_ids: list[int]) -> dict:
    """Check whether one encoded target can be emitted by the serving decoder.

    Validates the token ID sequence against the serving constraints:
    - BOS must be first
    - EOS must be last and not before MIN_OUTPUT_TOKENS
    - Token count must not exceed MAX_DECODE_TOKENS
    - Special tokens (PAD, BOS, UNK) are forbidden in the target
    - No repeated n-grams under the active NO_REPEAT_NGRAM_SIZE constraint

    Returns a dict with "reachable" bool and optional reason/position/token_id.
    """
    pad, unk, bos, eos = 0, 1, 2, 3
    ids = [int(token) for token in token_ids]
    if not ids or ids[0] != bos:
        return {"reachable": False, "reason": "missing_bos", "position": 0}
    if ids[-1] != eos or eos in ids[:-1]:
        return {"reachable": False, "reason": "invalid_eos_position", "position": len(ids) - 1}
    if len(ids) - 1 > MAX_DECODE_TOKENS:
        return {"reachable": False, "reason": "max_decode_tokens",
                "position": MAX_DECODE_TOKENS, "max_decode_tokens": MAX_DECODE_TOKENS}
    history = [bos]
    for position, token in enumerate(ids[1:]):
        if token in (pad, bos, unk):
            return {"reachable": False, "reason": "forbidden_special_token",
                    "position": position, "token_id": token}
        if token == eos and position < MIN_OUTPUT_TOKENS:
            return {"reachable": False, "reason": "minimum_output_tokens",
                    "position": position, "token_id": token}
        if NO_REPEAT_NGRAM_SIZE > 1 and len(history) >= NO_REPEAT_NGRAM_SIZE - 1:
            prefix = tuple(history[-(NO_REPEAT_NGRAM_SIZE - 1):])
            for start in range(len(history) - NO_REPEAT_NGRAM_SIZE + 1):
                ngram = tuple(history[start:start + NO_REPEAT_NGRAM_SIZE])
                if ngram[:-1] == prefix and ngram[-1] == token:
                    return {"reachable": False, "reason": "no_repeat_ngram",
                            "position": position, "token_id": token,
                            "blocked_ngram": list(ngram)}
        history.append(token)
    return {"reachable": True}


def select_gate_examples(uids: list[str], token_ids: list[list[int]],
                         limit: int, seed: int) -> tuple[list[int], list[dict]]:
    """Select a deterministic gate subset of UIDs whose token IDs are decoder-reachable.

    Args:
        uids: List of unique identifiers (one per sample).
        token_ids: List of token ID lists (each WITH BOS/EOS, as produced by
            encode_text) checked by reference_reachability.
        limit: Number of UIDs to select for the gate subset (typically 64).
        seed: Random seed for deterministic selection.

    Returns:
        selected: List of selected UID indices (sorted, length = min(limit,
            n_reachable)). Never includes unreachable examples: when fewer
            than limit are reachable the gate runs on fewer probes and the
            95% bar applies to that smaller n.
        excluded: List of dicts, each with "uid" and reason from reference_reachability
                  for UIDs that are unreachable under the active decoding constraints.
                  UIDs reachable but simply not selected (because limit < n_eligible)
                  are not included in this list.
    """
    if len(uids) != len(token_ids):
        raise ValueError("gate UID/token counts differ")

    # Classify each example as reachable or unreachable under the active decoder constraints
    eligible: list[int] = []
    excluded: list[dict] = []
    for index, (uid, ids) in enumerate(zip(uids, token_ids)):
        result = reference_reachability(ids)
        if result["reachable"]:
            eligible.append(index)
        else:
            excluded.append({"uid": str(uid), **result})

    # Deterministically select from eligible examples using the seeded RNG.
    # Unreachable examples are NEVER backfilled: emitting them exactly is
    # forbidden by the serving bans, so including them would cap the gate
    # below 95% no matter how well the model memorized.
    rng = np.random.default_rng(seed)
    n_eligible = len(eligible)
    n_select = min(limit, n_eligible)
    selected_indices = np.sort(rng.choice(eligible, size=n_select, replace=False)).tolist() if n_select else []

    selected_indices = selected_indices[:limit]
    selected = np.sort(selected_indices).tolist()

    # Rebuild excluded list: only UIDs that are unreachable under the active constraints,
    # i.e., reference_reachability returned reachable=False for them.
    # UIDs that are reachable but simply not selected (because limit < n_eligible)
    # are NOT included in the excluded list.
    selected_set = set(selected_indices)
    final_excluded = []
    # First, add truly excluded uids (constraint-blocked) that are not selected
    for ex in excluded:
        uid_idx = uids.index(ex["uid"]) if ex["uid"] in uids else -1
        if uid_idx not in selected_set:
            final_excluded.append(ex)
    # Do NOT add eligible uids not selected -- they're not "excluded" in the constraint sense

    return selected, final_excluded


def assert_gate_pass_alignment(expected_uids: list[str], actual_uids: list[str],
                               expected_ref_ids: list[list[int]],
                               constrained_ref_ids: list[list[int]],
                               unconstrained_ref_ids: list[list[int]],
                               constrained_refs: list[str],
                               unconstrained_refs: list[str]) -> None:
    """Validate that gate evaluation references are properly aligned.

    Checks:
    1. expected_uids match actual_uids (order-sensitive: UID order must
       match selected examples).
    2. expected_ref_ids match constrained_ref_ids (order-sensitive exact
       equality on the authoritative constrained pass).
    3. Every companion list (unconstrained_ref_ids, constrained_refs,
       unconstrained_refs) has exactly one entry per expected probe, so
       positional pairing downstream cannot silently misattribute rows.

    What is NOT claimed: string lists carry no token IDs, so their *order*
    is not verified by content -- only their length. Reordered ID rows are
    caught by check 2; reordered strings of equal length are not. Callers
    must keep all lists in the same row order. Unconstrained outputs are
    diagnostic only and need not match constrained content.

    Raises:
        RuntimeError: If UID alignment, reference alignment, or list-length
            parity checks fail.
    """
    if expected_uids != actual_uids:
        raise RuntimeError("gate evaluation UIDs are not aligned with selected examples")
    if expected_ref_ids != constrained_ref_ids:
        raise RuntimeError(
            "constrained gate references are not aligned with expected references"
        )
    # Positional pairing downstream indexes every list by row, so a length
    # mismatch on any of them would misattribute refs/hyps to the wrong
    # probe. Length parity is the only reliable check available for the
    # string lists (they carry no IDs to compare by content).
    for name, seq in (("unconstrained_ref_ids", unconstrained_ref_ids),
                      ("constrained_refs", constrained_refs),
                      ("unconstrained_refs", unconstrained_refs)):
        if len(seq) != len(expected_ref_ids):
            raise RuntimeError(
                f"{name} length mismatch with expected references "
                f"({len(seq)} vs {len(expected_ref_ids)})"
            )


if __name__ == "__main__":
    sys.exit(main())