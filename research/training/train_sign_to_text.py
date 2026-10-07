#!/usr/bin/env python3
"""SignSage v2 sign-to-text training — corrected post-352868924 configuration.

What this is: a reproducible, resumable training job for continuous ISL
pose-sequence -> English translation on iSign v1.1. It reuses the backend's
 OWN model, preprocessing, and masks (single canonical implementation), keeps
every training mechanic the 352868924 audit verified as correct
(shift / masks / ignore_index / video-level split / tokenizer procedure),
and changes only what the root-cause analysis justified (see
research/training/READINESS.md): pre-norm, larger capacity, label smoothing,
longer schedule, per-epoch diversity collapse gate, hard overfit gate.

Stages (select with --stages subset, default all):
    audit -> tokenizer -> prepare -> overfit-gate -> train -> evaluate -> export

    python train_sign_to_text.py --csv /path/iSign_v1.1.csv \\
        --pose-source /path/to/poses.zip --out /kaggle/working/s2t_v2
    python train_sign_to_text.py --smoke --out /tmp/s2t_smoke   # CPU plumbing check

Resumable: recovery snapshots every --snapshot-batches + best kept;
re-run with --resume to continue. Never overwrites run 352868924.

Multi-session (Kaggle): the prepare stage materializes a resumable local
pose cache (<out>/pose_cache) once; later sessions reuse it plus the token
cache (<out>/tokenizer/tokens.npz) and --resume continues from the recovery
checkpoint. /kaggle/working is ephemeral -- persist <out> via Save Version
and re-attach it as an input dataset next session (see signsage_kaggle.ipynb).
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

# v2 model capacity (root-cause fix: 352868924's d192x3L underfit 100k pairs).
D_MODEL, NHEAD, NLAYERS, FF = 256, 4, 4, 1024
DROPOUT = 0.15
NORM_FIRST = True  # pre-norm: stable from step 0 on short budgets
LABEL_SMOOTHING = 0.1
PEAK_LR = 3e-4
WARMUP_FRACTION = 0.06
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


def log(msg: str) -> None:
    print(f"[s2t-v2 {datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


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
        r = self._requests.get(url, headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0",
                                             "User-Agent": "SignSage-v2/1.0"},
                               timeout=(20, 60), allow_redirects=True)
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
        raise ConnectionError(f"range fetch failed after retries: {key}") from last

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
        self.manifest = json.loads((self.cache_dir / POSE_CACHE_MANIFEST).read_text())
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
    if not (cache_dir / POSE_CACHE_MANIFEST).is_file():
        return 0, len(uids)
    try:
        manifest = json.loads((cache_dir / POSE_CACHE_MANIFEST).read_text())
        index = manifest.get("uid_to_shard", {})
    except (OSError, ValueError):
        return 0, len(uids)
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


def materialize_pose_cache(uids: list[str], remote: PoseSource, cache_dir: Path,
                           shard_samples: int = DEFAULT_SHARD_SAMPLES,
                           verify_every: int = DEFAULT_VERIFY_EVERY,
                           max_cache_gb: float = 0.0, jobs: int = 1,
                           source_factory=None) -> Path:
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
                 store_spec: tuple | None = None, token_ids=None):
        self.uids = uids
        self.sp = sp
        self.store = store
        self.store_spec = store_spec  # (kind, arg); workers open their own (fork-safe)
        self._worker_store = None
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
        if self.store is None and self.store_spec is None:  # smoke mode: deterministic synthetic poses
            rng = np.random.default_rng(abs(hash(uid)) % (2 ** 31))
            frames = rng.normal(size=(MAX_FRAMES, LANDMARK_DIM)).astype(np.float32)
            length = MAX_FRAMES
        else:
            frames, length = load_pose_item(uid, self._active_store())
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


def run_epoch(model, loader, opt, crit, device, scheduler=None, train: bool = True) -> float:
    model.train(train)
    total = 0.0
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
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            total += loss.item()
            timer.add_batch(t_comp0 - t_data0, time.perf_counter() - t_comp0)
    log(timer.summary())
    return total / max(1, len(loader))


@torch.no_grad()
def batch_greedy(model, loader, sp, device, max_decode: int = MAX_DECODE_TOKENS):
    """Constrained batched greedy decode (same bans as serving).

    Parity contract with backend/app/ml/inference/sign_to_text.py
    greedy_decode_ids(): per generated row, ban PAD/BOS/UNK, ban EOS for the
    first MIN_OUTPUT_TOKENS steps, apply ban_repeated_ngram with
    NO_REPEAT_NGRAM_SIZE, then the REPETITION_PENALTY divide-if->=0-else-multiply
    over set(ids[1:]), then log_softmax + argmax. Serving is single-sample;
    this is the batched equivalent -- covered by test_decode_parity.py.
    """
    t0 = time.perf_counter()
    model.eval()
    refs, hyps, confs = [], [], []
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
        n_batches += 1
    log(f"decode: {len(hyps):,} hyps in {time.perf_counter() - t0:.1f}s ({n_batches} batches)")
    return refs, hyps, confs


def diversity_metrics(hyps: list[str]) -> dict:
    top, count = Counter(hyps).most_common(1)[0] if hyps else ("", 0)
    return {
        "unique": len(set(hyps)),
        "unique_fraction": len(set(hyps)) / max(1, len(hyps)),
        "top_prediction": top,
        "top_prediction_freq": count / max(1, len(hyps)),
    }


def compute_metrics(refs: list[str], hyps: list[str]) -> dict:
    import jiwer
    import sacrebleu
    from rouge_score import rouge_scorer

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return {
        "WER": float(jiwer.wer(refs, hyps)),
        "BLEU": float(bleu.score),
        "BLEU-1": float(bleu.precisions[0]),
        "BLEU-2": float(bleu.precisions[1]),
        "BLEU-3": float(bleu.precisions[2]),
        "BLEU-4": float(bleu.precisions[3]),
        "ROUGE-L": float(np.mean([rouge.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)])),
        "exact_match": float(np.mean([normalize_text(r) == normalize_text(h) for r, h in zip(refs, hyps)])),
        "nonempty_output": float(np.mean([bool(normalize_text(h)) for h in hyps])),
    }


# --------------------------------------------------------------------------
# Export (same bundle layout the backend already loads)
# --------------------------------------------------------------------------

def export_bundle(model, sp, out_dir: Path, run_id: str, train_cfg: dict,
                  eval_report: dict, contract_src: Path | None) -> Path:
    bundle = out_dir / "signsage_deployment_bundle"
    (bundle / "checkpoint").mkdir(parents=True, exist_ok=True)
    (bundle / "tokenizer").mkdir(parents=True, exist_ok=True)
    mc = {
        "input_dim": MODEL_INPUT_DIM, "raw_feature_dim": LANDMARK_DIM,
        "vocab_size": sp.get_piece_size(),
        "d_model": D_MODEL, "nhead": NHEAD, "layers": NLAYERS, "ff": FF,
        "dropout": DROPOUT, "norm_first": NORM_FIRST,
        "max_frames": MAX_FRAMES, "max_tokens": MAX_TOKENS,
    }
    torch.save({"state_dict": model.state_dict(), "model_config": mc,
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
    bundle.joinpath("sign_to_text_deployment.json").write_text(json.dumps({
        "project": "SignSage", "task": "ISL_to_English", "dataset": "iSign v1.1",
        "feature_dim": LANDMARK_DIM, "model_input_dim": MODEL_INPUT_DIM,
        "max_frames": MAX_FRAMES, "max_tokens": MAX_TOKENS, "vocab_size": sp.get_piece_size(),
        "tokenizer": "SentencePiece unigram trained on training targets only",
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
        "status": "unpromoted — set SIGNSAGE_MODEL_PROMOTED=true only after the quality gate passes",
    }
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
                    help="pilot mode: cap train/val/test to the first N rows each (0 = full data). "
                         "Loudly logged; representation and splits unchanged, just fewer videos.")
    ap.add_argument("--contract", default="research/evaluation/reports/352868924/landmark_contract.json",
                    help="landmark contract to copy into the bundle (representation unchanged)")
    ap.add_argument("--out", required=True, help="output dir for checkpoints/bundle/report")
    ap.add_argument("--stages", default="all",
                    help="comma subset of: audit,tokenizer,prepare,overfit,train,evaluate,export")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=PEAK_LR)
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
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"run {run_id} on {device}; stages={sorted(stages)}")

    # ---- data ----
    if args.smoke:
        log("SMOKE mode: synthetic poses + tiny vocab targets (no dataset).")
        vocab_texts = [f"hello world {i}" for i in range(8)]
        df_tr = pd.DataFrame({"uid": [f"smoke-{i}" for i in range(8)], "text": vocab_texts})
        df_va = df_tr.iloc[:4].reset_index(drop=True)
        df_te = df_tr.iloc[:4].reset_index(drop=True)
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
        if args.subset and args.subset > 0:
            log(f"SUBSET MODE: capping each split to its first {args.subset} rows "
                f"(pilot run only -- NOT the full V2 experiment; representation unchanged)")
            df_tr = df_tr.iloc[:args.subset].reset_index(drop=True)
            df_va = df_va.iloc[:args.subset].reset_index(drop=True)
            df_te = df_te.iloc[:args.subset].reset_index(drop=True)
        log(f"data: {cleaning['usable_rows']:,} usable; "
            f"train={len(df_tr):,} val={len(df_va):,} test={len(df_te):,}")
        if args.subset and args.subset > 0:
            split = dict(split, subset=args.subset,
                         note="pilot subset: first N rows of each seeded split")
        if "audit" in stages:
            (out / "dataset_audit.json").write_text(json.dumps(
                {"cleaning": cleaning, "split": split,
                 "note": "run audit_dataset.py --pose-source ... for pose-level checks"}, indent=1))
        if args.pose_source == "remote":
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise SystemExit("remote pose streaming needs HF_TOKEN")
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
        # ---- prepare: materialize the local pose cache once, train from it ----
        pose_cache_dir = Path(args.pose_cache) if args.pose_cache else (out / "pose_cache")
        need_uids = [str(u) for u in
                     list(df_tr["uid"]) + list(df_va["uid"]) + list(df_te["uid"])]
        need_uids = list(dict.fromkeys(need_uids))
        if "prepare" in stages:
            t_prep = time.perf_counter()
            materialize_pose_cache(need_uids, store, pose_cache_dir,
                                   shard_samples=args.shard_samples,
                                   verify_every=args.verify_every,
                                   max_cache_gb=args.max_cache_gb,
                                   jobs=args.jobs,
                                   source_factory=lambda: _open_pose_store(*store_spec))
            log(f"prepare: materialization took {(time.perf_counter() - t_prep) / 60:.1f} min")
        present, missing = cache_coverage(pose_cache_dir, need_uids)
        log(f"pose cache: {present:,}/{len(need_uids):,} samples cached "
            f"(tensor equivalent ~{estimate_processed_cache_gib(len(need_uids)):.1f} GiB fp32 NOT written)")
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
    tok_dir = out / "tokenizer"
    if "tokenizer" in stages or args.smoke:
        t_tok = time.perf_counter()
        train_texts = df_tr["text"].astype(str).tolist()
        vocab_size = 40 if args.smoke else 16000
        sp = train_tokenizer(train_texts, tok_dir, vocab_size)
        sp_train_artifacts["model"] = str(tok_dir / "signsage_en_sp.model")
        log(f"tokenizer vocab={sp.get_piece_size()} (trained in {time.perf_counter() - t_tok:.1f}s)")
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(tok_dir / "signsage_en_sp.model"))
        sp_train_artifacts["model"] = str(tok_dir / "signsage_en_sp.model")
        log(f"tokenizer reused: vocab={sp.get_piece_size()} (no retraining)")

    # ---- token cache (Phase B): tokenize each split once, reuse across sessions ----
    if args.smoke:
        split_tokens = None
    else:
        splits = {"train": (df_tr["uid"].to_numpy(), df_tr["text"].to_numpy()),
                  "val": (df_va["uid"].to_numpy(), df_va["text"].to_numpy()),
                  "test": (df_te["uid"].to_numpy(), df_te["text"].to_numpy())}
        split_tokens = load_or_build_token_cache(
            sp, tok_dir / "signsage_en_sp.model", tok_dir, splits)

    def make_loader(frame, train: bool, batch: int | None = None, token_ids=None):
        ds = SignToTextDataset(frame["uid"].to_numpy(), frame["text"].to_numpy(), sp, store, store_spec,
                               token_ids=token_ids)
        return DataLoader(ds, batch_size=batch or args.batch, shuffle=train,
                          collate_fn=collate_sign_text,
                          num_workers=2 if device.type == "cuda" else 0,
                          pin_memory=device.type == "cuda")

    # ---- model ----
    def build_model():
        return SignTransformer(MODEL_INPUT_DIM, sp.get_piece_size(), MAX_FRAMES, MAX_TOKENS,
                               pad_idx=sp.pad_id(), d_model=D_MODEL, nhead=NHEAD, layers=NLAYERS,
                               ff=FF, dropout=DROPOUT, norm_first=NORM_FIRST).to(device)

    # ---- overfit gate (mandatory) ----
    if "overfit" in stages:
        log("overfit gate: 64 memorization samples must reach >=95% exact.")
        import pandas as pd  # noqa: F811

        sub = df_tr.sample(min(64, len(df_tr)), random_state=args.seed).reset_index(drop=True)
        gate_loader = make_loader(sub, True, batch=16)
        gate = build_model()
        gate_opt = torch.optim.AdamW(gate.parameters(), lr=1e-3)
        gate_crit = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
        exact = 0.0
        for step in range(600):
            run_epoch(gate, gate_loader, gate_opt, gate_crit, device)
            if (step + 1) % 100 == 0:
                refs, hyps, _ = batch_greedy(gate, gate_loader, sp, device)
                exact = float(np.mean([normalize_text(r) == normalize_text(h) for r, h in zip(refs, hyps)]))
                log(f"  gate step {step + 1}: exact={exact:.1%}")
                if exact >= 0.95:
                    break
        if exact < 0.95:
            raise SystemExit(f"OVERFIT GATE FAILED (exact={exact:.1%}) -- fix implementation, do not train.")
        log("overfit gate PASSED.")

    # ---- train ----
    best_path = out / "checkpoints" / "signsage_transformer_full.pt"
    history: list = []
    if "train" in stages:
        train_loader = make_loader(df_tr, True, token_ids=None if args.smoke else split_tokens["train"])
        val_loader = make_loader(df_va, False, token_ids=None if args.smoke else split_tokens["val"])
        model = build_model()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = math.ceil(len(train_loader))
        scheduler = build_schedule(opt, max(1, steps_per_epoch * args.epochs))
        crit = nn.CrossEntropyLoss(ignore_index=sp.pad_id(), label_smoothing=LABEL_SMOOTHING)
        start_epoch, best, bad = 1, float("inf"), 0
        rec_path = out / "checkpoints" / "signsage_transformer_recovery.pt"
        if args.resume and rec_path.exists():
            rec = torch.load(rec_path, map_location=device, weights_only=False)
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
        div_loader_sample = make_loader(df_va.sample(min(DIVERSITY_SAMPLE, len(df_va)),
                                                      random_state=args.seed).reset_index(drop=True), False)
        for epoch in range(start_epoch, args.epochs + 1):
            tr_loss = run_epoch(model, train_loader, opt, crit, device, scheduler, train=True)
            va_loss = run_epoch(model, val_loader, opt, crit, device, train=False)
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
            log(f"epoch {epoch:02d}/{args.epochs}: train={tr_loss:.4f} val={va_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}")
            # Diversity collapse gate on real decodes (not teacher-forced loss).
            _, dhyps, _ = batch_greedy(model, div_loader_sample, sp, device)
            div = diversity_metrics(dhyps)
            log(f"  diversity: unique={div['unique']}/{len(dhyps)} ({div['unique_fraction']:.2%}) "
                f"top={div['top_prediction']!r} ({div['top_prediction_freq']:.1%})")
            if epoch >= 2 and div["unique_fraction"] < DIVERSITY_FLOOR:
                (out / "COLLAPSED.txt").write_text(
                    f"epoch {epoch}: unique_fraction={div['unique_fraction']:.4f} < {DIVERSITY_FLOOR}; "
                    f"val_loss={va_loss:.4f}; "
                    f"top={div['top_prediction']!r} ({div['top_prediction_freq']:.1%})\n")
                raise SystemExit("COLLAPSE GATE FIRED -- stopped instead of burning more GPU. See COLLAPSED.txt.")
            payload = {"state_dict": model.state_dict(), "epoch": epoch, "val_loss": va_loss, "best": best,
                       "bad": bad, "history": history,
                       "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                       "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                               "torch": torch.get_rng_state().cpu(),
                               "cuda": (torch.cuda.get_rng_state_all()
                                        if torch.cuda.is_available() else None)}}
            t_ckpt = time.perf_counter()
            torch.save(payload, rec_path)
            log(f"  recovery checkpoint saved ({time.perf_counter() - t_ckpt:.1f}s)")
            if va_loss < best - 1e-4:
                best, bad = va_loss, 0
                torch.save(payload, best_path)
                log(f"  new best (val {best:.4f})")
            else:
                bad += 1
                if bad >= args.patience:
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
    eval_report: dict = {"history": history}
    if "evaluate" in stages:
        model = build_model()
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["state_dict"])
        test_loader = make_loader(df_te, False, token_ids=None if args.smoke else split_tokens["test"])
        refs, hyps, confs = batch_greedy(model, test_loader, sp, device)
        test_metrics = compute_metrics(refs, hyps)
        div = diversity_metrics(hyps)
        pool = df_tr.loc[df_tr["text"].astype(str).str.split().str.len() >= 2, "text"].map(normalize_text)
        trivial = [pool.value_counts().idxmax()] * len(refs)
        trivial_metrics = compute_metrics(refs, trivial)
        samples = [{"ref": r, "hyp": h, "conf": c}
                   for r, h, c in zip(refs[:30], hyps[:30], confs[:30])]
        lengths = {"gen_mean": float(np.mean([len(normalize_text(h).split()) for h in hyps])),
                   "ref_mean": float(np.mean([len(normalize_text(r).split()) for r in refs]))}
        gate = {
            "beats_trivial_wer": test_metrics["WER"] < trivial_metrics["WER"],
            "beats_trivial_bleu": test_metrics["BLEU"] > trivial_metrics["BLEU"],
            "diversity_ok": div["unique_fraction"] >= DIVERSITY_FLOOR,
            "exact_match_nonzero": test_metrics["exact_match"] > 0,
        }
        gate["GO"] = all(gate.values())
        eval_report.update({"test_metrics": test_metrics, "trivial_baseline": trivial_metrics,
                            "diversity": div, "lengths": lengths, "samples": samples, "gate": gate,
                            "mean_confidence_proxy": float(np.mean(confs))})
        log(f"test: WER={test_metrics['WER']:.3f} BLEU={test_metrics['BLEU']:.2f} "
            f"unique={div['unique_fraction']:.2%} GO={gate['GO']}")
        (out / "evaluation_report.json").write_text(json.dumps(eval_report, indent=1))

    # ---- export ----
    if "export" in stages:
        model = build_model()
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["state_dict"])
        train_cfg = {"architecture": {"d_model": D_MODEL, "nhead": NHEAD, "layers": NLAYERS,
                                      "ff": FF, "dropout": DROPOUT, "norm_first": NORM_FIRST},
                     "label_smoothing": LABEL_SMOOTHING, "peak_lr": args.lr,
                     "batch": args.batch, "seed": args.seed, "history": history}
        bundle = export_bundle(model, sp, out, run_id, train_cfg, eval_report,
                               Path(args.contract) if args.contract else None)
        log(f"bundle exported: {bundle}")
    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
