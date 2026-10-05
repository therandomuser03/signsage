#!/usr/bin/env python3
"""
SignSage Full Project Audit
===========================

READ-ONLY project/dataset audit.

This script:
- Does NOT modify files
- Does NOT delete files
- Does NOT move files
- Does NOT extract archives
- Does NOT upload anything
- Does NOT recursively inspect the contents of huge datasets
- Does NOT load ML models
- Does NOT execute notebooks

It produces a Markdown report describing the current state of the
SignSage project and optionally the location of the downloaded iSign
dataset.

Tracked location: tools/signsage_full_audit.py (run from the repo root).

Usage:

    python3 tools/signsage_full_audit.py

Optional:

    python3 tools/signsage_full_audit.py --project D:\\Codes\\signsage

    python3 tools/signsage_full_audit.py ^
        --project D:\\Codes\\signsage ^
        --dataset-root D:\\Datasets

Linux:

    python3 tools/signsage_full_audit.py \
        --project /mnt/D/Codes/signsage \
        --dataset-root /mnt/D/iSign

Output:

    signsage_full_audit_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".next",
    "dist",
    "build",
    ".cache",
    ".npm",
    ".yarn",
    ".pnpm-store",
}

# These are interesting, but we don't want to recursively inspect
# their contents if they are enormous.
LARGE_DATA_EXTENSIONS = {
    ".zip",
    ".zip64",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".zst",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".onnx",
    ".bin",
}

IMPORTANT_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".csv",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
    ".html",
    ".css",
    ".lock",
}

MAX_CONTENT_SCAN_BYTES = 5 * 1024 * 1024

INTERESTING_NAME_PATTERNS = [
    r"isign",
    r"signsage",
    r"t5",
    r"pose",
    r"landmark",
    r"mediapipe",
    r"kaggle",
    r"model",
    r"checkpoint",
    r"dataset",
    r"train",
    r"eval",
    r"test",
    r"inference",
    r"adapter",
    r"translation",
    r"frontend",
    r"backend",
    r"research",
]


# ============================================================
# HELPERS
# ============================================================

def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]

    value = float(size)

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} EB"


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def safe_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        return "UNKNOWN"


def safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return -1


def is_probably_text(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_CONTENT_SCAN_BYTES:
            return False

        with path.open("rb") as f:
            data = f.read(4096)

        if b"\x00" in data:
            return False

        return True
    except Exception:
        return False


def iter_files(root: Path, ignored_dirs: set[str]) -> Iterable[Path]:
    """
    Efficient recursive traversal.

    Does not follow symlinks.
    """
    root = root.resolve()

    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [
            d for d in dirs
            if d not in ignored_dirs
            and not (Path(current) / d).is_symlink()
        ]

        for filename in files:
            path = Path(current) / filename

            try:
                if path.is_symlink():
                    continue
            except Exception:
                continue

            yield path


def run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout: int = 20,
) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )

        return result.returncode, result.stdout, result.stderr

    except Exception as exc:
        return -1, "", str(exc)


def git_status(project: Path) -> dict:
    result = {
        "is_git_repo": False,
        "branch": "",
        "status": "",
        "recent_commits": "",
        "remotes": "",
    }

    git_dir = project / ".git"

    if not git_dir.exists():
        return result

    result["is_git_repo"] = True

    code, out, err = run_command(
        ["git", "branch", "--show-current"],
        cwd=project,
    )

    if code == 0:
        result["branch"] = out.strip()

    code, out, err = run_command(
        ["git", "status", "--short"],
        cwd=project,
    )

    if code == 0:
        result["status"] = out.strip()

    code, out, err = run_command(
        [
            "git",
            "log",
            "--oneline",
            "--decorate",
            "-10",
        ],
        cwd=project,
    )

    if code == 0:
        result["recent_commits"] = out.strip()

    code, out, err = run_command(
        ["git", "remote", "-v"],
        cwd=project,
    )

    if code == 0:
        result["remotes"] = out.strip()

    return result


def classify_file(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()

    if suffix in {".ipynb"}:
        return "notebook"

    if suffix == ".py":
        return "python"

    if suffix in {".ts", ".tsx", ".js", ".jsx"}:
        return "frontend/javascript"

    if suffix in LARGE_DATA_EXTENSIONS:
        return "large-data/model/archive"

    if suffix in {".csv", ".tsv", ".parquet"}:
        return "dataset/table"

    if suffix in {".pt", ".pth", ".safetensors", ".onnx", ".bin"}:
        return "model/checkpoint"

    if suffix in {".md", ".txt"}:
        return "documentation"

    if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "configuration"

    if name in {
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "requirements.txt",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "render.yaml",
        "vercel.json",
        "readme",
    }:
        return "project/config"

    return "other"


def name_is_interesting(name: str) -> bool:
    lower = name.lower()

    return any(
        re.search(pattern, lower)
        for pattern in INTERESTING_NAME_PATTERNS
    )


# ============================================================
# DIRECTORY TREE
# ============================================================

def build_tree(root: Path, ignored_dirs: set[str], max_depth: int = 12) -> list[str]:
    lines = [f"{root.name}/"]

    def walk(directory: Path, prefix: str, depth: int):
        if depth >= max_depth:
            lines.append(prefix + "└── [MAX DEPTH REACHED]")
            return

        try:
            children = []

            for child in directory.iterdir():
                if child.name in ignored_dirs:
                    continue

                if child.is_symlink():
                    continue

                children.append(child)

            children.sort(
                key=lambda p: (
                    not p.is_dir(),
                    p.name.lower(),
                )
            )

        except Exception as exc:
            lines.append(prefix + f"└── [ACCESS ERROR: {exc}]")
            return

        for index, child in enumerate(children):
            last = index == len(children) - 1
            branch = "└── " if last else "├── "
            extension = "/" if child.is_dir() else ""

            if child.is_file():
                size = safe_size(child)
                size_text = human_size(size) if size >= 0 else "?"

                lines.append(
                    f"{prefix}{branch}{child.name} "
                    f"[{size_text}]"
                )
            else:
                lines.append(
                    f"{prefix}{branch}{child.name}{extension}"
                )

                walk(
                    child,
                    prefix + ("    " if last else "│   "),
                    depth + 1,
                )

    walk(root, "", 0)

    return lines


# ============================================================
# FILE INVENTORY
# ============================================================

def inventory_files(root: Path, ignored_dirs: set[str]) -> dict:
    files = []

    for path in iter_files(root, ignored_dirs):
        try:
            size = path.stat().st_size
        except Exception:
            size = -1

        files.append(
            {
                "path": safe_rel(path, root),
                "name": path.name,
                "suffix": path.suffix.lower(),
                "size": size,
                "mtime": safe_mtime(path),
                "classification": classify_file(path),
            }
        )

    return {
        "files": files,
        "total_files": len(files),
        "total_bytes": sum(
            x["size"]
            for x in files
            if x["size"] >= 0
        ),
    }


# ============================================================
# DUPLICATE NAME / SUSPICIOUS FILE DETECTION
# ============================================================

def duplicate_names(files: list[dict]) -> dict[str, list[str]]:
    by_name = defaultdict(list)

    for item in files:
        by_name[item["name"].lower()].append(item["path"])

    return {
        name: paths
        for name, paths in by_name.items()
        if len(paths) > 1
    }


def suspicious_files(files: list[dict]) -> list[dict]:
    result = []

    for item in files:
        name = item["name"].lower()

        reasons = []

        if re.search(r"\(\d+\)", name):
            reasons.append("numbered duplicate-style filename")

        if re.search(r"\(\d+\)\.", name):
            reasons.append("possible copied file")

        if any(
            token in name
            for token in [
                "backup",
                "old",
                "copy",
                "final",
                "final2",
                "final3",
                "new",
                "temp",
                "tmp",
                "test2",
                "fixed",
                "fix",
                "old2",
            ]
        ):
            reasons.append("possible temporary/duplicate artifact")

        if item["suffix"] in {
            ".zip",
            ".7z",
            ".rar",
            ".tar",
            ".gz",
        } and item["size"] > 10 * 1024 * 1024 * 1024:
            reasons.append("very large archive")

        if reasons:
            result.append(
                {
                    **item,
                    "reasons": reasons,
                }
            )

    return result


# ============================================================
# PROJECT CONTENT SEARCH
# ============================================================

KEYWORDS = [
    "iSign",
    "iSign_v1.1",
    "Exploration-Lab/iSign",
    "t5-small-motion",
    "t5-small",
    "RemoteConcatFile",
    "RemotePoseSource",
    "HF_TOKEN",
    "Kaggle",
    "MediaPipe",
    "543",
    "75",
    "600",
    "500",
    "pose_encoder",
    "PoseT5",
    "sign language",
    "translation",
    "landmark",
    "adapter",
]


def scan_keyword_references(
    root: Path,
    ignored_dirs: set[str],
) -> dict[str, list[str]]:
    results = defaultdict(list)

    for path in iter_files(root, ignored_dirs):
        if not is_probably_text(path):
            continue

        suffix = path.suffix.lower()

        if suffix not in {
            ".py",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ipynb",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".html",
            ".css",
            ".sh",
            ".bat",
            ".ps1",
        }:
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            continue

        for keyword in KEYWORDS:
            if keyword.lower() in text.lower():
                results[keyword].append(
                    safe_rel(path, root)
                )

    return dict(results)


# ============================================================
# DATASET ROOT AUDIT
# ============================================================

def dataset_audit(
    dataset_root: Path | None,
) -> dict:
    if dataset_root is None:
        return {
            "provided": False,
            "root": None,
            "exists": False,
        }

    dataset_root = dataset_root.expanduser().resolve()

    result = {
        "provided": True,
        "root": str(dataset_root),
        "exists": dataset_root.exists(),
        "csv_candidates": [],
        "archive_candidates": [],
        "pose_candidates": [],
        "isign_candidates": [],
        "total_visible_files": 0,
        "total_visible_bytes": 0,
    }

    if not dataset_root.exists():
        return result

    # We intentionally inspect metadata only.
    # No archive extraction.
    # No recursive traversal of .pose members inside archives.

    for current, dirs, files in os.walk(
        dataset_root,
        followlinks=False,
    ):
        dirs[:] = [
            d for d in dirs
            if d not in DEFAULT_IGNORED_DIRS
        ]

        for filename in files:
            path = Path(current) / filename

            try:
                size = path.stat().st_size
            except Exception:
                continue

            result["total_visible_files"] += 1
            result["total_visible_bytes"] += size

            lower = filename.lower()

            record = {
                "path": str(path),
                "size": size,
                "size_human": human_size(size),
            }

            if lower == "isign_v1.1.csv":
                result["csv_candidates"].append(record)

            if (
                "isign" in lower
                or "pose" in lower
            ):
                result["isign_candidates"].append(record)

            if lower.endswith(
                (
                    ".zip",
                    ".zip64",
                    ".7z",
                    ".rar",
                    ".tar",
                    ".gz",
                    ".bz2",
                    ".xz",
                )
            ):
                result["archive_candidates"].append(record)

            if lower.endswith(".pose"):
                result["pose_candidates"].append(record)

    return result


# ============================================================
# ARCHIVE METADATA INSPECTION
# ============================================================

def inspect_zip_metadata(
    path: Path,
    max_entries: int = 20,
) -> dict:
    """
    Read ZIP central-directory metadata only.

    Does NOT extract files.
    """
    import zipfile

    result = {
        "path": str(path),
        "is_zip": False,
        "readable": False,
        "entry_count": None,
        "sample_entries": [],
        "has_isign_pose_member": False,
        "error": None,
    }

    try:
        result["is_zip"] = zipfile.is_zipfile(path)

        if not result["is_zip"]:
            return result

        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()

            result["readable"] = True
            result["entry_count"] = len(names)
            result["sample_entries"] = names[:max_entries]

            result["has_isign_pose_member"] = any(
                name.startswith("iSign-poses_v1.1/")
                and name.endswith(".pose")
                for name in names[:]
            )

    except Exception as exc:
        result["error"] = repr(exc)

    return result


# ============================================================
# ENVIRONMENT / PROJECT SUMMARY
# ============================================================

def environment_summary() -> dict:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# ============================================================
# MARKDOWN REPORT
# ============================================================

def make_markdown_report(
    project: Path,
    inventory: dict,
    tree: list[str],
    git: dict,
    keyword_refs: dict,
    suspicious: list[dict],
    duplicates: dict[str, list[str]],
    dataset: dict,
    archive_inspections: list[dict],
    environment: dict,
) -> str:

    files = inventory["files"]

    by_class = Counter(
        item["classification"]
        for item in files
    )

    by_suffix = Counter(
        item["suffix"] or "[no extension]"
        for item in files
    )

    largest = sorted(
        files,
        key=lambda x: x["size"],
        reverse=True,
    )[:50]

    recent = sorted(
        files,
        key=lambda x: x["mtime"],
        reverse=True,
    )[:50]

    lines = []

    lines.append("# SignSage Full Project Audit")
    lines.append("")
    lines.append(
        f"> Generated: `{environment['timestamp']}`"
    )
    lines.append("")
    lines.append(
        "**READ-ONLY AUDIT. No project files were modified by this script.**"
    )
    lines.append("")

    # --------------------------------------------------------
    # Executive summary
    # --------------------------------------------------------

    lines.append("## 1. Executive Summary")
    lines.append("")

    lines.append(f"- Project root: `{project}`")
    lines.append(f"- Total files: **{inventory['total_files']:,}**")
    lines.append(
        f"- Total visible project size: "
        f"**{human_size(inventory['total_bytes'])}**"
    )

    if git["is_git_repo"]:
        lines.append(f"- Git repository: **YES**")
        lines.append(
            f"- Current branch: `{git['branch'] or 'UNKNOWN'}`"
        )
    else:
        lines.append("- Git repository: **NO**")

    lines.append("")

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    lines.append("## 2. Environment")
    lines.append("")

    for key, value in environment.items():
        lines.append(f"- **{key}:** `{value}`")

    lines.append("")

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    lines.append("## 3. File Classification")
    lines.append("")

    for key, value in sorted(
        by_class.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        lines.append(f"- `{key}`: **{value:,}**")

    lines.append("")

    # --------------------------------------------------------
    # Extension statistics
    # --------------------------------------------------------

    lines.append("## 4. File Extension Statistics")
    lines.append("")

    for suffix, count in by_suffix.most_common(40):
        lines.append(f"- `{suffix}`: {count:,}")

    lines.append("")

    # --------------------------------------------------------
    # Tree
    # --------------------------------------------------------

    lines.append("## 5. Project Tree")
    lines.append("")
    lines.append("```text")

    # Avoid generating a ridiculous report if tree is enormous.
    TREE_LIMIT = 10000

    if len(tree) <= TREE_LIMIT:
        lines.extend(tree)
    else:
        lines.extend(tree[:TREE_LIMIT])
        lines.append(
            f"... [{len(tree) - TREE_LIMIT:,} tree lines omitted]"
        )

    lines.append("```")
    lines.append("")

    # --------------------------------------------------------
    # Largest files
    # --------------------------------------------------------

    lines.append("## 6. Largest Project Files")
    lines.append("")
    lines.append("| Size | Classification | Path |")
    lines.append("|---:|---|---|")

    for item in largest:
        if item["size"] < 0:
            continue

        lines.append(
            f"| {human_size(item['size'])} "
            f"| `{item['classification']}` "
            f"| `{item['path']}` |"
        )

    lines.append("")

    # --------------------------------------------------------
    # Important files
    # --------------------------------------------------------

    lines.append("## 7. Important SignSage Files")
    lines.append("")

    interesting = [
        item
        for item in files
        if name_is_interesting(item["name"])
    ]

    interesting.sort(
        key=lambda x: (
            x["classification"],
            x["path"].lower(),
        )
    )

    for item in interesting[:500]:
        lines.append(
            f"- `{item['path']}` "
            f"({item['classification']}, "
            f"{human_size(item['size'])})"
        )

    if len(interesting) > 500:
        lines.append(
            f"- ... {len(interesting) - 500:,} additional matches omitted"
        )

    lines.append("")

    # --------------------------------------------------------
    # Notebooks
    # --------------------------------------------------------

    notebooks = [
        x for x in files
        if x["suffix"] == ".ipynb"
    ]

    lines.append("## 8. Notebooks")
    lines.append("")

    if notebooks:
        for item in notebooks:
            lines.append(
                f"- `{item['path']}` "
                f"({human_size(item['size'])})"
            )
    else:
        lines.append("- None found.")

    lines.append("")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = [
        x for x in files
        if x["classification"] == "model/checkpoint"
    ]

    lines.append("## 9. Models / Checkpoints")
    lines.append("")

    if models:
        for item in sorted(
            models,
            key=lambda x: x["size"],
            reverse=True,
        ):
            lines.append(
                f"- `{item['path']}` — "
                f"{human_size(item['size'])}"
            )
    else:
        lines.append("- None detected.")

    lines.append("")

    # --------------------------------------------------------
    # Archives
    # --------------------------------------------------------

    archives = [
        x for x in files
        if x["suffix"] in LARGE_DATA_EXTENSIONS
    ]

    lines.append("## 10. Archives / Large Data Files")
    lines.append("")

    if archives:
        for item in sorted(
            archives,
            key=lambda x: x["size"],
            reverse=True,
        ):
            lines.append(
                f"- `{item['path']}` — "
                f"{human_size(item['size'])}"
            )
    else:
        lines.append("- None detected.")

    lines.append("")

    # --------------------------------------------------------
    # Keyword references
    # --------------------------------------------------------

    lines.append("## 11. SignSage / iSign Keyword References")
    lines.append("")

    if keyword_refs:
        for keyword, paths in sorted(
            keyword_refs.items(),
            key=lambda x: x[0].lower(),
        ):
            lines.append(f"### `{keyword}`")
            lines.append("")

            for path in paths[:100]:
                lines.append(f"- `{path}`")

            if len(paths) > 100:
                lines.append(
                    f"- ... {len(paths) - 100:,} additional files"
                )

            lines.append("")
    else:
        lines.append("- No references detected.")

    # --------------------------------------------------------
    # Git
    # --------------------------------------------------------

    lines.append("## 12. Git State")
    lines.append("")

    if not git["is_git_repo"]:
        lines.append("Not a Git repository.")
    else:
        lines.append(f"### Branch")
        lines.append("")
        lines.append(f"`{git['branch'] or 'UNKNOWN'}`")
        lines.append("")

        lines.append("### Working Tree")
        lines.append("")
        lines.append("```text")

        if git["status"]:
            lines.append(git["status"])
        else:
            lines.append("(clean)")

        lines.append("```")
        lines.append("")

        lines.append("### Recent Commits")
        lines.append("")
        lines.append("```text")
        lines.append(git["recent_commits"] or "(none)")
        lines.append("```")
        lines.append("")

        lines.append("### Remotes")
        lines.append("")
        lines.append("```text")
        lines.append(git["remotes"] or "(none)")
        lines.append("```")
        lines.append("")

    # --------------------------------------------------------
    # Duplicate names
    # --------------------------------------------------------

    lines.append("## 13. Duplicate / Repeated Filenames")
    lines.append("")

    if duplicates:
        for name, paths in sorted(
            duplicates.items()
        ):
            lines.append(f"### `{name}`")
            lines.append("")

            for path in paths:
                lines.append(f"- `{path}`")

            lines.append("")
    else:
        lines.append("- No duplicate filenames detected.")

    # --------------------------------------------------------
    # Suspicious artifacts
    # --------------------------------------------------------

    lines.append("## 14. Possible Temporary / Duplicate Artifacts")
    lines.append("")

    if suspicious:
        for item in suspicious[:300]:
            reasons = ", ".join(item["reasons"])

            lines.append(
                f"- `{item['path']}` — {reasons}"
            )
    else:
        lines.append("- None detected.")

    lines.append("")

    # --------------------------------------------------------
    # Recent modifications
    # --------------------------------------------------------

    lines.append("## 15. Recently Modified Files")
    lines.append("")
    lines.append("| Modified | Size | Path |")
    lines.append("|---|---:|---|")

    for item in recent:
        lines.append(
            f"| `{item['mtime']}` "
            f"| {human_size(item['size'])} "
            f"| `{item['path']}` |"
        )

    lines.append("")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    lines.append("## 16. External iSign Dataset Audit")
    lines.append("")

    if not dataset.get("provided"):
        lines.append(
            "No `--dataset-root` was provided."
        )
        lines.append("")
        lines.append(
            "Run the audit again with `--dataset-root` pointing "
            "to the location containing the downloaded iSign dataset."
        )
    else:
        lines.append(
            f"- Dataset root: `{dataset['root']}`"
        )
        lines.append(
            f"- Exists: `{dataset['exists']}`"
        )
        lines.append(
            f"- Visible files: `{dataset['total_visible_files']:,}`"
        )
        lines.append(
            f"- Visible size: "
            f"`{human_size(dataset['total_visible_bytes'])}`"
        )
        lines.append("")

        lines.append("### iSign CSV Candidates")
        lines.append("")

        if dataset["csv_candidates"]:
            for item in dataset["csv_candidates"]:
                lines.append(
                    f"- `{item['path']}` — "
                    f"{item['size_human']}"
                )
        else:
            lines.append("- No `iSign_v1.1.csv` found.")

        lines.append("")

        lines.append("### Archive Candidates")
        lines.append("")

        if dataset["archive_candidates"]:
            for item in sorted(
                dataset["archive_candidates"],
                key=lambda x: x["size"],
                reverse=True,
            ):
                lines.append(
                    f"- `{item['path']}` — "
                    f"{item['size_human']}"
                )
        else:
            lines.append("- No archive candidates found.")

        lines.append("")

        lines.append("### Extracted `.pose` Files")
        lines.append("")

        if dataset["pose_candidates"]:
            lines.append(
                f"Found **{len(dataset['pose_candidates']):,}** `.pose` files."
            )

            for item in dataset["pose_candidates"][:100]:
                lines.append(
                    f"- `{item['path']}` — "
                    f"{item['size_human']}"
                )

            if len(dataset["pose_candidates"]) > 100:
                lines.append(
                    f"- ... "
                    f"{len(dataset['pose_candidates']) - 100:,} more"
                )
        else:
            lines.append(
                "No extracted `.pose` files found. "
                "They may be inside the ZIP archive."
            )

        lines.append("")

        lines.append("### iSign-Named Files")
        lines.append("")

        for item in dataset["isign_candidates"][:300]:
            lines.append(
                f"- `{item['path']}` — "
                f"{item['size_human']}"
            )

        if len(dataset["isign_candidates"]) > 300:
            lines.append(
                f"- ... "
                f"{len(dataset['isign_candidates']) - 300:,} more"
            )

        lines.append("")

    # --------------------------------------------------------
    # Archive inspection
    # --------------------------------------------------------

    lines.append("## 17. ZIP Metadata Inspection")
    lines.append("")

    if archive_inspections:
        for info in archive_inspections:
            lines.append(
                f"### `{info['path']}`"
            )
            lines.append("")

            lines.append(
                f"- ZIP detected: `{info['is_zip']}`"
            )
            lines.append(
                f"- Readable: `{info['readable']}`"
            )
            lines.append(
                f"- Entry count: `{info['entry_count']}`"
            )
            lines.append(
                f"- iSign pose member detected: "
                f"`{info['has_isign_pose_member']}`"
            )

            if info["error"]:
                lines.append(
                    f"- Error: `{info['error']}`"
                )

            if info["sample_entries"]:
                lines.append("")
                lines.append("Sample entries:")
                lines.append("")
                lines.append("```text")

                for entry in info["sample_entries"]:
                    lines.append(entry)

                lines.append("```")

            lines.append("")
    else:
        lines.append(
            "No candidate ZIP archive was inspected."
        )

    # --------------------------------------------------------
    # Interpretation hints
    # --------------------------------------------------------

    lines.append("## 18. Important Interpretation Notes")
    lines.append("")

    lines.append(
        "This report is an inventory, not a judgment of whether the "
        "project is production-ready."
    )
    lines.append("")

    lines.append(
        "The external iSign dataset is intentionally not extracted "
        "or loaded into memory by this audit."
    )
    lines.append("")

    lines.append(
        "Large archive contents are inspected only through filesystem "
        "metadata unless ZIP central-directory inspection is explicitly "
        "performed."
    )
    lines.append("")

    lines.append(
        "Do not delete or reorganize anything based on this report yet. "
        "The report should first be reviewed to establish a clean "
        "canonical SignSage structure."
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Read-only SignSage project audit."
    )

    parser.add_argument(
        "--project",
        type=str,
        default=".",
        help="SignSage project root.",
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Optional root containing the downloaded iSign dataset. "
            "This is scanned for metadata only."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Markdown report path.",
    )

    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()

    if not project.exists():
        print(f"ERROR: project does not exist: {project}")
        sys.exit(1)

    if not project.is_dir():
        print(f"ERROR: project is not a directory: {project}")
        sys.exit(1)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else project / "signsage_full_audit_report.md"
    )

    print("=" * 70)
    print("SignSage Full Project Audit")
    print("=" * 70)
    print()
    print(f"Project: {project}")
    print()
    print("READ-ONLY MODE")
    print("No files will be modified.")
    print("No archives will be extracted.")
    print()

    start = time.time()

    print("[1/8] Building project tree...")
    tree = build_tree(
        project,
        DEFAULT_IGNORED_DIRS,
    )

    print("[2/8] Inventorying project files...")
    inventory = inventory_files(
        project,
        DEFAULT_IGNORED_DIRS,
    )

    print(
        f"      Found {inventory['total_files']:,} files "
        f"({human_size(inventory['total_bytes'])})"
    )

    print("[3/8] Reading Git state...")
    git = git_status(project)

    print("[4/8] Searching SignSage/iSign references...")
    keyword_refs = scan_keyword_references(
        project,
        DEFAULT_IGNORED_DIRS,
    )

    print("[5/8] Detecting duplicate/suspicious files...")
    duplicates = duplicate_names(
        inventory["files"]
    )

    suspicious = suspicious_files(
        inventory["files"]
    )

    print("[6/8] Auditing optional dataset root...")

    dataset = dataset_audit(
        Path(args.dataset_root)
        if args.dataset_root
        else None
    )

    print("[7/8] Inspecting candidate ZIP metadata...")

    archive_inspections = []

    if dataset.get("provided"):
        for candidate in dataset.get(
            "archive_candidates",
            [],
        ):
            path = Path(candidate["path"])

            # Only inspect reasonably identifiable iSign archives.
            # Do not attempt to open every archive on the disk.
            lower = path.name.lower()

            if "isign" in lower or "pose" in lower:
                print(f"      Inspecting: {path}")
                archive_inspections.append(
                    inspect_zip_metadata(path)
                )

    print("[8/8] Generating Markdown report...")

    environment = environment_summary()

    report = make_markdown_report(
        project=project,
        inventory=inventory,
        tree=tree,
        git=git,
        keyword_refs=keyword_refs,
        suspicious=suspicious,
        duplicates=duplicates,
        dataset=dataset,
        archive_inspections=archive_inspections,
        environment=environment,
    )

    output.write_text(
        report,
        encoding="utf-8",
    )

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print()
    print(f"Report: {output}")
    print(f"Time:   {elapsed:.1f}s")
    print()
    print("Nothing was modified.")
    print()


if __name__ == "__main__":
    main()