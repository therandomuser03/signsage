
#!/usr/bin/env python3
"""Generate a text tree of the SignSage repository.

Run from the project root:
    python3 scan_signsage_structure.py
    python3 scan_signsage_structure.py --all
    python3 scan_signsage_structure.py /path/to/SignSage --all
"""

from pathlib import Path
from datetime import datetime
import argparse
import os

SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", ".next", "dist", "build", "coverage",
    ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".turbo",
    ".cache", ".parcel-cache",
}

SKIP_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}


def human_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def main():
    parser = argparse.ArgumentParser(
        description="Scan a SignSage project and generate a text tree."
    )
    parser.add_argument(
        "root", nargs="?", default=".",
        help="SignSage project root (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output", default="signsage_project_structure.txt",
        help="Output text file"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include .git, node_modules, caches, build folders, etc."
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    ignored_dirs = set() if args.all else SKIP_DIRS
    lines = [
        "SIGNSAGE — PROJECT STRUCTURE",
        f"Root: {root}",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Mode: {'ALL FILES' if args.all else 'SOURCE-FOCUSED; generated directories skipped'}",
        "",
        f"{root.name}/",
    ]

    stats = {
        "files": 0, "dirs": 0, "links": 0,
        "bytes": 0, "skipped_dirs": 0, "skipped_files": 0,
    }
    skipped_paths = []

    def walk(directory, prefix=""):
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda p: (
                    not (p.is_dir() and not p.is_symlink()),
                    p.name.casefold(),
                ),
            )
        except (PermissionError, OSError) as exc:
            lines.append(f"{prefix}└── [UNREADABLE DIRECTORY: {exc}]")
            return

        visible = []
        for child in children:
            if child.resolve(strict=False) == output:
                continue
            if child.name in SKIP_FILES:
                stats["skipped_files"] += 1
                continue
            if (
                child.is_dir()
                and not child.is_symlink()
                and child.name in ignored_dirs
            ):
                stats["skipped_dirs"] += 1
                skipped_paths.append(child.relative_to(root).as_posix() + "/")
                continue
            visible.append(child)

        for index, child in enumerate(visible):
            last = index == len(visible) - 1
            branch = "└── " if last else "├── "
            continuation = "    " if last else "│   "

            if child.is_symlink():
                try:
                    target = os.readlink(child)
                except OSError:
                    target = "unknown"
                label = f"{child.name}@ -> {target}"
                stats["links"] += 1

            elif child.is_dir():
                label = child.name + "/"
                stats["dirs"] += 1

            else:
                try:
                    size = child.stat().st_size
                    label = f"{child.name} ({human_size(size)})"
                    stats["bytes"] += size
                except OSError:
                    label = child.name + " [size unavailable]"
                stats["files"] += 1

            lines.append(prefix + branch + label)

            if child.is_dir() and not child.is_symlink():
                walk(child, prefix + continuation)

    walk(root)

    lines += [
        "",
        "SUMMARY",
        "=======",
        f"Directories: {stats['dirs']}",
        f"Files: {stats['files']}",
        f"Symlinks: {stats['links']}",
        f"Total listed file size: {human_size(stats['bytes'])}",
        f"Skipped generated directories: {stats['skipped_dirs']}",
        f"Skipped OS metadata files: {stats['skipped_files']}",
    ]

    if skipped_paths:
        lines += ["", "SKIPPED DIRECTORIES", "==================="]
        lines.extend(f"- {path}" for path in skipped_paths)
        lines.append("Use --all to include these directories.")

    lines += [
        "",
        "NOTES",
        "=====",
        "This report contains paths and file sizes, not file contents.",
        "No project files were deleted or modified.",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nReport created: {output}")
    print(f"Files: {stats['files']}")
    print(f"Directories: {stats['dirs']}")
    print(f"Symlinks: {stats['links']}")
    print(f"Listed size: {human_size(stats['bytes'])}")
    if stats["skipped_dirs"]:
        print(f"Skipped directories: {stats['skipped_dirs']} (use --all to include)")
    print("No project files were deleted or modified.")


if __name__ == "__main__":
    main()
