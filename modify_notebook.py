#!/usr/bin/env python3
"""Modify the SignSage dual-model notebook in place to fix Kaggle-readiness defects."""
import json
from pathlib import Path

NOTEBOOK = Path("research/notebooks/signsage/signsage_dual_model_training.ipynb")

with NOTEBOOK.open("r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]


def cell_source(cell):
    return "".join(cell["source"])


def set_source(cell, text):
    # nbformat stores each source line with a trailing "\n". Rebuild consistently.
    lines = text.split("\n")
    if text.endswith("\n"):
        cell["source"] = [line + "\n" for line in text.split("\n")[:-1]]
    else:
        cell["source"] = [line + "\n" for line in text.split("\n")]


def find_cell(cell_id):
    for c in cells:
        if c["id"] == cell_id:
            return c
    raise KeyError(f"Cell id {cell_id!r} not found")


def insert_after(cell_id, new_cell):
    for i, c in enumerate(cells):
        if c["id"] == cell_id:
            cells.insert(i + 1, new_cell)
            return
    raise KeyError(f"Cell id {cell_id!r} not found")


def replace_cell(cell_id, old, new, label):
    cell = find_cell(cell_id)
    src = cell_source(cell)
    if old not in src:
        # Try tolerating a missing/extra trailing newline on the last line.
        alt_old = old.rstrip("\n")
        alt_src = src.rstrip("\n")
        if alt_old in alt_src:
            alt_src = alt_src.replace(alt_old, new.rstrip("\n"), 1)
            set_source(cell, alt_src + "\n")
            print(f"  applied {label} (trailing-newline tolerant)")
            return
        raise AssertionError(f"{label}: match block not found in cell {cell_id!r}")
    src = src.replace(old, new, 1)
    set_source(cell, src)
    print(f"  applied {label}")


# ---------------------------------------------------------------------------
# Fix 1: Configuration cell - explicit TRAINING_MODE validation
# ---------------------------------------------------------------------------
replace_cell(
    "configuration",
    (
        'TRAINING_MODE = "pilot"  # "pilot" | "full"\n'
        "MAX_SAMPLES_PER_SPLIT = 256 if TRAINING_MODE == \"pilot\" else 0\n"
        'TRAINING_SCOPE = "pilot_subset" if (TRAINING_MODE == "pilot" and MAX_SAMPLES_PER_SPLIT > 0) else "full_dataset"\n'
    ),
    (
        'TRAINING_MODE = "pilot"  # "pilot" | "full"\n'
        "if TRAINING_MODE not in (\"pilot\", \"full\"):\n"
        "    raise ValueError(\n"
        "        f\"Invalid TRAINING_MODE={TRAINING_MODE!r}; expected 'pilot' or 'full'. \"\n"
        "        \"Invalid modes must not silently fall back to unlimited training.\"\n"
        "    )\n"
        "MAX_SAMPLES_PER_SPLIT = 256 if TRAINING_MODE == \"pilot\" else 0  # 0 = no limit\n"
        'TRAINING_SCOPE = "pilot_subset" if TRAINING_MODE == "pilot" else "full_dataset"\n'
    ),
    "Fix 1: TRAINING_MODE validation",
)

# ---------------------------------------------------------------------------
# Fix 2: Insert environment-report + dependency-readiness cell after config
# ---------------------------------------------------------------------------
env_cell = {
    "cell_type": "code",
    "id": "environment-report",
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import platform\n",
        "import shutil\n",
        "import sys\n",
        "\n",
        "def _gib(bytes_value):\n",
        "    try:\n",
        "        return float(bytes_value) / (1024 ** 3)\n",
        "    except Exception:\n",
        "        return float(\"nan\")\n",
        "\n",
        "print(\"=\" * 70)\n",
        "print(\"KAGGLE ENVIRONMENT REPORT\")\n",
        "print(\"=\" * 70)\n",
        "print(f\"Python version: {platform.python_version()}\")\n",
        "print(f\"PyTorch version: {torch.__version__}\")\n",
        "print(f\"NumPy version: {np.__version__}\")\n",
        "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
        "if torch.cuda.is_available():\n",
        "    print(f\"GPU count: {torch.cuda.device_count()}\")\n",
        "    for i in range(torch.cuda.device_count()):\n",
        "        try:\n",
        "            props = torch.cuda.get_device_properties(i)\n",
        "            print(f\"  GPU {i}: {props.name} | {_gib(props.total_memory):.2f} GiB memory\")\n",
        "        except Exception as exc:\n",
        "            print(f\"  GPU {i}: unable to query ({exc})\")\n",
        "else:\n",
        "    print(\"GPU: none (CPU-only session)\")\n",
        "try:\n",
        "    usage = shutil.disk_usage(\"/\")\n",
        "    print(f\"Disk: {_gib(usage.free):.2f} GiB free of {_gib(usage.total):.2f} GiB total\")\n",
        "except Exception as exc:\n",
        "    print(f\"Disk: unable to determine ({exc})\")\n",
        "try:\n",
        "    page = os.sysconf(\"SC_PAGE_SIZE\")\n",
        "    pages = os.sysconf(\"SC_PHYS_PAGES\")\n",
        "    print(f\"System RAM: {_gib(page * pages):.2f} GiB\")\n",
        "except Exception as exc:\n",
        "    print(f\"System RAM: unable to determine ({exc})\")\n",
        "print(f\"Kaggle input dir (/kaggle/input): {Path('/kaggle/input').exists()}\")\n",
        "if Path('/kaggle/input').exists():\n",
        "    try:\n",
        "        print(f\"  contents: {[p.name for p in Path('/kaggle/input').iterdir()]}\")\n",
        "    except Exception as exc:\n",
        "        print(f\"  list error: {exc}\")\n",
        "print(f\"Kaggle working dir (/kaggle/working): {Path('/kaggle/working').exists()}\")\n",
        "print(f\"Output root: {OUTPUT_ROOT.resolve()}\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "# Dependency readiness: install only what is missing; never reinstall torch/CUDA.\n",
        "import importlib\n",
        "\n",
        "print(\"Dependency readiness:\")\n",
        "for module, package in ((\"numpy\", \"numpy\"), (\"torch\", \"torch\")):\n",
        "    try:\n",
        "        importlib.import_module(module)\n",
        "        print(f\"  required present: {module}\")\n",
        "    except ImportError:\n",
        "        raise RuntimeError(\n",
        "            f\"Missing required dependency '{package}'. Install it before training.\"\n",
        "        )\n",
        "\n",
        "optional_by_feature = {\n",
        "    \"sentencepiece\": \"Model B translation tokenizer (pip install sentencepiece)\",\n",
        "    \"pose-format\": \"Model A/Model B .pose file parsing (pip install pose-format)\",\n",
        "    \"cv2\": \"Raw-video landmark extraction (pip install opencv-python-headless)\",\n",
        "    \"mediapipe\": \"Raw-video landmark extraction (pip install mediapipe)\",\n",
        "}\n",
        "for module, purpose in optional_by_feature.items():\n",
        "    try:\n",
        "        importlib.import_module(module)\n",
        "        name = purpose.split(\" (\")[0]\n",
        "        print(f\"  optional present: {module} ({name})\")\n",
        "    except ImportError:\n",
        "        print(f\"  optional missing: {module} - {purpose}\")\n",
        "print(\"=\" * 70)\n",
    ],
}
insert_after("configuration", env_cell)
print("  inserted Fix 2: environment-report cell")

# ---------------------------------------------------------------------------
# Fix 3: Insert dataset-discovery cell after data-functions cell
# ---------------------------------------------------------------------------
discovery_cell = {
    "cell_type": "code",
    "id": "dataset-discovery",
    "metadata": {},
    "outputs": [],
    "source": [
        "def discover_dataset_files(root):\n",
        "    root = Path(root)\n",
        "    found = {\"csv\": [], \"pose\": [], \"npy\": [], \"npz\": [], \"video\": []}\n",
        "    if not root.exists():\n",
        "        return found\n",
        "    for p in sorted(root.rglob(\"*\")):\n",
        "        if not p.is_file():\n",
        "            continue\n",
        "        suf = p.suffix.lower()\n",
        "        if suf == \".csv\":\n",
        "            found[\"csv\"].append(p)\n",
        "        elif suf == \".pose\":\n",
        "            found[\"pose\"].append(p)\n",
        "        elif suf == \".npy\":\n",
        "            found[\"npy\"].append(p)\n",
        "        elif suf == \".npz\":\n",
        "            found[\"npz\"].append(p)\n",
        "        elif suf in (\".mp4\", \".mov\", \".avi\", \".mkv\"):\n",
        "            found[\"video\"].append(p)\n",
        "    return found\n",
        "\n",
        "\n",
        "def _rel(path, root):\n",
        "    try:\n",
        "        return path.relative_to(root).as_posix()\n",
        "    except ValueError:\n",
        "        return str(path)\n",
        "\n",
        "\n",
        "def _sample_landmark_shape(path):\n",
        "    try:\n",
        "        arr = load_landmarks(path)\n",
        "        return tuple(arr.shape)\n",
        "    except Exception as exc:\n",
        "        return f\"ERROR: {exc}\"\n",
        "\n",
        "\n",
        "print(\"=\" * 70)\n",
        "print(\"DATASET DISCOVERY PREFLIGHT\")\n",
        "print(\"=\" * 70)\n",
        "search_roots = [Path(\"/kaggle/input\")]\n",
        "if Path(\"/kaggle/working\").exists():\n",
        "    search_roots.append(Path(\"/kaggle/working\"))\n",
        "all_stems = []\n",
        "for root in search_roots:\n",
        "    print(f\"\\nScanning root: {root}\")\n",
        "    found = discover_dataset_files(root)\n",
        "    for kind in (\"csv\", \"pose\", \"npy\", \"npz\", \"video\"):\n",
        "        paths = found[kind]\n",
        "        print(f\"  {kind}: {len(paths)} files\")\n",
        "        for p in paths[:8]:\n",
        "            print(f\"    - {_rel(p, root)}\")\n",
        "        if len(paths) > 8:\n",
        "            print(f\"    ... and {len(paths) - 8} more\")\n",
        "        if kind == \"pose\":\n",
        "            all_stems.extend(p.stem for p in paths)\n",
        "\n",
        "    # Inspect every discovered CSV manifest (fields + row count + empties).\n",
        "    for csv_path in found[\"csv\"]:\n",
        "        try:\n",
        "            rows, fields = read_csv_rows(csv_path)\n",
        "            print(f\"\\n  Manifest: {_rel(csv_path, root)}\")\n",
        "            print(f\"    fields: {fields}\")\n",
        "            print(f\"    rows: {len(rows)}\")\n",
        "            empty_text = sum(\n",
        "                1 for r in rows if not str(r.get(\"text\", \"\")).strip()\n",
        "            )\n",
        "            empty_label = sum(\n",
        "                1 for r in rows if not str(r.get(\"label\", \"\")).strip()\n",
        "            )\n",
        "            print(f\"    empty text: {empty_text}; empty label: {empty_label}\")\n",
        "            sample_cols = [\n",
        "                c\n",
        "                for c in (\n",
        "                    \"path\",\n",
        "                    \"pose_path\",\n",
        "                    \"pose_file\",\n",
        "                    \"label\",\n",
        "                    \"text\",\n",
        "                    \"video_id\",\n",
        "                    \"signer\",\n",
        "                    \"split\",\n",
        "                    \"video_ref\",\n",
        "                )\n",
        "                if c in fields\n",
        "            ]\n",
        "            for r in rows[:3]:\n",
        "                row_summary = {k: r.get(k) for k in sample_cols}\n",
        "                print(f\"    row: {row_summary}\")\n",
        "        except Exception as exc:\n",
        "            print(f\"\\n  Manifest {csv_path}: read error - {exc}\")\n",
        "\n",
        "# Duplicate pose-file stems across the scanned roots (possible duplicate samples).\n",
        "duplicates = [s for s, count in Counter(all_stems).items() if count > 1]\n",
        "print(f\"\\nDuplicate pose-file stems across roots: {len(duplicates)}\")\n",
        "for s in duplicates[:10]:\n",
        "    print(f\"  - {s}\")\n",
        "\n",
        "# Sample a few pose/npy/npz files to confirm landmark dimensions.\n",
        "checked = 0\n",
        "for root in search_roots:\n",
        "    found = discover_dataset_files(root)\n",
        "    for kind in (\"pose\", \"npy\", \"npz\"):\n",
        "        for p in found[kind][:3]:\n",
        "            shape = _sample_landmark_shape(p)\n",
        "            print(f\"  shape {_rel(p, root)}: {shape}\")\n",
        "            checked += 1\n",
        "            if checked >= 6:\n",
        "                break\n",
        "        if checked >= 6:\n",
        "            break\n",
        "    if checked >= 6:\n",
        "        break\n",
        "\n",
        "print(\"\\nNext: set ISOLATED_MANIFEST (Model A) and/or TRANSLATION_MANIFEST (Model B)\")\n",
        "print(\"      in the Configuration cell to the selected manifest path.\")\n",
        "print(\"=\" * 70)\n",
    ],
}
insert_after("data-functions", discovery_cell)
print("  inserted Fix 3: dataset-discovery cell")

# ---------------------------------------------------------------------------
# Fix 4: limit_split - enforce MIN_SPLIT_SIZE
# ---------------------------------------------------------------------------
replace_cell(
    "data-functions",
    "def limit_split(rows, limit, seed):\n",
    (
        "MIN_SPLIT_SIZE = 2  # after limiting; raise if a split would fall below this\n"
        "\n"
        "\n"
        "def limit_split(rows, limit, seed, split_name=\"split\"):\n"
    ),
    "Fix 4a: MIN_SPLIT_SIZE constant",
)
replace_cell(
    "data-functions",
    (
        "    if not limit or len(rows) <= limit:\n"
        "        return rows\n"
        "    return random.Random(seed).sample(rows, limit)\n"
    ),
    (
        "    if not limit or len(rows) <= limit:\n"
        "        return rows\n"
        "    sampled = random.Random(seed).sample(rows, limit)\n"
        "    if len(sampled) < MIN_SPLIT_SIZE:\n"
        "        raise ValueError(\n"
        "            f\"Limiting {split_name} to {limit} produced {len(sampled)} samples, \"\n"
        "            f\"below MIN_SPLIT_SIZE={MIN_SPLIT_SIZE}. Increase MAX_SAMPLES_PER_SPLIT \"\n"
        "            \"or use full mode.\"\n"
        "        )\n"
        "    return sampled\n"
    ),
    "Fix 4b: limit_split MIN_SPLIT_SIZE enforcement",
)

# ---------------------------------------------------------------------------
# Fix 5: Make matplotlib optional in run-recognition cell
# ---------------------------------------------------------------------------
replace_cell(
    "run-recognition",
    (
        "if TRAIN_RECOGNITION:\n"
        "    import matplotlib.pyplot as plt\n"
        "    history = RECOGNITION_REPORT[\"history\"]\n"
        "    plt.plot([x[\"epoch\"] for x in history], [x[\"train_loss\"] for x in history], label=\"train loss\")\n"
        "    plt.plot([x[\"epoch\"] for x in history], [x[\"validation_loss\"] for x in history], label=\"validation loss\")\n"
        "    plt.xlabel(\"Epoch\")\n"
        "    plt.ylabel(\"Cross-entropy\")\n"
        "    plt.legend()\n"
        "    plt.show()\n"
        "    print(\"Training accuracy is separate from held-out validation/test metrics.\")\n"
        "    print(json.dumps(RECOGNITION_REPORT, indent=2))\n"
    ),
    (
        "if TRAIN_RECOGNITION:\n"
        "    try:\n"
        "        import matplotlib.pyplot as plt\n"
        "        history = RECOGNITION_REPORT[\"history\"]\n"
        "        plt.plot([x[\"epoch\"] for x in history], [x[\"train_loss\"] for x in history], label=\"train loss\")\n"
        "        plt.plot([x[\"epoch\"] for x in history], [x[\"validation_loss\"] for x in history], label=\"validation loss\")\n"
        "        plt.xlabel(\"Epoch\")\n"
        "        plt.ylabel(\"Cross-entropy\")\n"
        "        plt.legend()\n"
        "        plt.show()\n"
        "    except ImportError:\n"
        "        print(\"matplotlib not installed; skipping training loss plot.\")\n"
        "    print(\"Training accuracy is separate from held-out validation/test metrics.\")\n"
        "    print(json.dumps(RECOGNITION_REPORT, indent=2))\n"
    ),
    "Fix 5: optional matplotlib in run-recognition",
)

# ---------------------------------------------------------------------------
# Fix 6: Make matplotlib optional in run-translation cell
# ---------------------------------------------------------------------------
replace_cell(
    "run-translation",
    (
        "if TRAIN_TRANSLATION:\n"
        "    import matplotlib.pyplot as plt\n"
        "    history = json.loads((OUTPUT_ROOT / \"translation\" / \"training_history.json\").read_text(encoding=\"utf-8\"))\n"
        "    plt.plot([x[\"epoch\"] for x in history], [x[\"train_loss\"] for x in history], label=\"train loss\")\n"
        "    plt.plot([x[\"epoch\"] for x in history], [x[\"validation_loss\"] for x in history], label=\"validation loss\")\n"
        "    plt.xlabel(\"Epoch\")\n"
        "    plt.ylabel(\"Masked token cross-entropy\")\n"
        "    plt.legend()\n"
        "    plt.show()\n"
        "    print(json.dumps(TRANSLATION_REPORT, indent=2))\n"
    ),
    (
        "if TRAIN_TRANSLATION:\n"
        "    try:\n"
        "        import matplotlib.pyplot as plt\n"
        "        history = json.loads((OUTPUT_ROOT / \"translation\" / \"training_history.json\").read_text(encoding=\"utf-8\"))\n"
        "        plt.plot([x[\"epoch\"] for x in history], [x[\"train_loss\"] for x in history], label=\"train loss\")\n"
        "        plt.plot([x[\"epoch\"] for x in history], [x[\"validation_loss\"] for x in history], label=\"validation loss\")\n"
        "        plt.xlabel(\"Epoch\")\n"
        "        plt.ylabel(\"Masked token cross-entropy\")\n"
        "        plt.legend()\n"
        "        plt.show()\n"
        "    except ImportError:\n"
        "        print(\"matplotlib not installed; skipping training loss plot.\")\n"
        "    print(json.dumps(TRANSLATION_REPORT, indent=2))\n"
    ),
    "Fix 6: optional matplotlib in run-translation",
)

# ---------------------------------------------------------------------------
# Write back
# ---------------------------------------------------------------------------
with NOTEBOOK.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Notebook modified successfully.")
print(f"Cells: {len(cells)}")
print("Cell order:", [c["id"] for c in cells])
