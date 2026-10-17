"""Validate the Kaggle notebook restore cell's _T import fix.

Covers: NameError: name '_T' is not defined at
`if ver != int(_T.POSE_CACHE_VERSION):` in the restore/preflight cell.
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "signsage_kaggle.ipynb"


def _restore_cell_source(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "_T.POSE_CACHE_VERSION" in src:
            return src
    raise AssertionError("restore cell with _T.POSE_CACHE_VERSION not found")


def test_notebook_json_valid():
    with NOTEBOOK.open() as f:
        nb = json.load(f)
    assert "cells" in nb
    assert len(nb["cells"]) > 0


def test_t_import_present_in_restore_cell():
    with NOTEBOOK.open() as f:
        nb = json.load(f)
    src = _restore_cell_source(nb)
    assert "import research.training.train_sign_to_text as _T" in src, (
        "missing `import research.training.train_sign_to_text as _T`"
    )


def test_sys_path_insert_before_t_import():
    with NOTEBOOK.open() as f:
        nb = json.load(f)
    src = _restore_cell_source(nb)
    lines = src.split("\n")
    t_idx = next(
        i for i, l in enumerate(lines)
        if "import research.training.train_sign_to_text as _T" in l
    )
    path_idx = next(
        i for i, l in enumerate(lines[:t_idx])
        if 'path.insert(0, "/kaggle/working/signsage")' in l
    )
    assert path_idx < t_idx, (
        "sys.path.insert must appear before the _T import"
    )


def test_pose_cache_version_references_resolved():
    with NOTEBOOK.open() as f:
        nb = json.load(f)
    src = _restore_cell_source(nb)
    assert "_T.POSE_CACHE_VERSION" in src
    # Both occurrences must reference the imported _T module
    count = src.count("_T.POSE_CACHE_VERSION")
    assert count == 2, f"expected 2 references to _T.POSE_CACHE_VERSION, found {count}"


def test_pose_cache_version_accessible_from_module():
    """End-to-end: importing the module as _T yields POSE_CACHE_VERSION."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research"))
    import research.training.train_sign_to_text as _T
    assert _T.POSE_CACHE_VERSION == 1
