"""Focused tests for measurement cell indentation fix.

Tests that cell 8 (the MEASURE cell) of signsage_kaggle.ipynb:
1. Compiles as valid Python after the indentation fix
2. Has the REUSE LOCK / else structure properly indented
3. Does not have column-0 code after the outer else: block
4. Preserves the intended control flow logic:
   - If _reuse_lock is True, skip measurement
   - Otherwise, run measurement and write auto_subset.json
   - No safe subset => training must not start
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "signsage_kaggle.ipynb"


def _load_notebook():
    with NOTEBOOK.open() as f:
        return json.load(f)


def _cell8_source(nb):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _notebook_cells import cell_source_by_role
    return cell_source_by_role(nb, "measure")


def _find_outer_else_idx(lines):
    """Find the index of the outer else: that follows the REUSE LOCK check."""
    # First find the REUSE LOCK if block
    reuse_lock_if_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("if globals().get(\"_reuse_lock\")"):
            reuse_lock_if_idx = i
            break
    if reuse_lock_if_idx is None:
        return None

    # Find the else: that comes after this if block
    for i in range(reuse_lock_if_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "else:":
            return i
    return None


def test_cell8_compiles():
    """Cell 8 must compile as valid Python after the indentation fix."""
    nb = _load_notebook()
    src = _cell8_source(nb)
    compile(src, "<cell8_measure>", "exec")


def test_else_block_indented():
    """The else: block after the REUSE LOCK check must have its body
    properly indented (not at column 0)."""
    nb = _load_notebook()
    src = _cell8_source(nb)
    lines = src.split("\n")

    outer_else_idx = _find_outer_else_idx(lines)
    assert outer_else_idx is not None, "Outer else: block not found"

    # The next non-blank, non-comment line after else: must be indented
    for j in range(outer_else_idx + 1, len(lines)):
        nl = lines[j]
        if not nl.strip():
            continue
        if nl.strip().startswith("#"):
            continue
        indent = len(nl) - len(nl.lstrip())
        assert indent > 0, (
            f"Line {j+1}: first code line after else: is at column 0 "
            f"(expected indent > 0): {nl.strip()[:60]}"
        )
        break


def test_no_column0_code_after_else():
    """No code line after the outer else: should be at column 0."""
    nb = _load_notebook()
    src = _cell8_source(nb)
    lines = src.split("\n")

    outer_else_idx = _find_outer_else_idx(lines)
    assert outer_else_idx is not None, "Outer else: block not found"

    issues = []
    for i in range(outer_else_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            issues.append(f"Line {i+1}: column 0 - {line.strip()[:60]}")

    assert not issues, (
        f"Found {len(issues)} column-0 code lines after outer else:\n"
        + "\n".join(issues)
    )


def test_reuse_lock_structure_preserved():
    """The REUSE LOCK check must still be present with proper structure:
    if _reuse_lock is True: print(...) else: <measurement code>"""
    nb = _load_notebook()
    src = _cell8_source(nb)

    assert "_reuse_lock" in src, "Must reference _reuse_lock"
    assert "REUSE LOCK" in src, "Must mention REUSE LOCK"
    assert "skipping measure/auto-subset" in src, (
        "Must mention skipping measure/auto-subset"
    )
    assert "left untouched" in src, (
        "Must mention auto_subset.json left untouched"
    )


def test_auto_subset_written():
    """auto_subset.json must be written in the appropriate branches."""
    nb = _load_notebook()
    src = _cell8_source(nb)

    assert "auto_subset.json" in src, "Must reference auto_subset.json"
    assert 'open(OUT_DIR + "/auto_subset.json", "w")' in src or \
           'open(OUT_DIR + \'/auto_subset.json\', \'w\')' in src, (
        "Must write auto_subset.json"
    )


def test_no_safe_subset_guard():
    """When no safe subset is found, training must not start."""
    nb = _load_notebook()
    src = _cell8_source(nb)

    assert "NO SAFE SUBSET FOUND" in src or "no safe subset" in src.lower(), (
        "Must warn about no safe subset"
    )
    assert "do NOT start training" in src or "do NOT run prepare/train" in src, (
        "Must explicitly prevent training when no safe subset"
    )


def test_measurement_output_paths():
    """All output paths must use s2t_small6 (not s2t_v2)."""
    nb = _load_notebook()
    src = _cell8_source(nb)
    lines = src.split("\n")

    assert "/kaggle/working/s2t_small6" in src, (
        "Must use s2t_small6 output directory"
    )

    # Check non-comment lines for s2t_v2
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "s2t_v2" not in stripped, (
            f"Line {i+1}: code references s2t_v2: {stripped[:60]}"
        )


if __name__ == "__main__":
    import sys
    tests = [
        test_cell8_compiles,
        test_else_block_indented,
        test_no_column0_code_after_else,
        test_reuse_lock_structure_preserved,
        test_auto_subset_written,
        test_no_safe_subset_guard,
        test_measurement_output_paths,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except (AssertionError, SyntaxError) as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
