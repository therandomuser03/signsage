"""Locate Kaggle-notebook cells by ROLE (header comment), not by fixed position.

Several tests used to index cells positionally (cells[8], cells[11]...). Any inserted cell then
silently pointed them at the wrong code, so they validated an older layout instead of the notebook
actually being run. Roles are stable; positions are not.
"""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "signsage_kaggle.ipynb"

ROLE_MARKERS = {
    "restore": "# SESSION RESTORE",
    "provision": "# Dataset provisioning",
    "measure": "# MEASURE FIRST",
    "certify": "# CERTIFY PILOT",
    "prepare": "# PREPARE",
    "scope": "# RUN SCOPE",
    "gate": "# OVERFIT GATE ONLY",
    "training": "# V2 TRAINING",
    "evaluate": "# Final test evaluation",
    "persist": "# PERSIST",
    "final": "# FINAL VERIFICATION",
}
# the positions the pre-existing tests were written against (old layout) -> role
LEGACY_INDEX_ROLE = {8: "measure", 9: "certify", 10: "prepare", 11: "training"}


def load_notebook(path=NOTEBOOK):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def find_cell_index(nb, role):
    marker = ROLE_MARKERS[role]
    hits = [i for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and "".join(c["source"]).lstrip().startswith(marker)]
    if len(hits) != 1:
        raise AssertionError(f"role {role!r} (marker {marker!r}) matched {len(hits)} cells: {hits}")
    return hits[0]


def cell_source_by_role(nb, role):
    return "".join(nb["cells"][find_cell_index(nb, role)]["source"])


def legacy_cell_source(nb, idx):
    """Old positional lookup, resolved through the cell's role so inserted cells cannot shift it."""
    if idx in LEGACY_INDEX_ROLE:
        return cell_source_by_role(nb, LEGACY_INDEX_ROLE[idx])
    cell = nb["cells"][idx]
    if cell["cell_type"] != "code":
        raise AssertionError(f"Cell {idx} is not a code cell")
    return "".join(cell["source"])
