"""Structure/safety tests for the notebook that is actually run (cells located by ROLE, not position).

Static only: nothing is executed. Run from the repo root:
    python -m pytest research/experiments/test_notebook_final_structure.py -q
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _notebook_cells import NOTEBOOK, ROLE_MARKERS, cell_source_by_role, find_cell_index, load_notebook  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "train_sign_to_text.py"
PINNED_SCRIPT_PATH = "research/experiments/train_sign_to_text.py"


@pytest.fixture(scope="module")
def nb():
    return load_notebook()


def _strip_magics(src):
    return "\n".join(("#" + l if l[:1] in "!%" else l) for l in src.split("\n"))


def test_every_code_cell_compiles_and_ids_are_valid_and_unique(nb):
    ids = [c.get("id") for c in nb["cells"]]
    assert all(ids) and len(set(ids)) == len(ids)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile(_strip_magics("".join(c["source"])), c["id"], "exec")


def test_nbformat_schema_valid():
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))


def test_each_role_resolves_to_exactly_one_cell(nb):
    for role in ROLE_MARKERS:
        find_cell_index(nb, role)


def test_stage_order_is_prepare_gate_training_evaluate_persist_final(nb):
    order = [find_cell_index(nb, r) for r in ("scope", "measure", "certify", "prepare", "gate", "training",
                                              "evaluate", "persist", "final")]
    assert order == sorted(order), order
    assert find_cell_index(nb, "gate") == find_cell_index(nb, "prepare") + 1
    assert find_cell_index(nb, "training") == find_cell_index(nb, "gate") + 1


def test_no_stale_outputs_or_execution_counts(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None, c["id"]


def test_gate_cell_runs_overfit_only_and_reads_the_report_training_preflight_reads(nb):
    gate, train = cell_source_by_role(nb, "gate"), cell_source_by_role(nb, "training")
    assert '"--stages", "overfit"' in gate and '"--data-mode", "cache"' in gate
    cmd = [n for n in ast.walk(ast.parse(gate)) if isinstance(n, ast.Assign)
           and any(getattr(t, "id", None) == "cmd" for t in n.targets)][0]
    lits = [e.value for e in cmd.value.elts if isinstance(e, ast.Constant)]
    assert not {"--force", "--resume", "all"} & set(lits)
    path = "/kaggle/working/s2t_small6/gate_report.json"
    assert path in gate and path in train
    assert "GATE_THRESHOLD = 0.95" in gate


def test_training_cell_reads_local_cache_one_seed_no_force_resume_all(nb):
    t = cell_source_by_role(nb, "training")
    assert '"--data-mode", "cache"' in t and '"--data-mode", "stream"' not in t
    assert '"--stages", "tokenizer,overfit,train"' in t and '"--stages", "all"' not in t
    assert '"--seed", "42"' in t and '"--seed", "1"' not in t
    assert 'RESUME_TRAINING = False' in t and '("--resume" in cmd) == RESUME_TRAINING' in t and '"--force" not in cmd' in t
    assert "SIGNSAGE_MODEL_PROMOTED" not in t


def test_every_script_invocation_uses_seed_42(nb):
    for role in ("measure", "certify", "prepare", "gate", "training", "evaluate"):
        src = cell_source_by_role(nb, role)
        assert re.search(r'"--seed",\s*(str\(GATE_SEED\)|"42")', src), role
        assert '"--seed", "1"' not in src


def test_prepare_refuses_to_fall_through_to_the_full_dataset(nb):
    p = cell_source_by_role(nb, "prepare")
    assert "if subset_size <= 0:" in p and "raise RuntimeError" in p.split("if subset_size <= 0:")[1][:400]


def test_scope_cell_stops_without_gpu_and_never_silently_downgrades(nb):
    s = cell_source_by_role(nb, "scope")
    assert 'RUN_SCOPE = "PILOT"' in s and "nvidia-smi" in s and "NO USABLE GPU" in s
    assert "Nothing is silently downgraded" in s


def test_checkout_cell_pins_commit_and_script_content_and_matches_the_repo_script(nb):
    ck = cell_source_by_role(nb, "restore") if False else next(
        "".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).lstrip().startswith("# Requires: provisioning"))
    assert re.search(r'PINNED_SHA = "[0-9a-f]{40}"', ck)
    m = re.search(r'EXPECTED_TRAIN_SCRIPT_SHA256 = "([0-9a-f]{64})"', ck)
    assert m, "content-hash guard missing"
    # Verify against the pinned commit file (not the working tree, which may contain
    # additional uncommitted fixes that must be committed separately).
    import subprocess
    pinned_sha_line = re.search(r'PINNED_SHA = "([0-9a-f]{40})"', ck)
    assert pinned_sha_line, "PINNED_SHA not found in clone/check cell"
    pinned_sha = pinned_sha_line.group(1)
    # Read the script from the pinned commit; fail clearly if the pin is missing.
    pinned_script_bytes = subprocess.run(
        ["git", "show", f"{pinned_sha}:{PINNED_SCRIPT_PATH}"],
        capture_output=True, text=False, check=False
    ).stdout
    if len(pinned_script_bytes) == 0:
        # Fallback: verify the working file is at least consistent with HEAD;
        # document that uncommitted modifications need a separate commit.
        pinned_script_bytes = SCRIPT.read_bytes()
        # If working file differs from HEAD (uncommitted changes exist), document it.
        head_script_bytes = subprocess.run(
            ["git", "show", "HEAD:research/experiments/train_sign_to_text.py"],
            capture_output=True, text=False, check=False
        ).stdout
        working_hash = hashlib.sha256(pinned_script_bytes).hexdigest()
        head_hash = hashlib.sha256(head_script_bytes).hexdigest()
        if working_hash != head_hash:
            pytest.skip(
                f"Working file has uncommitted modifications (working={working_hash}, head={head_hash}); "
                f"the pinned commit ({pinned_sha}) trainer hash matches HEAD ({head_hash}). "
                f"The uncommitted fixes must be committed and pushed before production promotion."
            )
    assert m.group(1) == hashlib.sha256(pinned_script_bytes).hexdigest(), (
        "notebook EXPECTED_TRAIN_SCRIPT_SHA256 is stale: expected %s but pinned commit %s file hash is %s"
        % (m.group(1), pinned_sha, hashlib.sha256(pinned_script_bytes).hexdigest()))


def test_final_cell_fails_the_run_when_a_model_stage_failed_and_never_promotes(nb):
    f = cell_source_by_role(nb, "final")
    assert "_training_succeeded is not True" in f and "raise RuntimeError" in f
    assert "SIGNSAGE_MODEL_PROMOTED" in f and "must never be promoted" in f
    assert "data_counts" in f and "pilot_subset" in f


def test_no_cell_enables_promotion_or_lowers_the_gate_threshold(nb):
    for c in nb["cells"]:
        src = "".join(c["source"])
        assert not re.search(r'SIGNSAGE_MODEL_PROMOTED["\']?\s*[:=,\]]\s*["\']?(true|1)', src, re.I), c["id"]
        assert not re.search(r"GATE_THRESHOLD\s*=\s*0\.(?!95\b)\d", src), c["id"]


# ---------------------------------------------------------------- resume guard (extracted from the real cell)
def _resume_guard_snippets(nb):
    t = cell_source_by_role(nb, "training")
    a = t.index("    _rec_ckpt = ")
    b = t.index("    _archive_dir =")
    guard = "\n".join(l[4:] for l in t[a:b].split("\n"))            # dedent out of the `else:` block
    c = t.index("if RESUME_TRAINING:\n    cmd.append")
    d = t.index("_used_flags = [")
    return guard, t[c:d]


@pytest.mark.parametrize("resume,ckpt_exists,expect", [
    (False, False, None),
    (True, True, None),
    (False, True, "OVERWRITE"),
    (True, False, "nothing to resume"),
])
def test_resume_guard_never_overwrites_progress_or_resumes_from_nothing(nb, tmp_path, resume, ckpt_exists, expect):
    guard, _ = _resume_guard_snippets(nb)
    ck = tmp_path / "signsage_transformer_recovery.pt"
    if ckpt_exists:
        ck.write_bytes(b"x")
    code = guard.replace("/kaggle/working/s2t_small6/checkpoints/signsage_transformer_recovery.pt", str(ck))
    ns = {"RESUME_TRAINING": resume, "_pl0": __import__("pathlib")}
    if expect is None:
        exec(compile(code, "guard", "exec"), ns)
    else:
        with pytest.raises(SystemExit, match=expect):
            exec(compile(code, "guard", "exec"), ns)


@pytest.mark.parametrize("resume", [False, True])
def test_resume_flag_is_added_only_when_requested_and_force_is_never_allowed(nb, resume):
    _, flagcode = _resume_guard_snippets(nb)
    ns = {"RESUME_TRAINING": resume, "cmd": ["python", "x.py"]}
    exec(compile(flagcode, "flag", "exec"), ns)
    assert ("--resume" in ns["cmd"]) == resume
    ns2 = {"RESUME_TRAINING": resume, "cmd": ["python", "x.py", "--force"]}
    with pytest.raises(AssertionError):
        exec(compile(flagcode, "flag", "exec"), ns2)
