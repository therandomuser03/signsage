"""Regression tests for the SignSage Kaggle notebook audit findings.

Run from the repository root:   python -m pytest research/experiments/test_notebook_audit_regressions.py -q

Scope: static cell-source checks plus small functional checks. No network, no GPU,
no real iSign data, no multi-hour training.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "research/experiments")
sys.path.insert(0, "backend")
import train_sign_to_text as T  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "signsage_kaggle.ipynb"

# ---- notebook cell helpers ---------------------------------------------------

def _load_nb():
    with NOTEBOOK.open(encoding="utf-8") as fh:
        return json.load(fh)


def _cell_source(nb, marker):
    """Return the source of the unique code cell containing `marker`."""
    hits = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "\n".join(cell["source"])
        if marker in src:
            hits.append(src)
    assert len(hits) == 1, f"expected 1 code cell containing {marker!r}, got {len(hits)}"
    return hits[0]


# ---- Finding 1: git pin vs script hash --------------------------------------

STALE_PIN = "de2688f21da0dfa4c260aa1e4f6826bf0fcfb43d"


def _extract_pin_and_expected(nb):
    src = _cell_source(nb, "# Private-repo clone:")
    pin_line = [l for l in src.splitlines() if l.startswith("PINNED_SHA = ")][0]
    pin = pin_line.split("=", 1)[1].strip().strip('"')
    exp_line = [l for l in src.splitlines() if l.startswith("EXPECTED_TRAIN_SCRIPT_SHA256 = ")][0]
    expected = exp_line.split("=", 1)[1].strip().strip('"')
    return pin, expected


def test_notebook_does_not_pin_stale_commit():
    nb = _load_nb()
    pin, _expected = _extract_pin_and_expected(nb)
    assert pin != STALE_PIN, (
        f"PINNED_SHA still points at stale commit {STALE_PIN}; that commit's "
        "train_sign_to_text.py is sha 63aafe57..., not the reviewed 1531bb28...")


def test_pinned_commit_contains_reviewed_script():
    """The pinned commit's research/training/train_sign_to_text.py (symlinked to
    research/experiments/train_sign_to_text.py) must hash to EXPECTED."""
    nb = _load_nb()
    pin, expected = _extract_pin_and_expected(nb)
    full = subprocess.run(
        ["git", "rev-parse", pin], capture_output=True, text=True, check=True).stdout.strip()
    blob = subprocess.run(
        ["git", "cat-file", "-p", f"{full}:research/experiments/train_sign_to_text.py"],
        capture_output=True, check=True).stdout
    import hashlib
    actual = hashlib.sha256(blob).hexdigest()
    assert actual == expected, (
        f"pinned commit {full} script sha {actual} != expected {expected}")


# ---- Finding 2: GitHub token never in subprocess command arguments ----------

def test_clone_uses_credential_helper_not_header_in_args():
    nb = _load_nb()
    src = _cell_source(nb, "# Private-repo clone:")
    assert "http.extraHeader=Authorization: Basic" not in src, (
        "credential passed in git command arguments (visible in ps/proc)")
    assert "base64.b64encode" not in src, "token base64-encoded into command arguments"
    assert 'credential.helper=store --file' in src, "no in-memory credential helper"
    assert "os.chmod(cred_path, 0o600)" in src, "credential temp file not mode 0600"
    assert "os.unlink(cred_path)" in src, "credential temp file not removed after clone"


# ---- Findings 3 & 6: restore sentinel / _blockers / no false RESTORE VERIFIED

def test_restore_sentinel_written_only_after_verification():
    nb = _load_nb()
    src = _cell_source(nb, "# SESSION RESTORE")
    writes = src.count('_SENTINEL.write_text("restored from " + restored + "\\n")')
    assert writes == 1, f"sentinel written {writes} times (must be exactly once)"
    verify_idx = src.index("print(\"RESTORE VERIFIED: all artifacts present and compatible\")")
    sentinel_idx = src.index('_SENTINEL.write_text("restored from " + restored + "\\n")')
    assert sentinel_idx > verify_idx, "sentinel must be written after RESTORE VERIFIED, not before"


def test_restore_fail_records_blockers_and_never_prints_verified_after_failure():
    nb = _load_nb()
    src = _cell_source(nb, "# SESSION RESTORE")
    assert "_blockers.append(msg)" in src, "_fail must record in _blockers"
    blockers_idx = src.index("if _blockers:")
    verified_idx = src.index("print(\"RESTORE VERIFIED: all artifacts present and compatible\")")
    assert blockers_idx < verified_idx, (
        "_blockers must be checked before RESTORE VERIFIED is printed")


# ---- Finding 4: tokenizer provenance ----------------------------------------

def test_restore_requires_tokenizer_fingerprint_on_restore():
    nb = _load_nb()
    src = _cell_source(nb, "# SESSION RESTORE")
    assert 'restored and not EXPECTED_TOKENIZER_FINGERPRINT' in src, (
        "restored run with unpinned tokenizer fingerprint must fail closed")
    assert "fresh run; tokenizer fingerprint check skipped" in src, (
        "fresh run may skip the fingerprint check, explicitly labelled")


def test_sp_fingerprint_detects_tokenizer_mismatch():
    """Functional: _sp_fingerprint is deterministic and differs across models,
    so a restored checkpoint paired with a different tokenizer is detectable."""
    import sentencepiece as spm
    from pathlib import Path
    import tempfile

    def _mk_model(path, vocab_texts):
        spm.SentencePieceTrainer.train(
            input=str(path / "corpus.txt"), model_prefix=str(path / "m"),
            vocab_size=200, model_type="unigram", character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
            hard_vocab_limit=False, shuffle_input_sentence=False, input_sentence_size=0,
            seed_sentencepiece_size=1000000, num_threads=1)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        d1 = Path(d1); d2 = Path(d2)
        (d1 / "corpus.txt").write_text("hello world\nfoo bar\n")
        (d2 / "corpus.txt").write_text("alpha beta gamma delta epsilon\n")
        _mk_model(d1, ["hello world"]); _mk_model(d2, ["alpha beta"])
        sp1 = spm.SentencePieceProcessor(model_file=str(d1 / "m.model"))
        sp2 = spm.SentencePieceProcessor(model_file=str(d2 / "m.model"))
        fp1 = T._sp_fingerprint(sp1, d1 / "m.model")
        fp2 = T._sp_fingerprint(sp2, d2 / "m.model")
        assert fp1 == T._sp_fingerprint(spm.SentencePieceProcessor(model_file=str(d1 / "m.model")), d1 / "m.model")
        assert fp1 != fp2, "different tokenizer models must produce different fingerprints"


# ---- Finding 5: streaming training subprocess log -----------------------------

def test_training_cell_streams_subprocess_output():
    nb = _load_nb()
    src = _cell_source(nb, "# V2 TRAINING (refuses")
    assert "proc = subprocess.Popen" in src, "training must stream via Popen"
    assert 'result = subprocess.run(cmd, cwd="/kaggle/working/signsage",' not in src, (
        "training subprocess must not be captured-and-written-after-exit")
    assert "_lf.write(line" in src and "_train_out_parts.append(line)" in src, (
        "output must be written to the log in real time")


# ---- Interrupted training recovery ------------------------------------------

def _smoke_frame():
    import pandas as pd
    rows = [{"uid": f"vid{v:03d}-{s}", "text": f"sentence {v} {s}"}
            for v in range(400) for s in range(3)]
    cleaned, _ = T.clean_dataframe(pd.DataFrame(rows))
    return cleaned


def test_resume_continues_from_saved_epoch(tmp_path):
    """A recovery checkpoint persists epoch/optimizer/scheduler/RNG state and
    --resume continues from saved_epoch + 1 instead of restarting at epoch 1."""
    import torch

    df = _smoke_frame()
    out = tmp_path / "out"
    out.mkdir()

    # Full run, epochs=2: recovery checkpoint saved at the end of each epoch.
    T.main(["--smoke", "--out", str(out), "--stages", "tokenizer,overfit,train",
            "--epochs", "2", "--seed", "42"])
    rec = out / "checkpoints" / "signsage_transformer_recovery.pt"
    assert rec.is_file()
    rec1 = torch.load(rec, map_location="cpu", weights_only=False)
    assert rec1["epoch"] == 2, f"full run should reach epoch 2, got {rec1['epoch']}"
    assert len(rec1.get("history", [])) == 2

    # Resume with epochs=3 must continue from epoch 3 (saved_epoch + 1).
    T.main(["--smoke", "--out", str(out), "--stages", "tokenizer,overfit,train",
            "--epochs", "3", "--seed", "42", "--resume"])
    rec2 = torch.load(rec, map_location="cpu", weights_only=False)
    assert rec2["epoch"] == 3, f"--resume should reach epoch 3, got {rec2['epoch']}"
    assert len(rec2.get("history", [])) == 3, "--resume history should span epochs 1..3"
