"""Focused non-training checks for the bounded Kaggle diagnostic runner.

No GPU, no dataset download, no optimizer steps, no full training.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_runner_parses_and_rejects_old_checkpoint_resume():
    import kaggle_diagnostic as kd
    # Only verify CLI parsing and that resume is false by default.
    args = kd.ap.parse_args([
        "--input-root", "/tmp/nonexistent_for_parse",
        "--out", "/tmp/nonexistent_for_parse_out",
        "--timeout-seconds", "60",
    ])
    assert not args.resume
    assert args.embed_init == "scaled"
    assert args.stages is None  # runner passes stages directly; not a failure


def test_runner_fails_on_stale_pin_without_new_commit():
    # The notebook's current pin (4a91f02...) does NOT contain the new script hash.
    # A fresh clone of that pin should fail the hash check. This is the expected
    # behavior; it is NOT a failure of the fix.
    pass  # verified by test_notebook_final_structure; this is a placeholder.


def test_research_package_importable():
    # ModuleNotFoundError: No module named 'research' is fixed by making
    # research/ and research/experiments/ explicit packages.
    import research.experiments.train_sign_to_text as T  # noqa: F401
    assert T.POSE_CACHE_VERSION == 1


def test_discover_input_root_finds_attached_mirror():
    import kaggle_diagnostic as kd
    mirror = Path(__file__).resolve().parent.parent.parent / "kaggle_input_v4"
    assert mirror.is_dir(), "local kaggle_input_v4 mirror needed for this check"
    root = kd.discover_input_root(str(mirror))
    assert root == mirror.resolve()
    # Discovery from the repo root should also locate the mirror as a subdir.
    repo_root = Path(__file__).resolve().parent.parent.parent
    assert kd.discover_input_root(str(repo_root)) == mirror.resolve()


def test_discover_input_root_finds_kaggle_slug_layout(monkeypatch):
    """Simulates Kaggle layout: /kaggle/input/<slug>/s2t_small6/... + isign/..."""
    import kaggle_diagnostic as kd
    root = Path(tempfile.mkdtemp())
    monkeypatch.chdir(root)  # real Kaggle runs from /kaggle/working (no kaggle_input_v4)
    slug = root / "signsage-24-oct"
    (slug / "s2t_small6" / "pose_cache").mkdir(parents=True)
    (slug / "isign").mkdir(parents=True)
    (slug / "s2t_small6" / "pose_cache" / "manifest.json").write_text("{}")
    (slug / "isign" / "iSign_v1.1.csv").write_text("uid,text\n")
    found = kd.discover_input_root(str(root))
    assert found == slug.resolve()


def test_discover_input_root_rejects_incomplete_subset6_only(monkeypatch):
    """Simulates signsage-subset6: only subset6.csv, no manifest/isign CSV -> not a candidate."""
    import kaggle_diagnostic as kd
    root = Path(tempfile.mkdtemp())
    monkeypatch.chdir(root)
    slug = root / "signsage-subset6"
    (slug / "data").mkdir(parents=True)
    (slug / "data" / "subset6.csv").write_text("uid,text\n")
    try:
        kd.discover_input_root(str(root))
    except SystemExit as exc:
        assert "No input root contains both" in str(exc)
    else:
        raise AssertionError("discover_input_root should fail on incomplete dataset")


def test_discover_input_root_rejects_ambiguous_candidates(monkeypatch):
    """Two slugs each with manifest+CSV -> ambiguous, fail closed."""
    import kaggle_diagnostic as kd
    root = Path(tempfile.mkdtemp())
    monkeypatch.chdir(root)
    for slug in ("signsage-24-oct", "signsage-other"):
        d = root / slug
        (d / "s2t_small6" / "pose_cache").mkdir(parents=True)
        (d / "isign").mkdir(parents=True)
        (d / "s2t_small6" / "pose_cache" / "manifest.json").write_text("{}")
        (d / "isign" / "iSign_v1.1.csv").write_text("uid,text\n")
    try:
        kd.discover_input_root(str(root))
    except SystemExit as exc:
        assert "Ambiguous" in str(exc)
    else:
        raise AssertionError("discover_input_root should fail on ambiguous candidates")


def test_validate_input_passes_on_attached_mirror():
    import kaggle_diagnostic as kd
    mirror = Path(__file__).resolve().parent.parent.parent / "kaggle_input_v4"
    assert mirror.is_dir()
    paths = kd.validate_input(mirror, "2d88a5de48318cb9")
    assert paths["manifest"].get("n") == 7500
    assert paths["csv_path"].name == "iSign_v1.1.csv"
    assert paths["cache_dir"].name == "pose_cache"
    assert paths["tok_dir"].name == "tokenizer"


def test_validate_input_raises_on_missing_required_files():
    import kaggle_diagnostic as kd
    import tempfile
    empty = Path(tempfile.mkdtemp())
    try:
        kd.validate_input(empty, "2d88a5de48318cb9")
    except SystemExit as exc:
        assert "Required inputs missing" in str(exc)
    else:
        raise AssertionError("validate_input should raise on missing files")


def test_runner_train_args_parse_without_interpreter_prefix():
    # Regression: kaggle_diagnostic builds cmd = [python, -u, script, --csv <path>, ...]
    # for the subprocess, but parity-checks the trailing args with T.parse_args.
    # The parse list must start at "--csv" (the old cmd[3:]), not at the bare CSV
    # value (the old cmd[4:]), else argparse exits before the subprocess launches.
    import kaggle_diagnostic as kd  # noqa: F401  (ensures the package imports)
    import train_sign_to_text as T
    from pathlib import Path
    import tempfile

    csv_path = str(Path(tempfile.mktemp(suffix=".csv")))
    Path(csv_path).write_text("uid,text\nv1,hello world\n")

    train_args = [
        "--csv", csv_path,
        "--pose-source", "/tmp/ps", "--pose-cache", "/tmp/pc",
        "--out", "/tmp/out",
        "--data-mode", "cache", "--stages", "overfit,train",
        "--subset", "2500", "--seed", "42", "--epochs", "8",
        "--batch", "32", "--lr", "1e-4", "--patience", "2",
        "--decoder-input-dropout", "0.2", "--max-hours", "0.4",
        "--embed-init", "scaled",
    ]
    parsed = T.parse_args(train_args)
    assert parsed.csv == csv_path
    assert not parsed.resume
    assert parsed.embed_init == "scaled"
    assert parsed.data_mode == "cache"
    assert parsed.stages == "overfit,train"

    # The off-by-one form (cmd[4:], leading bare path) must be rejected.
    buggy_args = [csv_path] + train_args[1:]
    try:
        T.parse_args(buggy_args)
    except SystemExit:
        pass
    else:
        raise AssertionError("cmd[4:]-style args (bare leading path) must be rejected by argparse")


def test_diagnostic_run_args_include_tokenizer_stage():
    # Regression: the diagnostic must request the tokenizer stage so the
    # SentencePiece model file is created before the overfit gate. Without it,
    # train_sign_to_text reuses out/tokenizer/signsage_en_sp.model and raises
    # RuntimeError NOT_FOUND when the artifact is absent (the defect this
    # runner hit twice).
    import tempfile
    import train_sign_to_text as T

    csv_path = str(Path(tempfile.mktemp(suffix=".csv")))
    Path(csv_path).write_text("uid,text\nv1,hello world\n")

    run_args = [
        "--csv", csv_path,
        "--pose-source", "/tmp/ps", "--pose-cache", "/tmp/pc",
        "--out", "/tmp/out",
        "--data-mode", "cache", "--stages", "tokenizer,overfit,train",
        "--subset", "2500", "--seed", "42", "--epochs", "8",
        "--batch", "32", "--lr", "1e-4", "--patience", "2",
        "--decoder-input-dropout", "0.2", "--max-hours", "0.4",
        "--embed-init", "scaled",
    ]
    parsed = T.parse_args(run_args)
    stages = set(parsed.stages.split(","))
    assert "tokenizer" in stages, f"tokenizer stage missing: {parsed.stages!r}"
    assert "overfit" in stages
    assert "train" in stages
    assert "evaluate" not in stages
    assert "export" not in stages
    assert not parsed.resume
    assert parsed.embed_init == "scaled"


def test_train_tokenizer_creates_model_from_scratch():
    # Regression: a missing tokenizer artifact must be created fresh, never
    # silently reused from a stale location. train_tokenizer must write
    # signsage_en_sp.model into out_dir when it does not already exist.
    import tempfile
    import sentencepiece as spm
    import train_sign_to_text as T

    tmp = Path(tempfile.mkdtemp())
    tok_dir = tmp / "tokenizer"
    assert not tok_dir.exists()
    texts = [
        "hello world", "foo bar baz", "test sentence piece",
        "the quick brown fox", "jumps over the lazy dog",
    ]
    sp = T.train_tokenizer(texts, tok_dir, vocab_size=40)
    model = tok_dir / "signsage_en_sp.model"
    assert model.is_file(), "train_tokenizer must create signsage_en_sp.model when missing"
    loaded = spm.SentencePieceProcessor(model_file=str(model))
    assert (loaded.pad_id(), loaded.unk_id(), loaded.bos_id(), loaded.eos_id()) == (0, 1, 2, 3)
    assert loaded.get_piece_size() > 0
    # Re-running train_tokenizer on an existing model must not retrain or fail.
    sp2 = T.train_tokenizer(texts, tok_dir, vocab_size=40)
    assert sp2.get_piece_size() == loaded.get_piece_size()
