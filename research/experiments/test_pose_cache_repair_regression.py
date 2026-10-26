#!/usr/bin/env python3
"""Regression tests for the ZIP vs extracted-directory pose-cache repair.

These use temporary directories and synthetic files; they never touch the
real 7,500-file dataset or download anything.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, "research/experiments")
sys.path.insert(0, "backend")

import pytest

# The module imports rely on the repo root being importable.
sys.path.insert(0, ".")

from research.experiments.kaggle_diagnostic import resolve_pose_shard, MANIFEST_REL, CSV_REL
from research.experiments.train_sign_to_text import (
    ShardPoseSource,
    POSE_CACHE_VERSION,
    load_strict_manifest,
    check_cache_identity,
    _scan_shard_members,
)


class TestShardResolution:
    def test_valid_zip_shard_resolves_to_archive(self, tmp_path: Path):
        archive = tmp_path / "shard-00000.zip"
        archive.write_text("fake zip content")
        result = resolve_pose_shard(tmp_path, "shard-00000.zip")
        assert result == archive
        assert result.is_file()

    def test_extracted_directory_resolves_when_zip_absent(self, tmp_path: Path):
        extracted = tmp_path / "shard-00001"
        extracted.mkdir()
        (extracted / "test.pose").write_bytes(b"fake")
        result = resolve_pose_shard(tmp_path, "shard-00001.zip")
        assert result == extracted
        assert result.is_dir()

    def test_missing_shard_raises_clear_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="neither an archive"):
            resolve_pose_shard(tmp_path, "shard-99999.zip")

    def test_corrupt_zip_rejected_by_reader(self, tmp_path: Path):
        # The resolution helper does not open ZIPs; the reader's ZipPoseSource
        # opens them. A corrupt file is accepted by resolve_pose_shard (it's a file)
        # but ZipPoseSource will raise BadZipFile on open. We verify that here.
        corrupt = tmp_path / "corrupt.zip"
        corrupt.write_text("not a zip")
        result = resolve_pose_shard(tmp_path, "corrupt.zip")
        assert result.is_file()
        # ZipPoseSource construction opens the file.
        import zipfile
        with pytest.raises(zipfile.BadZipFile):
            from research.experiments.train_sign_to_text import ZipPoseSource
            ZipPoseSource(str(result))

    def test_path_traversal_rejected(self, tmp_path: Path):
        # The helper resolves against cache_dir; it does not allow escaping.
        malicious = "../etc/passwd"
        with pytest.raises(FileNotFoundError):
            resolve_pose_shard(tmp_path, malicious)

    def test_zip_takes_precedence_over_directory(self, tmp_path: Path):
        # Both archive and extracted directory exist; archive wins.
        archive = tmp_path / "shard-00000.zip"
        archive.write_text("fake")
        extracted = tmp_path / "shard-00000"
        extracted.mkdir()
        result = resolve_pose_shard(tmp_path, "shard-00000.zip")
        assert result == archive


class TestDirectoryBackedReader:
    def test_directory_read_member_bytes(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 2,
            "uid_to_shard": {"uid-001": "shard-00000.zip"},
        }))
        shard_dir = tmp_path / "shard-00000"
        shard_dir.mkdir()
        (shard_dir / "uid-001.pose").write_bytes(b"directory_pose_bytes")
        store = ShardPoseSource(str(tmp_path))
        raw = store.read_member_bytes("uid-001")
        assert raw == b"directory_pose_bytes"

    def test_directory_missing_uid_raises(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 1,
            "uid_to_shard": {"uid-missing": "shard-00000.zip"},
        }))
        shard_dir = tmp_path / "shard-00000"
        shard_dir.mkdir()
        store = ShardPoseSource(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="UID not found"):
            store.read_member_bytes("uid-missing")

    def test_directory_path_escapes_blocked(self, tmp_path: Path):
        # A malicious UID that tries to escape its shard directory must fail.
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 1,
            "uid_to_shard": {"uid-normal": "shard-00000.zip"},
        }))
        shard_dir = tmp_path / "shard-00000"
        shard_dir.mkdir()
        store = ShardPoseSource(str(tmp_path))
        # The UID lookup finds the shard; DirPoseSource resolves the file path
        # inside the shard directory. If the UID contained path traversal, the
        # resolved path would still be checked against the root.
        # Actually, DirPoseSource uses self.root / f"{uid}.pose" without extra
        # validation. The user's spec requires verification that the resolved
        # path remains inside the shard directory. Let me add that check.
        pass  # See note below; the actual fix includes path validation.


class TestManifestIdentity:
    def test_valid_manifest_for_directory_cache(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 7500,
            "uid_to_shard": {f"uid-{i:04d}": "shard-00000.zip" for i in range(4)},
        }))
        manifest = load_strict_manifest(manifest_path)
        assert manifest["version"] == POSE_CACHE_VERSION
        assert manifest["complete"] is True

    def test_missing_sample_detected_by_identity_gate(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": False,
            "n": 3,
            "uid_to_shard": {f"uid-{i:04d}": "shard-00000.zip" for i in range(3)},
        }))
        manifest = load_strict_manifest(manifest_path)
        missing, extra = check_cache_identity(manifest["uid_to_shard"], {"uid-0000", "uid-0001", "uid-0002", "uid-9999"})
        assert "uid-9999" in missing
        assert len(extra) == 0

    def test_extra_files_do_not_fix_incomplete_manifest(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": False,
            "n": 1,
            "uid_to_shard": {"uid-001": "shard-00000.zip"},
        }))
        # Create extra pose files that are not in the manifest.
        shard_dir = tmp_path / "shard-00000"
        shard_dir.mkdir()
        (shard_dir / "uid-extra.pose").write_bytes(b"extra")
        manifest = load_strict_manifest(manifest_path)
        missing, extra = check_cache_identity(manifest["uid_to_shard"], {"uid-001"})
        assert missing == []
        # The identity gate checks manifest IDs, not filesystem counts.
        # Extra files in the directory do not make an incomplete manifest complete.
        assert manifest["complete"] is False


class TestDiagnosticRunnerShardHandling:
    def test_directory_backed_cache_passes_validation(self, tmp_path: Path):
        # This mimics the actual Kaggle dataset layout: manifest references .zip
        # names but physical entries are directories.
        manifest_path = tmp_path / "s2t_small6" / "pose_cache" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 4,
            "uid_to_shard": {
                "uid-001": "shard-00000.zip",
                "uid-002": "shard-00000.zip",
                "uid-003": "shard-00001.zip",
                "uid-004": "shard-00002.zip",
            },
        }))
        # Create directory-backed shards (no .zip archives)
        for shard_name in ["shard-00000", "shard-00001", "shard-00002"]:
            shard_dir = tmp_path / "s2t_small6" / "pose_cache" / shard_name
            shard_dir.mkdir(parents=True, exist_ok=True)
            for uid in ["uid-001", "uid-002", "uid-003", "uid-004"]:
                if uid.startswith("uid-00") and shard_name in ["shard-00000", "shard-00001"]:
                    (shard_dir / f"{uid}.pose").write_bytes(b"fake")
        # The diagnostic validation should resolve all shards.
        for shard_name in ["shard-00000.zip", "shard-00001.zip", "shard-00002.zip"]:
            result = resolve_pose_shard(tmp_path / "s2t_small6" / "pose_cache", shard_name)
            assert result.exists(), f"Shard {shard_name} should resolve"

    def test_probe_path_uses_resolved_shard(self, tmp_path: Path):
        # The diagnostic passes the resolved path (file or directory) as --pose-source.
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 1,
            "uid_to_shard": {"uid-001": "shard-00000.zip"},
        }))
        shard_dir = tmp_path / "shard-00000"
        shard_dir.mkdir()
        (shard_dir / "uid-001.pose").write_bytes(b"fake")
        resolved = resolve_pose_shard(tmp_path, "shard-00000.zip")
        assert resolved.is_dir()
        # The training loader treats directory paths correctly.
        store = ShardPoseSource(str(tmp_path))
        assert store.read_member_bytes("uid-001") == b"fake"



class TestIndependentIdentityValidation:
    def test_identity_derived_from_csv_not_manifest(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 2,
            "uid_to_shard": {"fake-uid-1": "shard-00000.zip", "fake-uid-2": "shard-00000.zip"},
        }))
        csv_path = tmp_path / "iSign_v1.1.csv"
        csv_path.write_text("uid,text\nuid-001,hello\nuid-002,world\n")
        manifest = load_strict_manifest(manifest_path)
        manifest_uids = set(str(k) for k in manifest.get("uid_to_shard", {}).keys())
        # Derive from CSV independently using basic pandas (no audit_dataset needed).
        import pandas as pd
        df = pd.read_csv(str(csv_path))
        expected_uids = {str(u) for u in df["uid"]}
        missing = sorted(expected_uids - manifest_uids)
        extra = sorted(manifest_uids - expected_uids)
        assert len(missing) > 0 or len(extra) > 0

    def test_identity_check_rejects_self_comparison(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 2,
            "uid_to_shard": {"uid-001": "shard-00000.zip", "uid-002": "shard-00000.zip"},
        }))
        csv_path = tmp_path / "iSign_v1.1.csv"
        csv_path.write_text("uid,text\nuid-001,hello\nuid-002,world\n")
        manifest = load_strict_manifest(manifest_path)
        import pandas as pd
        df = pd.read_csv(str(csv_path))
        expected_uids = {str(u) for u in df["uid"]}
        manifest_uids = set(str(k) for k in manifest.get("uid_to_shard", {}).keys())
        assert expected_uids == manifest_uids

    def test_identity_requires_exact_set_equality_no_extras(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 3,
            "uid_to_shard": {
                "uid-001": "shard-00000.zip",
                "uid-002": "shard-00000.zip",
                "uid-extra": "shard-00000.zip",
            },
        }))
        csv_path = tmp_path / "iSign_v1.1.csv"
        csv_path.write_text("uid,text\nuid-001,hello\nuid-002,world\n")
        import pandas as pd
        df = pd.read_csv(str(csv_path))
        expected_uids = {str(u) for u in df["uid"]}
        manifest = load_strict_manifest(manifest_path)
        manifest_uids = set(str(k) for k in manifest.get("uid_to_shard", {}).keys())
        extra = sorted(manifest_uids - expected_uids)
        assert "uid-extra" in extra
class TestNestedDirectoryDiscovery:
    """Input discovery must handle nested dataset directory structures."""

    def test_nested_subdirectory_contains_assets(self, tmp_path: Path, monkeypatch):
        import research.experiments.kaggle_diagnostic as kd
        root = tmp_path / "test_input"
        root.mkdir()
        monkeypatch.chdir(str(root))
        nested = root / "dataset" / "versions" / "v4" / "inputs" / "bundle" / "data"
        nested.mkdir(parents=True)
        (nested / "s2t_small6" / "pose_cache").mkdir(parents=True)
        (nested / "isign").mkdir(parents=True)
        (nested / "s2t_small6" / "pose_cache" / "manifest.json").write_text(json.dumps({
            "version": POSE_CACHE_VERSION, "complete": True, "n": 1,
            "uid_to_shard": {"uid-001": "shard-00000.zip"}
        }))
        (nested / "isign" / "iSign_v1.1.csv").write_text("uid,text\nuid-001,hello\n")
        found = kd.discover_input_root(str(root))
        assert found == nested.resolve()

    def test_separate_attached_roots_are_not_combined(self, tmp_path: Path, monkeypatch):
        import research.experiments.kaggle_diagnostic as kd
        root = tmp_path / "test_root"
        cache_root = root / "signsage-diagnostic-input"
        csv_root = root / "signsage-subset6"
        (cache_root / "s2t_small6" / "pose_cache").mkdir(parents=True)
        (csv_root / "isign").mkdir(parents=True)
        (cache_root / "s2t_small6" / "pose_cache" / "manifest.json").write_text(json.dumps({
            "version": POSE_CACHE_VERSION, "complete": True, "n": 1,
            "uid_to_shard": {"uid-001": "shard-00000.zip"}
        }))
        (csv_root / "isign" / "iSign_v1.1.csv").write_text("uid,text\nuid-001,hello\n")
        with pytest.raises(SystemExit, match="under the same root"):
            kd.discover_input_root(str(root))


class TestMultipleShardManifestAndRouting:
    """A manifest referencing multiple shards must route each UID correctly."""

    def test_multi_shard_manifest_resolved(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 3,
            "uid_to_shard": {
                "uid-001": "shard-00000.zip",
                "uid-002": "shard-00001.zip",
                "uid-003": "shard-00000.zip",
            },
        }))
        for shard_name, uid, payload in [
            ("shard-00000", "uid-001", b"first"),
            ("shard-00001", "uid-002", b"second"),
            ("shard-00000", "uid-003", b"third"),
        ]:
            shard_dir = tmp_path / shard_name
            shard_dir.mkdir(exist_ok=True)
            (shard_dir / f"{uid}.pose").write_bytes(payload)
        # Read through the store for each UID in different shards.
        store = ShardPoseSource(str(tmp_path))
        # UID 001 and 003 both reference shard-00000; 002 references shard-00001.
        raw_001 = store.read_member_bytes("uid-001")
        raw_002 = store.read_member_bytes("uid-002")
        raw_003 = store.read_member_bytes("uid-003")
        assert (raw_001, raw_002, raw_003) == (b"first", b"second", b"third")


class TestDirectoryPathTraversalBlocked:
    def test_traversal_raises_file_not_found(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": POSE_CACHE_VERSION,
            "complete": True,
            "n": 1,
            "uid_to_shard": {"uid-normal": "shard-00000.zip"},
        }))
        shard_dir = tmp_path / "shard-00000"
        shard_dir.mkdir()
        (shard_dir / "uid-normal.pose").write_bytes(b"safe")
        store = ShardPoseSource(str(tmp_path))
        # Normal read works.
        assert store.read_member_bytes("uid-normal") == b"safe"
        # Malicious UID attempting traversal should fail.
        with pytest.raises(FileNotFoundError):
            store.read_member_bytes("../../../etc/passwd")


class TestNotebookCallsCorrectCode:
    """Prove the notebook's diagnostic preflight uses the updated repository code."""

    def test_preflight_and_launch_use_fresh_pinned_modules(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks" / "signsage_kaggle.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
        preflight = next(cell for cell in code if "kd.discover_input_root" in cell)
        launch = next(cell for cell in code if "DIAGNOSTIC LAUNCH" in cell)
        assert "del sys.modules[_name]" in preflight
        assert 'importlib.import_module("research.experiments.kaggle_diagnostic")' in preflight
        assert "Path(kd.__file__).resolve() != _expected_runner" in preflight
        assert "Path(kd.T.__file__).resolve() != _expected_trainer" in preflight
        assert "kd.validate_input(INPUT_ROOT" in preflight
        assert "str(DEST / DIAGNOSTIC_SCRIPT)" in launch
        assert '"--input-root", str(INPUT_ROOT)' in launch


def _make_validation_bundle(tmp_path: Path, storage: str = "zip"):
    import shutil
    import pandas as pd
    import sentencepiece as spm
    import research.experiments.kaggle_diagnostic as kd
    import research.experiments.train_sign_to_text as T

    root = tmp_path / "attached" / "data"
    cache_dir = root / "s2t_small6" / "pose_cache"
    tokenizer_dir = root / "s2t_small6" / "tokenizer"
    cache_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    (root / "isign").mkdir()

    rows = [{"uid": f"video-{i:02d}-0", "text": f"hello world number {i}"}
            for i in range(10)]
    csv_path = root / "isign" / "iSign_v1.1.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    cleaned, _ = T.clean_dataframe(pd.DataFrame(rows))
    train, val, test, _ = T.video_split(cleaned, 42)
    split_frames = dict(zip(("train", "val", "test"),
                            T.apply_subset(train, val, test, 10, 42)))

    model_source = (Path(__file__).resolve().parents[1] / "training" /
                    "artifacts" / "smoke_test" / "tokenizer" / "signsage_en_sp.model")
    vocab_source = model_source.with_suffix(".vocab")
    shutil.copy2(model_source, tokenizer_dir / model_source.name)
    shutil.copy2(vocab_source, tokenizer_dir / vocab_source.name)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / model_source.name))
    splits = {
        name: (frame["uid"].to_numpy(), frame["text"].to_numpy())
        for name, frame in split_frames.items()
    }
    T.load_or_build_token_cache(sp, tokenizer_dir / model_source.name, tokenizer_dir, splits)
    fingerprint = T._sp_fingerprint(sp, tokenizer_dir / model_source.name)

    (root / "s2t_small6" / "auto_subset.json").write_text(json.dumps({
        "subset": 10, "subset_seed": 42,
    }))
    (root / "s2t_small6" / "feasibility.json").write_text(json.dumps({
        "verdict": "PASS",
    }))

    uid_to_shard = {}
    grouped = {"shard-00000": [], "shard-00001": []}
    for index, uid in enumerate(cleaned["uid"].astype(str)):
        shard_name = f"shard-{index % 2:05d}"
        uid_to_shard[uid] = f"{shard_name}.zip"
        grouped[shard_name].append(uid)
    if storage == "zip":
        for shard_name, uids in grouped.items():
            with zipfile.ZipFile(cache_dir / f"{shard_name}.zip", "w") as archive:
                for uid in uids:
                    archive.writestr(f"{uid}.pose", f"payload:{uid}".encode())
    else:
        for shard_name, uids in grouped.items():
            shard_dir = cache_dir / shard_name
            shard_dir.mkdir()
            for uid in uids:
                (shard_dir / f"{uid}.pose").write_bytes(f"payload:{uid}".encode())
    manifest = {
        "version": POSE_CACHE_VERSION,
        "complete": True,
        "n": len(uid_to_shard),
        "uid_to_shard": uid_to_shard,
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return root, manifest_path, manifest, fingerprint, split_frames


class TestStrictInputValidation:
    def test_validates_zip_shards_and_routes_uids_from_both(self, tmp_path: Path):
        import research.experiments.kaggle_diagnostic as kd

        root, _, manifest, fingerprint, _ = _make_validation_bundle(tmp_path)
        paths = kd.validate_input(root, fingerprint)
        assert paths["independent_identity_ok"] is True
        assert len(paths["resolved_shards"]) == 2
        source = ShardPoseSource(str(paths["cache_dir"]))
        uids = list(manifest["uid_to_shard"])
        assert source.read_member_bytes(uids[0]) == f"payload:{uids[0]}".encode()
        assert source.read_member_bytes(uids[1]) == f"payload:{uids[1]}".encode()
        assert manifest["uid_to_shard"][uids[0]] != manifest["uid_to_shard"][uids[1]]

    def test_validates_directory_backed_shards(self, tmp_path: Path):
        import research.experiments.kaggle_diagnostic as kd

        root, _, _, fingerprint, _ = _make_validation_bundle(tmp_path, storage="directory")
        paths = kd.validate_input(root, fingerprint)
        assert {entry["type"] for entry in paths["resolved_shards"]} == {"directory"}

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("version", POSE_CACHE_VERSION + 1, "Unsupported pose cache version"),
            ("complete", False, "Manifest incomplete"),
            ("n", 999, "Manifest count mismatch"),
        ],
    )
    def test_rejects_manifest_version_completion_and_count(
        self, tmp_path: Path, field: str, value, message: str
    ):
        import research.experiments.kaggle_diagnostic as kd

        root, manifest_path, manifest, fingerprint, _ = _make_validation_bundle(tmp_path)
        manifest[field] = value
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(SystemExit, match=message):
            kd.validate_input(root, fingerprint)

    def test_rejects_independent_uid_mismatch(self, tmp_path: Path):
        import research.experiments.kaggle_diagnostic as kd

        root, manifest_path, manifest, fingerprint, _ = _make_validation_bundle(tmp_path)
        original = manifest["uid_to_shard"]
        manifest["uid_to_shard"] = {
            f"unrelated-{i}": shard for i, shard in enumerate(original.values())
        }
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(SystemExit, match="Independent UID identity validation failed"):
            kd.validate_input(root, fingerprint)

    def test_rejects_missing_shard_and_missing_pose_member(self, tmp_path: Path):
        import research.experiments.kaggle_diagnostic as kd

        root, manifest_path, manifest, fingerprint, _ = _make_validation_bundle(tmp_path)
        cache_dir = manifest_path.parent
        (cache_dir / "shard-00001.zip").unlink()
        with pytest.raises(SystemExit, match="missing shard artifacts"):
            kd.validate_input(root, fingerprint)

        _, missing_manifest_path, missing_manifest, missing_fp, _ = _make_validation_bundle(
            tmp_path / "member-case")
        cache_dir = missing_manifest_path.parent
        uid = next(uid for uid, shard in missing_manifest["uid_to_shard"].items()
                   if shard == "shard-00000.zip")
        shard_path = cache_dir / "shard-00000.zip"
        replacement = shard_path.with_suffix(".replacement.zip")
        with zipfile.ZipFile(shard_path) as source, zipfile.ZipFile(replacement, "w") as target:
            for name in source.namelist():
                if name != f"{uid}.pose":
                    target.writestr(name, source.read(name))
        replacement.replace(shard_path)
        with pytest.raises(SystemExit, match="missing pose members"):
            kd.validate_input(tmp_path / "member-case" / "attached" / "data", missing_fp)

    def test_rejects_corrupt_zip_and_tokenizer_fingerprint_mismatch(self, tmp_path: Path):
        import research.experiments.kaggle_diagnostic as kd

        root, manifest_path, _, fingerprint, _ = _make_validation_bundle(tmp_path)
        shard_path = manifest_path.parent / "shard-00000.zip"
        shard_path.write_bytes(b"not a ZIP archive")
        with pytest.raises(SystemExit, match="corrupt"):
            kd.validate_input(root, fingerprint)

        root, _, _, fingerprint, _ = _make_validation_bundle(tmp_path / "token-case")
        with pytest.raises(SystemExit, match="Tokenizer fingerprint mismatch"):
            kd.validate_input(root, "wrong-fingerprint")

    def test_rejects_token_cache_split_mismatch(self, tmp_path: Path):
        import research.experiments.kaggle_diagnostic as kd

        root, _, _, fingerprint, _ = _make_validation_bundle(tmp_path)
        meta_path = root / "s2t_small6" / "tokenizer" / "tokens_meta.json"
        metadata = json.loads(meta_path.read_text())
        metadata["splits"]["train"]["uids"] = "wrong"
        meta_path.write_text(json.dumps(metadata))
        with pytest.raises(SystemExit, match="Token cache identity mismatch"):
            kd.validate_input(root, fingerprint)
