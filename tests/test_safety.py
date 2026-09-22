"""Write-safety: the guard rails that the Phase 0 incident proved were missing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from multimodal_auth.errors import SafetyError
from multimodal_auth.safety import (
    OUTPUT_DIRNAME,
    PROTECTED_DIRS,
    assert_not_protected,
    file_digest,
    is_within,
    prepare_output_dir,
    snapshot_digests,
)


def test_is_within(tmp_path):
    assert is_within(tmp_path / "a" / "b", tmp_path)
    assert is_within(tmp_path, tmp_path)
    assert not is_within(tmp_path.parent / "other", tmp_path)


def test_output_dir_is_created_under_outputs(project_root, tmp_path):
    target = prepare_output_dir(project_root, "tests-sandbox", "run-1", force=True)
    try:
        assert target == (project_root / OUTPUT_DIRNAME / "tests-sandbox" / "run-1").resolve()
        assert target.is_dir()
    finally:
        import shutil

        shutil.rmtree(project_root / OUTPUT_DIRNAME / "tests-sandbox", ignore_errors=True)


@pytest.mark.parametrize("protected", ["weights", "data", "src", "configs", "tests"])
def test_protected_directories_are_refused(project_root, protected):
    """Nothing asks to write into the repository sources directly."""
    with pytest.raises(SafetyError):
        assert_not_protected(project_root / protected / "x.bin", project_root)


@pytest.mark.parametrize("protected", ["weights", "data", "src", "configs", "tests"])
def test_protected_names_are_nested_under_outputs(project_root, protected):
    """A same-named subdir is redirected into outputs/, never into the root."""
    target = prepare_output_dir(project_root, protected, None, force=True)
    try:
        assert target.is_relative_to(project_root / OUTPUT_DIRNAME)
        assert not target.is_relative_to(project_root / protected)
    finally:
        import shutil

        shutil.rmtree(project_root / OUTPUT_DIRNAME / protected, ignore_errors=True)


def test_traversal_in_subdir_is_refused(project_root):
    for bad in ("../weights", "a/../../weights", str(project_root / "weights")):
        with pytest.raises(SafetyError):
            prepare_output_dir(project_root, bad, None, force=True)


def test_traversal_in_run_name_is_refused(project_root):
    with pytest.raises(SafetyError):
        prepare_output_dir(project_root, "benchmarks", "../../weights", force=True)


def test_fails_closed_when_directory_is_not_empty(project_root):
    target = prepare_output_dir(project_root, "tests-sandbox", "non-empty", force=True)
    (target / "artifact.json").write_text("{}", encoding="utf-8")
    try:
        with pytest.raises(SafetyError, match="already contains"):
            prepare_output_dir(project_root, "tests-sandbox", "non-empty")
        # explicit opt-in still works
        again = prepare_output_dir(project_root, "tests-sandbox", "non-empty", force=True)
        assert again == target
    finally:
        import shutil

        shutil.rmtree(project_root / OUTPUT_DIRNAME / "tests-sandbox", ignore_errors=True)


def test_protected_list_covers_the_repository_sources():
    for name in ("weights", "data", "configs", "src", "tests", "app_pi"):
        assert name in PROTECTED_DIRS


def test_file_digest_matches_hashlib(tmp_path):
    payload = b"phase 1 " * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert file_digest(path) == hashlib.sha256(payload).hexdigest()


def test_snapshot_digests(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    snapshot = snapshot_digests([a, b])
    assert len(snapshot) == 2
    assert all(len(value) == 64 for value in snapshot.values())

    b.write_bytes(b"changed")
    assert snapshot_digests([b])[str(b.resolve())] != snapshot[str(b.resolve())]
