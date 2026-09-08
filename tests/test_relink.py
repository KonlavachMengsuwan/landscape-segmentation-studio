"""Exact-byte missing-image recovery; all fixtures stay on the verified SSD."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
from PIL import Image
import pytest

from backend.storage import Storage, StorageError, VolumeGuard, VolumeUnavailable, encode_mask


@pytest.fixture
def recovery_case():
    workspace = Path(__file__).resolve().parents[2]
    configuration = json.loads((workspace / "setup-notes" / "volume.json").read_text())
    VolumeGuard(configuration["mount"], configuration["uuid"]).check()
    fixtures = workspace / "tmp" / "relink-tests"
    fixtures.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=fixtures) as temporary:
        mount = Path(temporary)
        root = mount / "workspace"
        root.mkdir()
        info = {"VolumeUUID": "RELINK-FIXTURE", "MountPoint": str(mount)}
        storage = Storage(root, VolumeGuard(mount, "RELINK-FIXTURE", lambda _: info))
        array = np.array([[[255, 0, 0], [0, 200, 0], [0, 0, 190]],
                          [[27, 61, 130], [115, 14, 54], [11, 122, 231]]], dtype=np.uint8)
        exif = Image.Exif()
        exif[274] = 6
        output = io.BytesIO()
        Image.fromarray(array).save(output, format="PNG", exif=exif)
        source = output.getvalue()
        project = storage.create_project("Preserve recovery history")
        identity = storage.import_image(project["id"], "Original 山.png", source)
        project = storage.load_project(project["id"])
        project["runs"] = [{"id": "recovery-run", "image_id": identity["id"],
                            "prompts": {"points": [{"x": 1, "y": 0, "label": 1}]},
                            "masks": [{"id": "mask-a", "original_rle": encode_mask(np.eye(3, 2, dtype=bool)),
                                       "label": "Reviewed surface", "status": "accepted"}]}]
        project["style"] = {"fillOpacity": .32, "paletteSeed": 41}
        project = storage.save_project(project["id"], project)
        base = root / "projects" / project["id"]
        paths = {"original": base / identity["original_path"], "canonical": base / identity["canonical_path"]}
        yield storage, info, project, identity, source, paths


@pytest.mark.parametrize("removed", [("original",), ("canonical",), ("original", "canonical")])
def test_restores_missing_assets_without_changing_identity_runs_or_revision(recovery_case, removed):
    storage, info, project, identity, source, paths = recovery_case
    prior_assets = {name: path.read_bytes() for name, path in paths.items()}
    pointer = storage.root / "projects" / project["id"] / "current.json"
    pointer_bytes = pointer.read_bytes()
    for name in removed:
        paths[name].unlink()
    returned = storage.relink_image(project["id"], identity["id"], "renamed-exact-copy.png", source)
    assert returned == identity
    assert returned["name"] == "Original 山.png"
    assert {name: path.read_bytes() for name, path in paths.items()} == prior_assets
    assert storage.load_project(project["id"]) == project
    assert pointer.read_bytes() == pointer_bytes
    assert hashlib.sha256(paths["original"].read_bytes()).hexdigest() == identity["sha256"]
    with Image.open(paths["canonical"]) as restored:
        assert restored.size == (identity["width"], identity["height"]) == (2, 3)


def test_same_filename_different_bytes_is_refused_before_any_write(recovery_case):
    storage, info, project, identity, source, paths = recovery_case
    for path in paths.values():
        path.unlink()
    with mock.patch.object(storage, "_write_atomic", side_effect=AssertionError("Unexpected write")):
        with pytest.raises(StorageError, match="SHA256"):
            storage.relink_image(project["id"], identity["id"], identity["name"], source + b"different bytes")
    assert all(not path.exists() for path in paths.values())
    assert storage.load_project(project["id"]) == project


def test_existing_correct_assets_are_byte_identical_and_never_rewritten(recovery_case):
    storage, info, project, identity, source, paths = recovery_case
    before = {name: (path.read_bytes(), path.stat().st_mtime_ns) for name, path in paths.items()}
    with mock.patch.object(storage, "_write_atomic", side_effect=AssertionError("Unexpected write")):
        assert storage.relink_image(project["id"], identity["id"], "selected.png", source) == identity
    assert {name: (path.read_bytes(), path.stat().st_mtime_ns) for name, path in paths.items()} == before


@pytest.mark.parametrize("corrupt", ["original", "canonical"])
def test_corrupt_existing_asset_is_not_overwritten_and_prevents_partial_relink(recovery_case, corrupt):
    storage, info, project, identity, source, paths = recovery_case
    missing = "canonical" if corrupt == "original" else "original"
    paths[missing].unlink()
    paths[corrupt].write_bytes(b"Controlled corrupt asset fixture")
    with mock.patch.object(storage, "_write_atomic", side_effect=AssertionError("Unexpected write")):
        with pytest.raises(StorageError, match="will not overwrite"):
            storage.relink_image(project["id"], identity["id"], identity["name"], source)
    assert not paths[missing].exists()
    assert paths[corrupt].read_bytes() == b"Controlled corrupt asset fixture"


def test_missing_drive_rejects_relink_without_creating_assets(recovery_case):
    storage, info, project, identity, source, paths = recovery_case
    paths["canonical"].unlink()
    info["VolumeUUID"] = "SIMULATED-MISSING-OR-WRONG-SSD"
    with pytest.raises(VolumeUnavailable):
        storage.relink_image(project["id"], identity["id"], identity["name"], source)
    assert not paths["canonical"].exists()


@pytest.mark.parametrize("field", ["original", "canonical"])
def test_asset_symlink_is_rejected_even_if_content_would_match(recovery_case, field):
    storage, info, project, identity, source, paths = recovery_case
    # ExFAT cannot create normal symlinks. Exercise the real rejection branch
    # by reporting the controlled asset as a symlink, as in storage tests.
    original = Path.is_symlink
    with mock.patch.object(Path, "is_symlink", lambda path: path == paths[field] or original(path)):
        with pytest.raises(StorageError, match="Symlink"):
            storage.relink_image(project["id"], identity["id"], identity["name"], source)


def test_changed_canonical_decode_is_refused_even_when_original_hash_matches(recovery_case):
    storage, info, project, identity, source, paths = recovery_case
    paths["canonical"].unlink()
    from backend.storage import decode_image
    canonical, altered = decode_image(source)
    altered["width"] += 1
    with mock.patch("backend.storage.decode_image", return_value=(canonical, altered)):
        with pytest.raises(StorageError, match="canonical hash or pixel grid"):
            storage.relink_image(project["id"], identity["id"], identity["name"], source)
    assert not paths["canonical"].exists()


def test_partial_write_failure_preserves_project_and_can_be_retried(recovery_case):
    storage, info, project, identity, source, paths = recovery_case
    for path in paths.values():
        path.unlink()
    actual = storage._write_atomic
    def fail_canonical(relative, payload, **kwargs):
        if Path(relative).name == "canonical.png":
            raise OSError("simulated canonical write failure")
        return actual(relative, payload, **kwargs)
    with mock.patch.object(storage, "_write_atomic", side_effect=fail_canonical):
        with pytest.raises(OSError, match="simulated"):
            storage.relink_image(project["id"], identity["id"], identity["name"], source)
    assert paths["original"].read_bytes() == source
    assert not paths["canonical"].exists()
    assert storage.load_project(project["id"]) == project
    assert storage.relink_image(project["id"], identity["id"], identity["name"], source) == identity
    assert paths["canonical"].exists()
