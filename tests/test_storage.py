"""Meaningful storage failures and asymmetric pixel-grid fixtures.

Run with `python -m unittest discover -s app/tests -p test_storage.py` from
the dedicated workspace. Fixtures stay in its tmp directory.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from backend.storage import (
    Storage, StorageError, RevisionConflict, VolumeGuard, VolumeUnavailable,
    binary_mask_png, decode_image, decode_mask, encode_mask,
)


def image_bytes(array, image_format="PNG", **kwargs):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format=image_format, **kwargs)
    return output.getvalue()


class StorageTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[2]
        # Verify the real SSD before creating controlled filesystem fixtures.
        config = json.loads((workspace / "setup-notes" / "volume.json").read_text())
        VolumeGuard(config["mount"], config["uuid"]).check()
        temporary_root = workspace / "tmp" / "storage-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.mount = Path(self.temporary.name)
        self.root = self.mount / "workspace"
        self.root.mkdir()
        self.info = {"VolumeUUID": "TEST-VOLUME", "MountPoint": str(self.mount)}
        self.guard = VolumeGuard(self.mount, "TEST-VOLUME", lambda _: self.info)
        self.storage = Storage(self.root, self.guard)

    def tearDown(self):
        self.temporary.cleanup()

    def test_volume_identity_and_missing_mount_fail_closed(self):
        project = self.storage.create_project("Before disconnect")
        pointer = self.root / "projects" / project["id"] / "current.json"
        original = pointer.read_bytes()
        self.info["VolumeUUID"] = "DIFFERENT-VOLUME"
        project["name"] = "Must not persist"
        with self.assertRaises(VolumeUnavailable):
            self.storage.save_project(project["id"], project)
        self.assertEqual(pointer.read_bytes(), original)
        missing = self.mount / "disconnected-mount"
        with self.assertRaises(VolumeUnavailable):
            VolumeGuard(missing, "TEST-VOLUME", lambda _: self.info).check()
        self.assertFalse(missing.exists())

    def test_wrong_mount_path_is_rejected_without_mounted_flag(self):
        self.info["MountPoint"] = str(self.mount / "other")
        with self.assertRaises(VolumeUnavailable):
            self.storage.check()

    def test_missing_project_root_is_not_recreated(self):
        self.root.rmdir()
        with self.assertRaises(VolumeUnavailable):
            self.storage.create_project("No replacement")
        self.assertFalse(self.root.exists())

    def test_paths_reject_traversal_absolute_and_symlink_components(self):
        for candidate in ("../escape", "/tmp/escape", "images/../../escape", "images\\..\\escape"):
            with self.subTest(candidate=candidate), self.assertRaises(StorageError):
                self.storage.safe_path(candidate)
        # ExFAT cannot create real symlinks; inject a reported symlink into the
        # component walk to test the same rejection branch deterministically.
        original_is_symlink = Path.is_symlink
        forbidden = self.root / "escape"
        with mock.patch.object(Path, "is_symlink", lambda path: path == forbidden or original_is_symlink(path)):
            with self.assertRaisesRegex(StorageError, "Symlink"):
                self.storage.safe_path("escape", "nested", "asset.png")

    def test_failed_pointer_replace_preserves_last_valid_save(self):
        first = self.storage.create_project("Saved before failure")
        updated = copy.deepcopy(first)
        updated["name"] = "Uncommitted"
        actual_replace = os.replace

        def fail_pointer(source, target):
            if Path(target).name == "current.json":
                raise OSError("simulated disk write failure")
            return actual_replace(source, target)

        with mock.patch("backend.storage.os.replace", side_effect=fail_pointer):
            with self.assertRaises(OSError):
                self.storage.save_project(first["id"], updated)
        reopened = self.storage.load_project(first["id"])
        self.assertEqual(reopened, first)
        history = [path for path in (self.root / "projects" / first["id"] / "history").glob("*.json")
                   if not path.name.startswith(".")]
        self.assertEqual(len(history), 2)  # New uncommitted history is harmless.

    def test_save_restore_and_optimistic_revision(self):
        project = self.storage.create_project("山 • Landscape")
        project["runs"] = [{"id": "run-1", "prompts": {"points": [[2, 1, 1]]}, "masks": [{"id": "mask-1", "rle": encode_mask(np.eye(3, dtype=bool))}]}]
        project["style"] = {"theme": "dark", "label": "树冠", "fillOpacity": 0.3}
        saved = self.storage.save_project(project["id"], project)
        reopened = Storage(self.root, self.guard).load_project(project["id"])
        self.assertEqual(reopened, saved)
        self.assertEqual(reopened["revision"], 2)
        with self.assertRaises(RevisionConflict):
            self.storage.save_project(project["id"], project)

    def test_nonfinite_data_is_rejected_before_version_commit(self):
        project = self.storage.create_project()
        project["invalid"] = float("nan")
        with self.assertRaises(StorageError):
            self.storage.save_project(project["id"], project)
        self.assertEqual(self.storage.load_project(project["id"])["revision"], 1)

    def test_import_preserves_source_and_identity(self):
        project = self.storage.create_project()
        array = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[20, 40, 80], [11, 22, 33], [99, 88, 77]]], dtype=np.uint8)
        source = image_bytes(array)
        identity = self.storage.import_image(project["id"], "../照片.png", source)
        self.assertEqual(identity["name"], "照片.png")
        self.assertEqual(identity["sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(self.storage.original_path(project["id"], identity["id"]).read_bytes(), source)
        with Image.open(self.storage.image_path(project["id"], identity["id"])) as decoded:
            np.testing.assert_array_equal(np.array(decoded), array)
        loaded = self.storage.load_project(project["id"])
        loaded["images"][0]["width"] = 20
        with self.assertRaisesRegex(StorageError, "immutable"):
            self.storage.save_project(project["id"], loaded)

    def test_source_substitution_fails_hash_check(self):
        project = self.storage.create_project()
        identity = self.storage.import_image(project["id"], "sample.png", image_bytes(np.zeros((2, 3, 3), dtype=np.uint8)))
        source_path = self.storage.original_path(project["id"], identity["id"])
        source_path.write_bytes(b"Controlled tampering fixture")
        with self.assertRaisesRegex(StorageError, "integrity"):
            self.storage.original_path(project["id"], identity["id"])

    def test_integrity_failure_does_not_silently_load_orphan_history(self):
        project = self.storage.create_project()
        pointer = json.loads((self.root / "projects" / project["id"] / "current.json").read_text())
        history = self.root / "projects" / project["id"] / "history" / pointer["version"]
        history.write_bytes(b"{}")
        with self.assertRaisesRegex(StorageError, "integrity"):
            self.storage.load_project(project["id"])

    def test_export_stays_project_local_and_checks_volume(self):
        project = self.storage.create_project()
        path = self.storage.write_export(project["id"], "result.png", b"fixture")
        self.assertEqual(path.read_bytes(), b"fixture")
        self.assertTrue(path.is_relative_to(self.root / "exports"))
        for name in ("../original.png", "/tmp/result.png", "sub/result.png"):
            with self.assertRaises(StorageError):
                self.storage.write_export(project["id"], name, b"escape")
        self.info["VolumeUUID"] = "DIFFERENT"
        with self.assertRaises(VolumeUnavailable):
            self.storage.write_export(project["id"], "after.png", b"no")


class PixelGridTests(unittest.TestCase):
    def test_all_exif_orientations_on_asymmetric_rgb_grid(self):
        source_grid = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[20, 40, 80], [11, 22, 33], [99, 88, 77]]], dtype=np.uint8)
        for image_format, orientation in ((image_format, orientation) for image_format in ("PNG", "TIFF") for orientation in range(1, 9)):
            with self.subTest(image_format=image_format, orientation=orientation):
                exif = Image.Exif()
                exif[274] = orientation
                canonical_png, metadata = decode_image(image_bytes(source_grid, image_format, exif=exif))
                canonical = np.array(Image.open(io.BytesIO(canonical_png)))
                forward = np.array(metadata["transforms"]["original_to_canonical"])
                inverse = np.array(metadata["transforms"]["canonical_to_original"])
                np.testing.assert_array_equal(forward @ inverse, np.eye(3, dtype=int))
                for y in range(2):
                    for x in range(3):
                        canonical_x, canonical_y, _ = forward @ [x, y, 1]
                        np.testing.assert_array_equal(canonical[canonical_y, canonical_x], source_grid[y, x])
                self.assertEqual(canonical.shape[:2], (metadata["height"], metadata["width"]))
                self.assertEqual((metadata["original_width"], metadata["original_height"]), (3, 2))

    def test_transparency_composites_white_without_changing_original(self):
        array = np.array([[[255, 0, 0, 0], [0, 0, 255, 255]]], dtype=np.uint8)
        source = image_bytes(array)
        canonical, metadata = decode_image(source)
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(canonical))), [[[255, 255, 255], [0, 0, 255]]])
        self.assertEqual(metadata["transforms"]["alpha_background"], "#ffffff")
        self.assertEqual(metadata["sha256"], hashlib.sha256(source).hexdigest())

    def test_ordinary_grayscale_tiff_keeps_full_grid(self):
        array = np.array([[0, 15, 250], [70, 111, 222]], dtype=np.uint8)
        canonical, metadata = decode_image(image_bytes(array, "TIFF"))
        self.assertEqual(metadata["format"], "TIFF")
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(canonical))), np.repeat(array[..., None], 3, axis=2))

    def test_scientific_multiframe_and_invalid_images_are_rejected(self):
        scientific = image_bytes(np.array([[10, 1000], [10000, 60000]], dtype=np.uint16), "TIFF")
        with self.assertRaisesRegex(StorageError, "Scientific"):
            decode_image(scientific)
        image = Image.new("RGB", (3, 2))
        output = io.BytesIO()
        image.save(output, format="TIFF", save_all=True, append_images=[image])
        with self.assertRaisesRegex(StorageError, "multi-page"):
            decode_image(output.getvalue())
        for invalid in (b"not an image", b"", b"GIF89a"):
            with self.assertRaises(StorageError):
                decode_image(invalid)

    def test_pixel_limit_is_checked_without_silent_resizing(self):
        source = image_bytes(np.zeros((3, 4, 3), dtype=np.uint8))
        with mock.patch("backend.storage.MAX_IMAGE_PIXELS", 10):
            with self.assertRaisesRegex(StorageError, "megapixel"):
                decode_image(source)

    def test_lossless_rle_holes_disconnected_and_overlapping_instances(self):
        ring = np.array([[1, 1, 1, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 0, 1]], dtype=bool)
        overlap = np.array([[0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=bool)
        for mask in (ring, overlap, np.zeros((2, 3), dtype=bool), np.ones((2, 3), dtype=bool)):
            encoded = encode_mask(mask)
            np.testing.assert_array_equal(decode_mask(encoded), mask)
            decoded_png = np.array(Image.open(io.BytesIO(binary_mask_png(encoded))))
            np.testing.assert_array_equal(decoded_png, mask.astype(np.uint8) * 255)
            self.assertTrue(set(np.unique(decoded_png)).issubset({0, 255}))
        self.assertGreater(int((decode_mask(encode_mask(ring)) & decode_mask(encode_mask(overlap))).sum()), 0)

    def test_invalid_rle_cannot_overallocate_or_truncate(self):
        for rle in (
            {"size": [2, 3], "counts": [5]},
            {"size": [2, 3], "counts": [-1, 7]},
            {"size": [2, 3], "counts": [True, 5]},
            {"size": [2, 3], "counts": [0.0, 6]},
            {"size": [100000, 100000], "counts": [10000000000]},
            {"size": [2, 3], "counts": []},
        ):
            with self.subTest(rle=rle), self.assertRaises(StorageError):
                decode_mask(rle)


if __name__ == "__main__":
    unittest.main()
