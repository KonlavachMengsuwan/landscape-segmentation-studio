"""Bundle membership, alpha, audit, CSV, and hostile ZIP fixtures."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image

from backend.exports import build_export, import_bundle
from backend.storage import Storage, StorageError, VolumeGuard, encode_mask, decode_mask


def png(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def archive_bytes(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue()


def refresh_manifest(files):
    manifest = json.loads(files["manifest.json"])
    manifest["files"] = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                         for name, data in files.items() if name != "manifest.json"}
    files["manifest.json"] = json.dumps(manifest).encode()


class ExportTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[2]
        config = json.loads((workspace / "setup-notes" / "volume.json").read_text())
        VolumeGuard(config["mount"], config["uuid"]).check()
        temporary_root = workspace / "tmp" / "export-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        mount = Path(self.temporary.name)
        root = mount / "workspace"
        root.mkdir()
        guard = VolumeGuard(mount, "TEST-VOLUME", lambda _: {"VolumeUUID": "TEST-VOLUME", "MountPoint": str(mount)})
        self.storage = Storage(root, guard)
        self.source = png(np.arange(45, dtype=np.uint8).reshape(3, 5, 3))
        self.project = self.storage.create_project("Synthetic export fixture")
        self.image = self.storage.import_image(self.project["id"], "asymmetric.png", self.source)
        self.project = self.storage.load_project(self.project["id"])
        self.original = np.array([[1, 1, 1, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=bool)
        self.edited = self.original.copy()
        self.edited[2, 4] = True
        self.run = {"id": "run-A", "image_id": self.image["id"], "model": "fixture-only", "checkpoint": "synthetic-test",
                    "settings": {"points_per_side": 8}, "prompts": {"points": [[1, 1, 1]]},
                    "masks": [
                        {"id": "mask-A", "run_id": "run-A", "label": "=SUM(1,2)", "status": "accepted", "visible": True,
                         "original_rle": encode_mask(self.original), "edited_rle": encode_mask(self.edited), "score": .8, "stability": None, "area": 999},
                        {"id": "mask-hidden", "run_id": "run-A", "label": "Hidden overlap", "status": "proposed", "visible": False,
                         "original_rle": encode_mask(self.original), "edited_rle": None, "score": None},
                        {"id": "mask-rejected", "run_id": "run-A", "label": " \t@SUM(A1)", "status": "rejected", "visible": True,
                         "original_rle": encode_mask(self.original), "edited_rle": None, "score": None},
                    ]}
        self.project["runs"] = [self.run]
        self.project["style"] = {"fillOpacity": .25, "outlineWidth": 2, "background": "white"}
        self.project["selectedImageId"] = self.image["id"]
        self.project["selectedRunId"] = self.run["id"]
        self.project["selectedMaskIds"] = ["mask-A"]
        self.project["ui"] = {"imageId": self.image["id"], "runId": self.run["id"], "selectedIds": ["mask-A"]}
        self.project = self.storage.save_project(self.project["id"], self.project)
        rgba = np.zeros((3, 5, 4), dtype=np.uint8)
        rgba[self.edited] = (20, 110, 90, 96)
        self.overlay = png(rgba)
        self.styled = png(np.full((3, 5, 3), 210, dtype=np.uint8))

    def tearDown(self):
        self.temporary.cleanup()

    def bundle(self):
        path = build_export(self.storage, self.project, "run-A", self.styled, self.overlay)
        with zipfile.ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        return path, files

    def test_segment_numbers_match_complete_run_even_when_middle_mask_is_hidden(self):
        _, files = self.bundle()
        rows = list(csv.DictReader(io.StringIO(files["metadata.csv"].decode("utf-8-sig"))))
        self.assertEqual([(row["mask_id"], row["segment_index"]) for row in rows],
                         [("mask-A", "1"), ("mask-rejected", "3")])

    def test_full_grid_binary_masks_originals_and_alpha_are_exact(self):
        path, files = self.bundle()
        self.assertTrue(path.is_relative_to(self.storage.root / "exports"))
        self.assertEqual(files[self.image["original_path"]], self.source)
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(files["masks/mask-A.png"]))), self.edited.astype(np.uint8) * 255)
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(files["masks-original/mask-A.png"]))), self.original.astype(np.uint8) * 255)
        self.assertNotIn("masks/mask-hidden.png", files)
        self.assertIn("masks/mask-rejected.png", files)
        self.assertEqual(files["overlay.png"], self.overlay)
        overlay = np.array(Image.open(io.BytesIO(files["overlay.png"])))
        np.testing.assert_array_equal(overlay[..., 3], self.edited.astype(np.uint8) * 96)
        self.assertEqual(len(json.loads(files["audit/run.json"])["masks"]), 3)
        manifest = json.loads(files["manifest.json"])
        self.assertIn("Independent instance", manifest["overlap_policy"])
        self.assertEqual(set(manifest["files"]), set(files) - {"manifest.json"})
        for name, record in manifest["files"].items():
            self.assertEqual(record["sha256"], hashlib.sha256(files[name]).hexdigest())

    def test_csv_neutralizes_formulas_and_counts_effective_mask(self):
        _, files = self.bundle()
        rows = list(csv.DictReader(io.StringIO(files["metadata.csv"].decode("utf-8-sig"))))
        self.assertEqual(rows[0]["label"], "'=SUM(1,2)")
        self.assertEqual(rows[0]["reviewed_label"], "'=SUM(1,2)")
        self.assertEqual(int(rows[0]["pixel_count"]), int(self.edited.sum()))
        self.assertEqual(int(rows[0]["original_pixel_count"]), int(self.original.sum()))
        self.assertEqual(rows[0]["stability_score"], "")
        self.assertEqual(rows[1]["label"], "' \t@SUM(A1)")
        self.assertEqual(json.loads(files["audit/run.json"])["masks"][0]["label"], "=SUM(1,2)")

    def test_deleted_masks_are_audited_but_not_exported_as_visible(self):
        self.project["runs"][0]["masks"][0]["deleted"] = True
        path, files = self.bundle()
        self.assertNotIn("masks/mask-A.png", files)
        self.assertNotIn("mask-A", json.loads(files["manifest.json"])["mask_ids"])
        self.assertTrue(json.loads(files["audit/run.json"])["masks"][0]["deleted"])
        imported = import_bundle(self.storage, path.read_bytes())
        self.assertTrue(imported["runs"][0]["masks"][0]["deleted"])

    def test_bundle_roundtrip_creates_new_ids_and_preserves_audit(self):
        path, _ = self.bundle()
        imported = import_bundle(self.storage, path.read_bytes())
        self.assertNotEqual(imported["id"], self.project["id"])
        self.assertNotEqual(imported["images"][0]["id"], self.image["id"])
        run = imported["runs"][0]
        self.assertNotEqual(run["id"], self.run["id"])
        self.assertEqual(run["provenance"]["source_run_id"], self.run["id"])
        self.assertNotEqual(run["masks"][0]["id"], "mask-A")
        self.assertEqual(run["masks"][0]["label"], "=SUM(1,2)")
        self.assertEqual(run["settings"], self.run["settings"])
        np.testing.assert_array_equal(decode_mask(run["masks"][0]["original_rle"]), self.original)
        np.testing.assert_array_equal(decode_mask(run["masks"][0]["edited_rle"]), self.edited)
        self.assertEqual(self.storage.original_path(imported["id"], imported["images"][0]["id"]).read_bytes(), self.source)
        self.assertEqual(imported["style"], self.project["style"])
        self.assertEqual(imported["selectedRunId"], run["id"])
        self.assertEqual(imported["selectedMaskIds"], [run["masks"][0]["id"]])
        self.assertEqual(imported["ui"], {"imageId": imported["images"][0]["id"], "runId": run["id"], "selectedIds": [run["masks"][0]["id"]]})
        self.assertEqual(self.storage.load_project(imported["id"]), imported)
        self.assertEqual(self.storage.load_project(self.project["id"]), self.project)

    def test_exports_reject_misaligned_png_non_rgba_overlay_and_rle(self):
        with self.assertRaisesRegex(StorageError, "dimensions"):
            build_export(self.storage, self.project, "run-A", png(np.zeros((5, 3, 3), dtype=np.uint8)))
        with self.assertRaisesRegex(StorageError, "RGBA"):
            build_export(self.storage, self.project, "run-A", overlay_png=self.styled)
        bad = copy.deepcopy(self.project)
        bad["runs"][0]["masks"][0]["edited_rle"] = encode_mask(np.ones((2, 2), dtype=bool))
        with self.assertRaisesRegex(StorageError, "dimensions"):
            build_export(self.storage, bad, "run-A")

    def test_exports_reject_unsafe_and_case_colliding_ids(self):
        for bad_id in ("../escape", "nested/id", "x\\id", "/tmp/escape", "x" * 81):
            bad = copy.deepcopy(self.project)
            bad["runs"][0]["masks"][0]["id"] = bad_id
            with self.subTest(bad_id=bad_id), self.assertRaises(StorageError):
                build_export(self.storage, bad, "run-A")
        bad = copy.deepcopy(self.project)
        bad["runs"][0]["masks"][1]["id"] = "MASK-a"
        with self.assertRaisesRegex(StorageError, "unique"):
            build_export(self.storage, bad, "run-A")

    def test_asset_tampering_is_rejected_before_creating_a_project(self):
        _, files = self.bundle()
        original_count = len(self.storage.list_projects())
        files[self.image["original_path"]] = b"changed"
        with self.assertRaisesRegex(StorageError, "integrity"):
            import_bundle(self.storage, archive_bytes(files))
        self.assertEqual(len(self.storage.list_projects()), original_count)

    def test_original_hash_and_pixel_mapping_are_checked_beyond_manifest(self):
        _, original_files = self.bundle()
        for target in ("original", "canonical"):
            files = copy.deepcopy(original_files)
            project = json.loads(files["project.json"])
            if target == "original":
                files[self.image["original_path"]] = self.styled
            else:
                files[self.image["canonical_path"]] = self.styled
                project["images"][0]["canonical_sha256"] = hashlib.sha256(self.styled).hexdigest()
                files["project.json"] = json.dumps(project).encode()
            refresh_manifest(files)
            with self.subTest(target=target), self.assertRaises(StorageError):
                import_bundle(self.storage, archive_bytes(files))

    def test_binary_and_audit_disagreement_is_rejected(self):
        _, files = self.bundle()
        files["masks/mask-A.png"] = png(np.zeros((3, 5), dtype=np.uint8))
        refresh_manifest(files)
        with self.assertRaisesRegex(StorageError, "binary mask pixels"):
            import_bundle(self.storage, archive_bytes(files))

    def test_zip_traversal_backslash_and_symlinks_never_extract(self):
        for unsafe in ("../outside", "/absolute", "nested/../../outside", "C:/outside", "nested\\outside"):
            with self.subTest(unsafe=unsafe), self.assertRaisesRegex(StorageError, "unsafe"):
                import_bundle(self.storage, archive_bytes({unsafe: b"malicious"}))
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            entry = zipfile.ZipInfo("link")
            entry.create_system = 3
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(entry, "../../outside")
        with self.assertRaisesRegex(StorageError, "symlinks"):
            import_bundle(self.storage, output.getvalue())
        self.assertFalse((self.storage.root.parent / "outside").exists())

    def test_zip_duplicate_paths_and_decompression_limits(self):
        with self.assertRaisesRegex(StorageError, "duplicate"):
            import_bundle(self.storage, archive_bytes({"Asset": b"1", "asset": b"2"}))
        bomb = archive_bytes({"large": b"0" * 100_000})
        with mock.patch("backend.exports.MAX_BUNDLE_BYTES", 1000):
            with self.assertRaisesRegex(StorageError, "uncompressed"):
                import_bundle(self.storage, bomb)
        with mock.patch("backend.exports.MAX_BUNDLE_ENTRIES", 2):
            with self.assertRaisesRegex(StorageError, "entry"):
                import_bundle(self.storage, archive_bytes({"a": b"1", "b": b"1", "c": b"1"}))

    def test_manifest_path_mismatch_and_invalid_zip_are_rejected(self):
        _, files = self.bundle()
        manifest = json.loads(files["manifest.json"])
        manifest["files"]["unlisted"] = {"bytes": 0, "sha256": "0" * 64}
        files["manifest.json"] = json.dumps(manifest).encode()
        with self.assertRaisesRegex(StorageError, "manifest"):
            import_bundle(self.storage, archive_bytes(files))
        for invalid in (b"", b"not a zip", b"PK\x03\x04"):
            with self.subTest(invalid=invalid), self.assertRaises(StorageError):
                import_bundle(self.storage, invalid)


if __name__ == "__main__":
    unittest.main()
