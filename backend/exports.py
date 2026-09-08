"""Portable, auditable image bundles with bounded, extraction-free import."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import re
import stat
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .storage import (
    MAX_IMAGE_PIXELS, Storage, StorageError, binary_mask_png, decode_image,
    decode_mask,
)

BUNDLE_SCHEMA_VERSION = 1
MAX_BUNDLE_BYTES = 200 * 1024 * 1024
MAX_BUNDLE_ENTRIES = 1000
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def _safe_id(value: Any, kind: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise StorageError(f"{kind} IDs must be safe, unique filename identifiers of at most 80 characters.")
    return value


def _archive_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise StorageError("Bundle contains an unsafe asset path.")
    path = PurePosixPath(value)
    if (path.is_absolute() or any(part in {".", "..", ""} for part in value.split("/"))
            or path.as_posix() != value):
        raise StorageError("Bundle contains an unsafe asset path.")
    return value


def _json(data: Any) -> bytes:
    try:
        return (json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise StorageError("Export contains data that is not valid portable JSON.") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON value: {value}")


def _read_json(data: bytes, name: str) -> dict[str, Any]:
    try:
        result = json.loads(data, parse_constant=_reject_json_constant)
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        return result
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise StorageError(f"Bundle {name} is not valid JSON.") from exc


def _dimensions(image: dict[str, Any]) -> tuple[int, int]:
    width, height = image.get("width"), image.get("height")
    if (type(width) is not int or type(height) is not int
            or width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS):
        raise StorageError("Image has invalid canonical dimensions.")
    return width, height


def _source_paths(image: dict[str, Any]) -> tuple[str, str]:
    image_id = _safe_id(image.get("id"), "Image")
    extension = {"JPEG": "jpg", "PNG": "png", "TIFF": "tif"}.get(image.get("format"))
    original = _archive_path(image.get("original_path"))
    canonical = _archive_path(image.get("canonical_path"))
    if (not extension or original != f"images/{image_id}/original.{extension}"
            or canonical != f"images/{image_id}/canonical.png"):
        raise StorageError("Bundle source asset paths do not match their image identity.")
    return original, canonical


def _validated_png(data: bytes, width: int, height: int, *, overlay: bool = False) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (width, height) or getattr(image, "n_frames", 1) != 1:
                raise StorageError("Export PNG dimensions must exactly match the canonical image grid.")
            if overlay and image.mode != "RGBA":
                raise StorageError("The overlay must be an RGBA PNG with an explicit alpha channel.")
            image.load()
            return np.array(image)
    except StorageError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise StorageError("Export preview is not a valid PNG.") from exc


def _validate_run(run: dict[str, Any], image: dict[str, Any]) -> None:
    if not isinstance(run, dict):
        raise StorageError("Each run must be a JSON object.")
    _safe_id(run.get("id"), "Run")
    if run.get("image_id") != image.get("id"):
        raise StorageError("Run refers to a different source image.")
    width, height = _dimensions(image)
    masks = run.get("masks")
    if not isinstance(masks, list):
        raise StorageError("Run masks must be a list.")
    identifiers = set()
    for mask in masks:
        if not isinstance(mask, dict):
            raise StorageError("Each mask must be a JSON object.")
        mask_id = _safe_id(mask.get("id"), "Mask")
        if mask_id.casefold() in identifiers:
            raise StorageError("Mask IDs must be unique, including on case-insensitive filesystems.")
        identifiers.add(mask_id.casefold())
        if mask.get("run_id", run["id"]) != run["id"]:
            raise StorageError("Mask run identity does not match its parent run.")
        if mask.get("status") not in ("proposed", "accepted", "rejected"):
            raise StorageError("Mask review status is invalid.")
        if not isinstance(mask.get("label", ""), str) or len(mask.get("label", "")) > 160:
            raise StorageError("Mask labels must be text of at most 160 characters.")
        for key in ("score", "stability"):
            score = mask.get(key)
            if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
                raise StorageError("Mask model scores must be finite numbers or unavailable.")
        for name in ("original_rle", "edited_rle"):
            rle = mask.get(name)
            if rle is None and name == "edited_rle":
                continue
            decoded = decode_mask(rle)
            if decoded.shape != (height, width):
                raise StorageError("Every original and edited mask must match the canonical image dimensions.")


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        # CSV quoting does not prevent spreadsheet formula execution. Prefix
        # only the CSV display cell; exact labels remain in the audit JSON.
        return "'" + value
    return value


def _metadata_csv(run: dict[str, Any], masks: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["mask_id", "run_id", "label", "review_status", "proposed_label", "reviewed_label",
                     "pixel_count", "original_pixel_count", "model_score", "stability_score", "manually_edited", "visible", "segment_index"])
    indices = {mask["id"]: index + 1 for index, mask in enumerate(run["masks"])}
    for mask in masks:
        label, status = mask.get("label", ""), mask.get("status", "proposed")
        original = mask["original_rle"]
        edited = mask.get("edited_rle")
        writer.writerow([_csv_cell(value) for value in (
            mask["id"], run["id"], label, status,
            mask.get("proposed_label", label if status != "accepted" else ""),
            mask.get("reviewed_label", label if status == "accepted" else ""),
            int(decode_mask(edited if edited is not None else original).sum()),
            int(decode_mask(original).sum()), mask.get("score"), mask.get("stability"),
            edited is not None, bool(mask.get("visible", True)), indices[mask["id"]],
        )])
    return output.getvalue().encode("utf-8-sig")


def build_export(storage: Storage, project: dict[str, Any], run_id: str,
                 styled_png: bytes | None = None, overlay_png: bytes | None = None) -> Path:
    """Save a ZIP containing the selected image, its runs, and visible masks.

    PNG renderings are supplied by the frontend's full-resolution export
    renderer. Their pixel grid and RGBA alpha channel are validated here.
    """
    storage.check()
    _safe_id(run_id, "Run")
    if not isinstance(project, dict):
        raise StorageError("Export project must be a JSON object.")
    runs = project.get("runs", [])
    if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        raise StorageError("Project runs must be a list of run records.")
    if not isinstance(project.get("images"), list) or any(not isinstance(image, dict) for image in project["images"]):
        raise StorageError("Project images must be a list of image records.")
    target_runs = [run for run in runs if run.get("id") == run_id]
    if len(target_runs) != 1:
        raise StorageError("Choose exactly one existing run to export.")
    target = target_runs[0]
    images = [image for image in project.get("images", []) if image.get("id") == target.get("image_id")]
    if len(images) != 1:
        raise StorageError("The selected run's source image is missing or ambiguous.")
    image = images[0]
    width, height = _dimensions(image)
    selected_runs = [run for run in runs if run.get("image_id") == image["id"]]
    run_ids = set()
    for run in selected_runs:
        _validate_run(run, image)
        if run["id"].casefold() in run_ids:
            raise StorageError("Run IDs must be unique within an exported project.")
        run_ids.add(run["id"].casefold())
    original_path, canonical_path = _source_paths(image)
    files = {
        original_path: storage.original_path(project["id"], image["id"]).read_bytes(),
        canonical_path: storage.image_path(project["id"], image["id"]).read_bytes(),
    }
    if (hashlib.sha256(files[original_path]).hexdigest() != image.get("sha256")
            or hashlib.sha256(files[canonical_path]).hexdigest() != image.get("canonical_sha256")):
        raise StorageError("The export document does not match its immutable source identity.")
    _validated_png(files[canonical_path], width, height)
    portable = copy.deepcopy(project)
    portable["images"] = [copy.deepcopy(image)]
    portable["runs"] = copy.deepcopy(selected_runs)
    portable["exported_run_id"] = run_id
    for key in ("selectedImageId", "selected_image_id", "activeImageId"):
        if key in portable:
            portable[key] = image["id"]
    for key in ("selectedRunId", "selected_run_id", "activeRunId"):
        if key in portable:
            portable[key] = run_id
    if isinstance(portable.get("ui"), dict):
        portable["ui"]["imageId"] = image["id"]
        portable["ui"]["runId"] = run_id
    files["project.json"] = _json(portable)
    files["audit/run.json"] = _json(target)
    visible = [mask for mask in target["masks"] if mask.get("visible", True) and not mask.get("deleted", False)]
    if len(files) + len(visible) * 2 + 2 + int(styled_png is not None) + int(overlay_png is not None) > MAX_BUNDLE_ENTRIES:
        raise StorageError("This export exceeds the 1000-entry bundle limit. Reduce the number of visible masks.")
    for mask in visible:
        effective = mask.get("edited_rle")
        effective = mask["original_rle"] if effective is None else effective
        files[f"masks/{mask['id']}.png"] = binary_mask_png(effective)
        files[f"masks-original/{mask['id']}.png"] = binary_mask_png(mask["original_rle"])
    files["metadata.csv"] = _metadata_csv(target, visible)
    if styled_png is not None:
        _validated_png(styled_png, width, height)
        files["styled.png"] = styled_png
    if overlay_png is not None:
        _validated_png(overlay_png, width, height, overlay=True)
        files["overlay.png"] = overlay_png
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project["id"], "image_id": image["id"], "run_id": run_id,
        "canonical_dimensions": {"width": width, "height": height},
        "original_sha256": image["sha256"],
        "selection": "Visible, non-deleted masks in the selected run, including visible rejected proposals; all masks remain in audit JSON.",
        "mask_ids": [mask["id"] for mask in visible],
        "binary_values": {"outside": 0, "inside": 255},
        "mask_grid": "Canonical, full-resolution, zero-based x right and y down.",
        "overlap_policy": "Independent instance masks preserve every overlap. The last visible mask is topmost for display only.",
        "scores": "Model scores are model-specific ranking estimates, not scientific accuracy.",
        "files": {path: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for path, data in files.items()},
    }
    files["manifest.json"] = _json(manifest)
    if len(files) > MAX_BUNDLE_ENTRIES or sum(map(len, files.values())) > MAX_BUNDLE_BYTES:
        raise StorageError("This export exceeds the 200 MiB or 1000-entry bundle limit. Reduce the number of visible masks or export a smaller project.")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, data in files.items():
            _archive_path(path)
            archive.writestr(path, data)
    bundle = output.getvalue()
    if len(bundle) > MAX_BUNDLE_BYTES:
        raise StorageError("Compressed export exceeds the 200 MiB bundle limit.")
    filename = f"landscape-{run_id}-{uuid.uuid4().hex[:8]}.zip"
    return storage.write_export(project["id"], filename, bundle)


def _bundle_files(data: bytes) -> dict[str, bytes]:
    if not isinstance(data, bytes) or not data or len(data) > MAX_BUNDLE_BYTES:
        raise StorageError("Choose a nonempty Landscape bundle smaller than 200 MiB.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_BUNDLE_ENTRIES:
                raise StorageError("Bundle exceeds the 1000-entry limit.")
            if sum(entry.file_size for entry in entries) > MAX_BUNDLE_BYTES:
                raise StorageError("Bundle exceeds the 200 MiB uncompressed limit.")
            names = set()
            for entry in entries:
                name = _archive_path(entry.filename)
                if (name.casefold() in names or stat.S_ISLNK(entry.external_attr >> 16)
                        or entry.flag_bits & 1 or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                    raise StorageError("Bundle has duplicate paths, symlinks, encryption, or unsupported compression.")
                names.add(name.casefold())
            files, actual_size = {}, 0
            for entry in entries:
                chunks, entry_size = [], 0
                with archive.open(entry) as handle:
                    while chunk := handle.read(1024 * 1024):
                        actual_size += len(chunk)
                        entry_size += len(chunk)
                        if actual_size > MAX_BUNDLE_BYTES or entry_size > entry.file_size:
                            raise StorageError("Bundle expanded beyond its declared or permitted size.")
                        chunks.append(chunk)
                if entry_size != entry.file_size:
                    raise StorageError("Bundle asset size is inconsistent.")
                files[entry.filename] = b"".join(chunks)
            return files
    except StorageError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise StorageError("This file is not a valid supported Landscape export bundle.") from exc


def import_bundle(storage: Storage, bundle_bytes: bytes, name: str | None = None) -> dict[str, Any]:
    """Validate a whole bundle in memory, then import into a NEW project.

    No ZIP member is extracted to disk. Source bytes pass through the normal
    immutable image importer, and all new writes use the existing SSD guard.
    """
    storage.check()
    files = _bundle_files(bundle_bytes)
    if "manifest.json" not in files or "project.json" not in files:
        raise StorageError("Bundle is missing its manifest or project JSON.")
    manifest = _read_json(files["manifest.json"], "manifest")
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise StorageError("Unsupported Landscape bundle version.")
    assets = manifest.get("files")
    if not isinstance(assets, dict) or set(assets) != set(files) - {"manifest.json"}:
        raise StorageError("Bundle contents do not match the manifest.")
    for path, expected in assets.items():
        _archive_path(path)
        if (not isinstance(expected, dict) or expected.get("bytes") != len(files[path])
                or expected.get("sha256") != hashlib.sha256(files[path]).hexdigest()):
            raise StorageError("Bundle asset integrity check failed.")
    project = _read_json(files["project.json"], "project")
    if project.get("schema_version") != 1 or not isinstance(project.get("images"), list) or len(project["images"]) != 1:
        raise StorageError("Bundle must contain one supported canonical source image.")
    image = project["images"][0]
    if not isinstance(image, dict):
        raise StorageError("Bundle image identity is invalid.")
    old_image_id = _safe_id(image.get("id"), "Image")
    width, height = _dimensions(image)
    original_path, canonical_path = _source_paths(image)
    if original_path == canonical_path or original_path not in files or canonical_path not in files:
        raise StorageError("Bundle is missing distinct original and canonical image assets.")
    original, canonical = files[original_path], files[canonical_path]
    if (hashlib.sha256(original).hexdigest() != image.get("sha256")
            or image.get("sha256") != manifest.get("original_sha256")
            or hashlib.sha256(canonical).hexdigest() != image.get("canonical_sha256")):
        raise StorageError("Bundle source identity hash does not match its image assets.")
    supplied_pixels = _validated_png(canonical, width, height)
    recomputed_png, decoded_identity = decode_image(original)
    recomputed_pixels = _validated_png(recomputed_png, width, height)
    if not np.array_equal(supplied_pixels, recomputed_pixels):
        raise StorageError("Bundle canonical pixels do not match the orientation-corrected original source.")
    if any(image.get(key) != decoded_identity[key] for key in (
            "transforms", "original_width", "original_height", "format", "original_mode")):
        raise StorageError("Bundle source-to-canonical transformation does not match its original image.")
    runs = project.get("runs")
    if not isinstance(runs, list):
        raise StorageError("Bundle runs must be a list.")
    run_id_map, mask_id_maps = {}, {}
    for run in runs:
        _validate_run(run, image)
        if run["id"].casefold() in {key.casefold() for key in run_id_map}:
            raise StorageError("Bundle run IDs must be unique.")
        run_id_map[run["id"]] = uuid.uuid4().hex
        mask_id_maps[run["id"]] = {mask["id"]: uuid.uuid4().hex for mask in run["masks"]}
    target_id = manifest.get("run_id")
    if target_id not in run_id_map or manifest.get("image_id") != old_image_id:
        raise StorageError("Bundle selected run or image does not match its project.")
    if manifest.get("canonical_dimensions") != {"width": width, "height": height}:
        raise StorageError("Bundle manifest has inconsistent canonical dimensions.")
    target_run = next(run for run in runs if run["id"] == target_id)
    visible = [mask for mask in target_run["masks"] if mask.get("visible", True) and not mask.get("deleted", False)]
    if manifest.get("mask_ids") != [mask["id"] for mask in visible]:
        raise StorageError("Bundle selected mask list does not match its project.")
    if "audit/run.json" not in files or _read_json(files["audit/run.json"], "run audit") != target_run:
        raise StorageError("Bundle run audit does not match its project.")
    for mask in visible:
        effective = mask["original_rle"] if mask.get("edited_rle") is None else mask["edited_rle"]
        for folder, rle in (("masks", effective), ("masks-original", mask["original_rle"])):
            path = f"{folder}/{mask['id']}.png"
            if path not in files:
                raise StorageError("Bundle is missing a selected binary mask.")
            pixels = _validated_png(files[path], width, height)
            if not np.array_equal(pixels, decode_mask(rle).astype(np.uint8) * 255):
                raise StorageError("Bundle binary mask pixels do not match its lossless audit data.")
    if "styled.png" in files:
        _validated_png(files["styled.png"], width, height)
    if "overlay.png" in files:
        _validated_png(files["overlay.png"], width, height, overlay=True)
    # All untrusted structures and pixels are checked before the first write.
    display_name = str(name if name is not None else project.get("name", "Imported landscape"))[:160]
    created = storage.create_project(f"Import in progress: {display_name}")
    new_image = storage.import_image(created["id"], image.get("name", "Imported image"), original)
    current = storage.load_project(created["id"])
    imported = copy.deepcopy(project)
    imported.update({"id": current["id"], "revision": current["revision"],
                     "created_at": current["created_at"], "name": display_name,
                     "images": [new_image], "exported_run_id": run_id_map[target_id],
                     "provenance": {"source_project_id": project.get("id"), "source_image_id": old_image_id,
                                    "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
                                    "imported_at": datetime.now(timezone.utc).isoformat(),
                                    "prior": project.get("provenance")}})
    for run in imported["runs"]:
        old_run_id = run["id"]
        run["id"], run["image_id"] = run_id_map[old_run_id], new_image["id"]
        run["provenance"] = {"source_run_id": old_run_id, "source_project_id": project.get("id"), "prior": run.get("provenance")}
        for mask in run["masks"]:
            old_mask_id = mask["id"]
            mask["id"], mask["run_id"] = mask_id_maps[old_run_id][old_mask_id], run["id"]
            mask["provenance"] = {"source_mask_id": old_mask_id, "source_run_id": old_run_id, "prior": mask.get("provenance")}
    for key in ("selectedImageId", "selected_image_id", "activeImageId"):
        if key in imported:
            imported[key] = new_image["id"]
    for key in ("selectedRunId", "selected_run_id", "activeRunId"):
        if key in imported:
            imported[key] = run_id_map[target_id]
    for key in ("selectedMaskIds", "selected_mask_ids"):
        if key in imported:
            selected = imported[key]
            imported[key] = [mask_id_maps[target_id][value] for value in selected if isinstance(value, str) and value in mask_id_maps[target_id]] if isinstance(selected, list) else []
    if isinstance(imported.get("ui"), dict):
        imported["ui"]["imageId"] = new_image["id"]
        imported["ui"]["runId"] = run_id_map[target_id]
        selected = imported["ui"].get("selectedIds", [])
        imported["ui"]["selectedIds"] = [mask_id_maps[target_id][value] for value in selected if isinstance(value, str) and value in mask_id_maps[target_id]] if isinstance(selected, list) else []
    return storage.save_project(current["id"], imported, expected_revision=current["revision"])
