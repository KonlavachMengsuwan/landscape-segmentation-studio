"""SSD-bound, versioned project storage and canonical image/mask encoding.

No function here opens a user-supplied path outside the dedicated workspace.
Imported source bytes are never rewritten. All pixel coordinates are zero-based
canonical pixel indices, with x increasing rightward and y downward.
"""
from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import os
import plistlib
import re
import subprocess
import threading
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.platform_support import WINDOWS, NOFOLLOW, linked, windows_disk_info

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = 24_000_000
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
SCHEMA_VERSION = 1
ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class StorageError(ValueError):
    """Recoverable input or persistence error, safe to display to the user."""


class VolumeUnavailable(StorageError):
    pass


class RevisionConflict(StorageError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(document: Any) -> bytes:
    try:
        return (json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StorageError("Project contains data that cannot be saved as JSON.") from exc


def _no_symlinks(path: Path) -> None:
    for component in (path, *path.parents):
        if linked(component):
            raise StorageError("Symlink paths are not allowed in project storage.")


def _disk_info(mount: Path) -> dict[str, Any]:
    try:
        if WINDOWS:
            return windows_disk_info(mount)
        result = subprocess.run(
            ["/usr/sbin/diskutil", "info", "-plist", str(mount)],
            capture_output=True, check=True, timeout=10,
        )
        return plistlib.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, plistlib.InvalidFileException) as exc:
        raise VolumeUnavailable("Cannot verify the project SSD. Reconnect it and retry.") from exc


class VolumeGuard:
    """Verify the saved volume identity rather than trusting its directory name.

    info_reader is injectable only for controlled tests. The default reads macOS
    disk metadata and never creates a missing mount point.
    """

    def __init__(self, mount: str | Path, expected_uuid: str,
                 info_reader: Callable[[Path], dict[str, Any]] | None = None):
        self.mount = Path(mount).absolute()
        self.expected_uuid = expected_uuid.upper()
        self.info_reader = info_reader or _disk_info

    def check(self) -> None:
        if not self.mount.is_dir():
            raise VolumeUnavailable("Project SSD is disconnected. Reconnect the same drive to continue; unsaved changes remain in this window.")
        _no_symlinks(self.mount)
        info = self.info_reader(self.mount)
        if (str(info.get("VolumeUUID", "")).upper() != self.expected_uuid
                or info.get("MountPoint") != str(self.mount)
                or info.get("Mounted") is False):
            raise VolumeUnavailable("The mounted drive does not match this workspace's saved SSD identity. Reconnect the original SSD.")


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise StorageError("Invalid project or image ID.")
    return value


def orientation_transform(orientation: int, width: int, height: int) -> list[list[int]]:
    """Original pixel index to canonical pixel index, including mirrored EXIF."""
    return {
        1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        2: [[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]],
        3: [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
        4: [[1, 0, 0], [0, -1, height - 1], [0, 0, 1]],
        5: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        6: [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
        7: [[0, -1, height - 1], [-1, 0, width - 1], [0, 0, 1]],
        8: [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
    }[orientation]


def decode_image(data: bytes) -> tuple[bytes, dict[str, Any]]:
    """Return an orientation-corrected full-resolution RGB PNG plus identity.

    PNG transparency is composited on white, while the original bytes remain
    available unchanged. Scientific TIFFs and large images require deliberate
    preprocessing outside this release rather than silent data reduction.
    """
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise StorageError("Choose a nonempty image smaller than 100 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "TIFF"}:
                    raise StorageError("Supported still-image formats are JPEG, PNG, and ordinary 8-bit RGB or grayscale TIFF.")
                if getattr(source, "n_frames", 1) != 1:
                    raise StorageError("Animated or multi-page images are unsupported. Choose one still image.")
                # Pillow may expose an already oriented TIFF size before
                # load(). TIFF tags retain the actual stored pixel grid.
                width, height = ((int(source.tag_v2[256]), int(source.tag_v2[257]))
                                 if source.format == "TIFF" else source.size)
                if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                    raise StorageError("This image exceeds the 24 megapixel limit. Prepare a separate smaller display image; keep the original intact.")
                original_format, original_mode = source.format, source.mode
                if source.format == "TIFF":
                    bits = source.tag_v2.get(258, (8,))
                    bits = (bits,) if isinstance(bits, int) else bits
                    description = str(source.tag_v2.get(270, ""))
                    if (source.mode not in {"L", "RGB"} or any(bit != 8 for bit in bits)
                            or source.tag_v2.get(339, (1,)) not in ((1,), (1, 1, 1), 1)
                            or "<OME" in description or "<ome:" in description):
                        raise StorageError("Scientific, radiometric, high-bit-depth, or multiband TIFF data is unsupported. Import an 8-bit RGB or grayscale display image instead.")
                if source.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
                    raise StorageError("High-bit-depth or scientific image pixels are unsupported. Use an 8-bit display image.")
                orientation = int(source.getexif().get(274, 1))
                if orientation not in range(1, 9):
                    raise StorageError("The image contains an invalid EXIF orientation.")
                source.load()
                canonical = ImageOps.exif_transpose(source)
                has_alpha = canonical.mode in {"RGBA", "LA"} or "transparency" in canonical.info
                if has_alpha:
                    rgba = canonical.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    canonical = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    canonical = canonical.convert("RGB")
                # Remove EXIF/ICC/text from the display derivative. The immutable
                # source retains its complete metadata; no private EXIF is logged.
                canonical.info.clear()
                output = io.BytesIO()
                canonical.save(output, format="PNG")
                png = output.getvalue()
                forward = orientation_transform(orientation, width, height)
                inverse = np.rint(np.linalg.inv(np.asarray(forward))).astype(int).tolist()
                metadata = {
                    "width": canonical.width, "height": canonical.height,
                    "original_width": width, "original_height": height,
                    "format": original_format, "original_mode": original_mode,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "canonical_sha256": hashlib.sha256(png).hexdigest(),
                    "transforms": {
                        "exif_orientation": orientation,
                        "original_to_canonical": forward,
                        "canonical_to_original": inverse,
                        "coordinate_convention": "zero-based pixel indices; x right, y down",
                        "alpha_background": "#ffffff" if has_alpha else None,
                        "canonical_color_mode": "RGB",
                        "color_conversion": "Pillow RGB conversion, no ICC profile transformation",
                    },
                }
                return png, metadata
    except StorageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise StorageError("The image could not be decoded safely. Choose a valid supported still image within the size limit.") from exc


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    """Lossless uncompressed RLE; C/row-major order, initial background run."""
    array = np.asarray(mask)
    if array.ndim != 2 or not array.size or array.size > MAX_IMAGE_PIXELS:
        raise StorageError("Mask must have a nonempty two-dimensional canonical pixel grid within the size limit.")
    flat = array.astype(bool).ravel(order="C")
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"size": list(array.shape), "counts": counts}


def decode_mask(rle: dict[str, Any]) -> np.ndarray:
    if not isinstance(rle, dict):
        raise StorageError("Invalid mask RLE.")
    size, counts = rle.get("size"), rle.get("counts")
    if (not isinstance(size, list) or len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size)
            or size[0] * size[1] > MAX_IMAGE_PIXELS):
        raise StorageError("Invalid mask dimensions.")
    total = size[0] * size[1]
    if (not isinstance(counts, list) or not counts or len(counts) > total + 1
            or any(type(n) is not int or n < 0 for n in counts)
            or sum(counts) != total):
        raise StorageError("Mask RLE counts do not match its canonical dimensions.")
    return np.repeat(np.arange(len(counts), dtype=np.int64) % 2, counts).astype(bool).reshape(size)


def binary_mask_png(rle: dict[str, Any]) -> bytes:
    output = io.BytesIO()
    Image.fromarray(decode_mask(rle).astype(np.uint8) * 255).save(output, format="PNG")
    return output.getvalue()


class Storage:
    def __init__(self, root: str | Path, guard: VolumeGuard | None = None):
        self.root = Path(root).absolute()
        _no_symlinks(self.root)
        if not self.root.is_dir():
            raise VolumeUnavailable("The dedicated workspace is missing. Reconnect the project SSD.")
        if guard is None:
            config_path = self.root / "setup-notes" / "volume.json"
            _no_symlinks(config_path)
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                guard = VolumeGuard(config["mount"], config["uuid"])
            except (OSError, ValueError, KeyError) as exc:
                raise VolumeUnavailable("Saved SSD identity is missing. Run the documented project setup before starting.") from exc
        self.guard = guard
        try:
            self.root.relative_to(self.guard.mount)
        except ValueError as exc:
            raise StorageError("Workspace must be within the verified project SSD.") from exc
        self.guard.check()
        self._lock = threading.RLock()

    def check(self) -> None:
        self.guard.check()
        _no_symlinks(self.root)
        if not self.root.is_dir():
            raise VolumeUnavailable("The dedicated project folder is missing. No replacement folder was created.")

    def safe_path(self, *parts: str | Path) -> Path:
        for part in parts:
            value = Path(part)
            if value.is_absolute() or ".." in value.parts or "\\" in str(part) or "\x00" in str(part):
                raise StorageError("Paths must remain inside the dedicated workspace.")
        candidate = self.root.joinpath(*parts)
        _no_symlinks(candidate)
        try:
            candidate.resolve().relative_to(self.root.resolve())
        except ValueError as exc:
            raise StorageError("Path escapes the dedicated workspace.") from exc
        return candidate

    def _mkdir(self, relative: str | Path) -> Path:
        self.check()
        path = self.safe_path(relative)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_atomic(self, relative: str | Path, data: bytes, *, immutable: bool = False) -> Path:
        self.check()
        target = self.safe_path(relative)
        self._mkdir(target.parent.relative_to(self.root))
        if immutable and target.exists():
            raise StorageError("An immutable project asset already exists.")
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            self.check()
            self.safe_path(temporary.relative_to(self.root))
            with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600), "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self.check()
            self.safe_path(target.relative_to(self.root))
            if immutable and target.exists():
                raise StorageError("An immutable project asset already exists.")
            os.replace(temporary, target)
            # ExFAT/macOS may reject directory fsync. File fsync and atomic
            # replace are still used; no physical-unplug durability is claimed.
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                try:
                    os.fsync(directory_fd)
                except OSError as exc:
                    if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
                        raise
            finally:
                os.close(directory_fd)
            return target
        finally:
            # Never create directories or write after a disconnect just to
            # clean up. An uncommitted temp file is safe to leave in place.
            try:
                self.check()
                self.safe_path(temporary.relative_to(self.root))
                temporary.unlink(missing_ok=True)
            except (OSError, StorageError):
                pass

    def create_project(self, name: str = "Untitled landscape") -> dict[str, Any]:
        with self._lock:
            project_id = uuid.uuid4().hex
            document = {"schema_version": SCHEMA_VERSION, "id": project_id,
                        "name": str(name).strip()[:160] or "Untitled landscape",
                        "created_at": _now(), "revision": 0,
                        "images": [], "runs": [], "style": {}}
            return self._save(project_id, document, previous=None)

    def _read_json(self, path: Path) -> Any:
        self.safe_path(path.relative_to(self.root))
        try:
            return json.loads(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise StorageError("Saved project data is unavailable or damaged. Earlier history files have been preserved.") from exc

    def load_project(self, project_id: str) -> dict[str, Any]:
        self.check()
        project_id = _identifier(project_id)
        base = Path("projects") / project_id
        pointer = self._read_json(self.safe_path(base / "current.json"))
        if (not isinstance(pointer, dict) or not isinstance(pointer.get("version"), str)
                or not re.fullmatch(r"[0-9]{8,}-[a-f0-9]{32}\.json", pointer["version"])):
            raise StorageError("Project save pointer is invalid. Earlier history files have been preserved.")
        path = self.safe_path(base / "history" / pointer["version"])
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise StorageError("The current project version is missing. Earlier history files have been preserved.") from exc
        if hashlib.sha256(data).hexdigest() != pointer.get("sha256"):
            raise StorageError("Project integrity check failed. Earlier history files have been preserved.")
        try:
            document = json.loads(data)
        except ValueError as exc:
            raise StorageError("Saved project JSON is invalid.") from exc
        if (not isinstance(document, dict) or document.get("id") != project_id
                or document.get("schema_version") != SCHEMA_VERSION):
            raise StorageError("Unsupported or mismatched project version.")
        return document

    def list_projects(self) -> list[dict[str, Any]]:
        self.check()
        projects_path = self.safe_path("projects")
        if not projects_path.exists():
            return []
        result = []
        for path in projects_path.iterdir():
            if ID_PATTERN.fullmatch(path.name) and path.is_dir():
                try:
                    document = self.load_project(path.name)
                    result.append({key: document.get(key) for key in ("id", "name", "revision", "created_at", "updated_at")}
                                     | {"image_count": len(document.get("images", []))})
                except VolumeUnavailable:
                    raise
                except StorageError:
                    result.append({"id": path.name, "name": "Project needs recovery", "error": "Saved data unavailable or damaged"})
        return sorted(result, key=lambda item: item.get("updated_at") or "", reverse=True)

    def save_project(self, project_id: str, document: dict[str, Any], expected_revision: int | None = None) -> dict[str, Any]:
        with self._lock:
            previous = self.load_project(project_id)
            expected_revision = document.get("revision") if expected_revision is None else expected_revision
            if expected_revision is not None and expected_revision != previous["revision"]:
                raise RevisionConflict("A newer project version is already saved. Reload it before saving these changes.")
            return self._save(project_id, document, previous=previous)

    def _save(self, project_id: str, document: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
        _identifier(project_id)
        if not isinstance(document, dict):
            raise StorageError("Project must be a JSON object.")
        saved = copy.deepcopy(document)
        if previous:
            images = saved.get("images", [])
            if not isinstance(images, list) or any(not isinstance(image, dict) for image in images):
                raise StorageError("Project images must be a list of image records.")
            for image in images:
                image_id = _identifier(image.get("id"))
                identity = self._read_json(self.safe_path("projects", project_id, "images", image_id, "identity.json"))
                if any(image.get(key) != value for key, value in identity.items()):
                    raise StorageError("Imported image identity is immutable. Reimport a new source instead of changing its dimensions, hash, or paths.")
            if len({image["id"] for image in images}) != len(images):
                raise StorageError("Project image IDs must be unique.")
            # Removing an image from the browser does not delete its source.
            saved["created_at"] = previous["created_at"]
        saved.update({"schema_version": SCHEMA_VERSION, "id": project_id,
                      "revision": previous["revision"] + 1 if previous else 1,
                      "updated_at": _now()})
        data = _json_bytes(saved)
        version = f"{saved['revision']:08d}-{uuid.uuid4().hex}.json"
        base = Path("projects") / project_id
        self._write_atomic(base / "history" / version, data, immutable=True)
        pointer = {"version": version, "sha256": hashlib.sha256(data).hexdigest(), "revision": saved["revision"]}
        self._write_atomic(base / "current.json", _json_bytes(pointer))
        return saved

    def import_image(self, project_id: str, filename: str, data: bytes) -> dict[str, Any]:
        with self._lock:
            project = self.load_project(project_id)
            png, identity = decode_image(data)
            image_id = uuid.uuid4().hex
            extension = {"JPEG": "jpg", "PNG": "png", "TIFF": "tif"}[identity["format"]]
            name = Path(str(filename).replace("\\", "/")).name
            name = "".join(character for character in name if character.isprintable())[:240] or f"Image.{extension}"
            relative = Path("images") / image_id
            identity.update({"id": image_id, "name": name, "imported_at": _now(),
                             "original_path": (relative / f"original.{extension}").as_posix(),
                             "canonical_path": (relative / "canonical.png").as_posix()})
            base = Path("projects") / project_id
            self._write_atomic(base / identity["original_path"], data, immutable=True)
            self._write_atomic(base / identity["canonical_path"], png, immutable=True)
            self._write_atomic(base / relative / "identity.json", _json_bytes(identity), immutable=True)
            project.setdefault("images", []).append(identity)
            self._save(project_id, project, previous=project)
            return identity

    def relink_image(self, project_id: str, image_id: str, filename: str, data: bytes) -> dict[str, Any]:
        """Restore missing image assets from explicitly selected identical bytes.

        Return the unchanged immutable image metadata. This does not rename the
        image, increment the project revision, alter runs, or overwrite any asset.
        Existing corrupt assets require recovery from a trusted portable bundle.
        A partial restore after a write failure can be retried safely.
        """
        with self._lock:
            self.check()
            project_id, image_id = _identifier(project_id), _identifier(image_id)
            project = self.load_project(project_id)
            image = next((item for item in project.get("images", [])
                          if isinstance(item, dict) and item.get("id") == image_id), None)
            if image is None:
                raise StorageError("This image is not part of the current project.")
            base = Path("projects") / project_id
            identity = self._read_json(self.safe_path(base / "images" / image_id / "identity.json"))
            if (not isinstance(identity, dict) or identity.get("id") != image_id
                    or any(image.get(key) != value for key, value in identity.items())):
                raise StorageError("Saved image identity is inconsistent. Restore a trusted portable export bundle.")
            # filename is deliberately not used to choose paths or change the
            # saved name. Renamed copies are accepted only by their exact bytes.
            if not isinstance(data, bytes) or not data or len(data) > MAX_UPLOAD_BYTES:
                raise StorageError("Choose a nonempty original image smaller than 100 MiB.")
            if hashlib.sha256(data).hexdigest() != identity.get("sha256"):
                raise StorageError("The selected file does not match the saved original SHA256. A matching filename is not enough; choose the exact original bytes.")
            canonical, decoded = decode_image(data)
            if any(decoded.get(key) != identity.get(key) for key in decoded):
                raise StorageError("The selected original decodes to a different canonical hash or pixel grid. Restore a trusted portable export bundle using its recorded image assets.")
            extension = {"JPEG": "jpg", "PNG": "png", "TIFF": "tif"}[decoded["format"]]
            expected_paths = {
                "original_path": f"images/{image_id}/original.{extension}",
                "canonical_path": f"images/{image_id}/canonical.png",
            }
            if any(identity.get(key) != value for key, value in expected_paths.items()):
                raise StorageError("Saved image asset paths are invalid. Restore a trusted portable export bundle.")
            assets = [(base / identity["original_path"], data, identity["sha256"]),
                      (base / identity["canonical_path"], canonical, identity["canonical_sha256"])]
            missing = []
            # Preflight every existing asset before recreating anything. Thus a
            # corrupt sibling cannot cause even the missing asset to be written.
            for relative, payload, expected_hash in assets:
                self.check()
                path = self.safe_path(relative)
                if path.exists():
                    try:
                        valid = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
                    except OSError as exc:
                        raise StorageError("An existing image asset cannot be read. It was not overwritten; restore a trusted portable export bundle.") from exc
                    if not valid:
                        raise StorageError("An existing original or canonical image is corrupt. Relink will not overwrite it; restore a trusted portable export bundle into a separate recovered project.")
                else:
                    missing.append((relative, payload))
            for relative, payload in missing:
                self._write_atomic(relative, payload, immutable=True)
            self.check()
            return copy.deepcopy(identity)

    def image_path(self, project_id: str, image_id: str) -> Path:
        return self._asset_path(project_id, image_id, "canonical_path", "canonical_sha256")

    def original_path(self, project_id: str, image_id: str) -> Path:
        return self._asset_path(project_id, image_id, "original_path", "sha256")

    def _asset_path(self, project_id: str, image_id: str, field: str, hash_field: str) -> Path:
        self.check()
        _identifier(project_id)
        _identifier(image_id)
        identity = self._read_json(self.safe_path("projects", project_id, "images", image_id, "identity.json"))
        path = self.safe_path("projects", project_id, identity[field])
        try:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise StorageError("An imported image is missing. Use Relink image and select the exact original bytes to restore it without changing its runs.") from exc
        if actual_hash != identity[hash_field]:
            raise StorageError("Image integrity check failed. This asset has changed; relink will not overwrite it. Restore a trusted portable export bundle.")
        return path

    def export_path(self, project_id: str, filename: str) -> Path:
        self.check()
        _identifier(project_id)
        if (not isinstance(filename, str) or Path(filename).name != filename
                or filename in {"", ".", ".."} or "\\" in filename):
            raise StorageError("Export filename must be a simple name within this project's exports folder.")
        return self.safe_path("exports", project_id, filename)

    def write_export(self, project_id: str, filename: str, data: bytes) -> Path:
        path = self.export_path(project_id, filename)
        return self._write_atomic(path.relative_to(self.root), data)
