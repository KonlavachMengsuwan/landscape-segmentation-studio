"""Standard-library SSD checks shared by setup and process management."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import subprocess
import uuid

try:
    from .platform_support import WINDOWS, NOFOLLOW, linked, windows_disk_info
except ImportError:
    from platform_support import WINDOWS, NOFOLLOW, linked, windows_disk_info

APP = Path(__file__).absolute().parents[1]
ROOT = APP.parent


def safe(path: Path) -> Path:
    path = path.absolute()
    if not path.is_relative_to(ROOT):
        raise RuntimeError("Requested path is outside the dedicated workspace.")
    if any(linked(item) for item in (path, *path.parents)):
        raise RuntimeError("Symlink paths are not allowed for setup or application state.")
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Requested path escaped the dedicated workspace.")
    return path


def disk_info(mount: Path) -> dict:
    if WINDOWS:
        return windows_disk_info(mount)
    result = subprocess.run(["/usr/sbin/diskutil", "info", "-plist", str(mount)],
                            check=True, capture_output=True, timeout=10)
    return plistlib.loads(result.stdout)


def check_volume(*, initialize: bool = False) -> dict:
    if not ROOT.is_dir():
        raise RuntimeError("The SSD workspace is unavailable. No replacement mount directory was created.")
    safe(ROOT)
    note = safe(ROOT / "setup-notes" / "volume.json")
    if note.is_file():
        expected = json.loads(note.read_text(encoding="utf-8"))
        mount = Path(expected["mount"])
        if not mount.is_mount() or not ROOT.is_relative_to(mount):
            raise RuntimeError("Reconnect the original project SSD before continuing.")
        info = disk_info(mount)
        if str(info.get("VolumeUUID", "")).upper() != str(expected["uuid"]).upper() or info.get("MountPoint") != str(mount):
            raise RuntimeError("Mounted volume does not match the saved project SSD identity.")
        if info.get("WritableVolume") is False:
            raise RuntimeError("The project SSD is read-only.")
        return expected
    if not initialize:
        raise RuntimeError("SSD identity record is missing. Run scripts/setup.py --initialize-ssd after checking this dedicated folder is on the intended external SSD.")
    mount = next((item for item in (ROOT, *ROOT.parents) if item.is_mount()), None)
    if mount is None or (not WINDOWS and mount.parent != Path("/Volumes")):
        raise RuntimeError("Setup requires an existing mounted external volume below /Volumes. It never creates or moves a volume.")
    info = disk_info(mount)
    if (not WINDOWS and info.get("Internal") is not False) or not info.get("VolumeUUID") or info.get("WritableVolume") is False:
        raise RuntimeError("The folder's mounted volume is not an identifiable writable external drive.")
    if info.get("FilesystemType") not in {"exfat", "apfs", "hfs", "ntfs"}:
        raise RuntimeError("This filesystem has not been validated. Use the recorded supported setup or request a scoped environment alternative.")
    expected = dict(mount=str(mount), uuid=info["VolumeUUID"], filesystem=info.get("FilesystemType"),
                    architecture=platform.machine(), macos=platform.mac_ver()[0], system=platform.system())
    note.parent.mkdir(exist_ok=True)
    # The mounted identity was verified immediately before this first scoped write.
    descriptor = os.open(note, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as target:
        json.dump(expected, target, indent=2)
        target.flush()
        os.fsync(target.fileno())
    return expected


def atomic_json(path: Path, document: dict | list) -> None:
    check_volume()
    safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as target:
            json.dump(document, target, indent=2, allow_nan=False)
            target.flush()
            os.fsync(target.fileno())
        check_volume()
        os.replace(temporary, path)
    finally:
        try:
            check_volume()
            safe(temporary).unlink(missing_ok=True)
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass


def local_environment(*, offline: bool) -> dict[str, str]:
    check_volume()
    environment = os.environ.copy()
    locations = {
        "TMPDIR": ROOT / "tmp", "TMP": ROOT / "tmp", "TEMP": ROOT / "tmp",
        "PYTHONPYCACHEPREFIX": ROOT / "runtime" / "pycache",
        "PIP_CACHE_DIR": ROOT / "runtime" / "pip-cache",
        "XDG_CACHE_HOME": ROOT / "runtime" / "cache",
        "HF_HOME": ROOT / "model-cache" / "huggingface",
        "HF_HUB_CACHE": ROOT / "model-cache" / "huggingface" / "hub",
        "TORCH_HOME": ROOT / "model-cache" / "torch",
        "MPLCONFIGDIR": ROOT / "runtime" / "matplotlib",
        "npm_config_cache": ROOT / "runtime" / "npm-cache",
    }
    for name, path in locations.items():
        check_volume()
        safe(path).mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    environment.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                       PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL="https://pypi.org/simple",
                       HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTORCH_ENABLE_MPS_FALLBACK="0",
                       HF_HUB_OFFLINE="1" if offline else "0", TRANSFORMERS_OFFLINE="1" if offline else "0",
                       LSS_WORKSPACE_ROOT=str(ROOT))
    # Use PyTorch's default safety limits even if a parent shell changed them.
    environment.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    environment.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO", None)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    for name in ("PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        environment.pop(name, None)
    return environment


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with safe(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()
