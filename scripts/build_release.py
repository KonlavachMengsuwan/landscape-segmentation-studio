"""Package reviewed, committed source and prebuilt UI without local data or weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

from workspace import APP, ROOT, atomic_json, check_volume, safe

EXCLUDED_PARTS = {".git", "node_modules", "runtime", "model-cache", "projects", "exports", "setup-notes", "tmp", "publication", "releases", "SAM1", "SAM2", "SAM3", "SAM3.1", "__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pt", ".pth", ".safetensors", ".ckpt", ".onnx", ".pem", ".key", ".pyc", ".log"}
DIRECTORIES = {"backend", "config", "docs", "frontend", "licenses", "samples", "scripts", "tests"}
SECRET_PATTERNS = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")


def validate_relative(name: str, *, built: bool = False) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError(f"Unsafe package path: {name}")
    if any(p in EXCLUDED_PARTS or p.startswith("._") or p.startswith(".env") or p == ".DS_Store" for p in path.parts):
        raise ValueError(f"Private/local package path: {name}")
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        raise ValueError(f"Excluded package file: {name}")
    if len(path.parts) > 1 and path.parts[0] not in DIRECTORIES:
        raise ValueError(f"Unreviewed package directory: {name}")
    if "dist" in path.parts and (not built or path.parts[:2] != ("frontend", "dist")):
        raise ValueError(f"Built file not permitted here: {name}")
    return path


def write_archive(destination: Path, sources: list[Path], prefix: str, system: str | None = None) -> dict:
    if destination.exists():
        raise RuntimeError(f"Archive exists; choose a new version instead of overwriting {destination.name}.")
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            if system == "mac" and source.suffix == ".cmd":
                continue
            if system == "windows" and source.suffix == ".command":
                continue
            check_volume()
            if source.is_symlink():
                raise RuntimeError("Do not package symlinked source files.")
            safe(source)
            relative = source.relative_to(APP).as_posix()
            validate_relative(relative, built=system is not None)
            content = source.read_bytes()
            if SECRET_PATTERNS.search(content):
                raise RuntimeError(f"Credential-shaped content found in {relative}; review before packaging.")
            info = zipfile.ZipInfo(prefix + relative)
            info.create_system = 3
            info.external_attr = (0o100755 if source.suffix == ".command" else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed.")
        count = len(archive.infolist())
    with destination.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": destination.name, "bytes": destination.stat().st_size, "sha256": checksum, "files": count}


def main() -> None:
    check_volume()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=APP).strip():
        raise RuntimeError("Review and commit source changes before packaging; untracked/modified files are not silently included.")
    version = json.loads((APP / "frontend/package.json").read_text())["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError("Expected a numeric release version.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=APP, text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=APP).decode().split("\0")
    sources = [APP / str(validate_relative(name)) for name in tracked if name]
    built = sorted(p for p in (APP / "frontend/dist").rglob("*") if p.is_file() and not p.name.startswith("._"))
    for p in built:
        validate_relative(p.relative_to(APP).as_posix(), built=True)
    if not (APP / "frontend/dist/index.html").is_file():
        raise RuntimeError("Build the frontend first.")
    for required in ["LICENSE", "NOTICE", "START_HERE.md", "THIRD_PARTY_NOTICES.md", "licenses/SAM3-LICENSE.txt", "licenses/REACT-MIT.txt", "licenses/REACT-DOM-MIT.txt", "licenses/SCHEDULER-MIT.txt", "licenses/VITE-MIT.txt"]:
        if APP / required not in sources:
            raise RuntimeError(f"Required tracked notice or guide missing: {required}")
    releases = safe(ROOT / "releases")
    publication = safe(ROOT / "publication")
    releases.mkdir(exist_ok=True)
    publication.mkdir(exist_ok=True)
    destinations = [releases / f"LandscapeSegmentationStudio-{system}-{version}.zip" for system in ["Mac-Apple-Silicon", "Windows-x64"]]
    source_zip = publication / f"LandscapeSegmentationStudio-GitHub-source-{version}.zip"
    manifests = [releases / f"release-manifest-{version}.json", publication / f"source-manifest-{version}.json"]
    if any(p.exists() for p in [*destinations, source_zip, *manifests]):
        raise RuntimeError("This release version already has artifacts; do not overwrite them.")
    common = {"version": version, "source_revision": revision, "excludes": "Model weights, user projects/images, exports, environments, caches, credentials, local test reports and .git history"}
    source = {**common, **write_archive(source_zip, sources, "landscape-segmentation-studio/"), "purpose": "Extract and upload contents as repository root; source only, frontend build not included"}
    reports = []
    for destination, system in zip(destinations, ["mac", "windows"]):
        reports.append({**common, **write_archive(destination, sources + built, "LandscapeSegmentationStudio/app/", system), "purpose": "User setup ZIP with prebuilt interface; Python prerequisite and setup downloads required", "native_validation": "Existing Mac M4 app/inference verified; fresh recipient must run setup tests" if system == "mac" else "Pending native Windows/Intel/NVIDIA validation"})
    atomic_json(manifests[0], reports)
    atomic_json(manifests[1], source)
    print(json.dumps({"source": source, "platforms": reports}, indent=2))


if __name__ == "__main__":
    main()
