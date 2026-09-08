#!/usr/bin/env python3
"""Explicit setup download of pinned official SAM 2 / 2.1 files, with integrity checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from urllib.parse import urlparse
import urllib.request
import uuid

from platform_support import NOFOLLOW
from workspace import APP, ROOT, atomic_json, check_volume, digest, safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[entry["id"] for entry in json.loads((APP / "config/model-manifest.json").read_text(encoding="utf-8"))] + ["all", "baseline"], default="all")
    parser.add_argument("--verify-only", action="store_true", help="Check installed files without network access or writes")
    args = parser.parse_args()
    try:
        check_volume()
        manifest = json.loads(safe(APP / "config" / "model-manifest.json").read_text(encoding="utf-8"))
        selected = [entry for entry in manifest if args.model == "all" or (args.model == "baseline" and entry["id"] in ("tiny", "small")) or entry["id"] == args.model]
        required = 0
        for entry in selected:
            if entry["id"] not in {"tiny", "small", "sam2.1-base-plus", "sam2.1-large", "sam2-tiny", "sam2-small", "sam2-base-plus", "sam2-large"}:
                raise RuntimeError("Unexpected model ID in the pinned manifest.")
            for item in entry["files"]:
                if item["name"] not in {"config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json"}:
                    raise RuntimeError("Unexpected checkpoint filename in the manifest.")
                url = urlparse(item["url"])
                expected_prefix = f"/{entry['checkpoint']}/resolve/{entry['revision']}/"
                if url.scheme != "https" or url.netloc != "huggingface.co" or not url.path.startswith(expected_prefix):
                    raise RuntimeError("Download source differs from the pinned official publisher.")
                path = safe(ROOT / "model-cache" / entry["directory"] / item["name"])
                valid = path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
                if not valid:
                    if args.verify_only:
                        raise RuntimeError(f"{entry['id']} / {item['name']} is missing or failed integrity verification.")
                    required += item["bytes"]
        if args.verify_only:
            print("Selected official model files passed size and SHA256 checks. No network was used.")
            return 0
        if shutil.disk_usage(ROOT).free < required + 1024**3:
            raise RuntimeError("Insufficient SSD space for the required files and a 1 GiB safety margin.")
        for entry in selected:
            for item in entry["files"]:
                check_volume()
                path = safe(ROOT / "model-cache" / entry["directory"] / item["name"])
                if path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]:
                    continue
                check_volume()
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
                print(f"Downloading official SAM 2 / 2.1 {entry['id']} / {item['name']} ({item['bytes']:,} bytes)", flush=True)
                try:
                    request = urllib.request.Request(item["url"], headers={"User-Agent": "LandscapeSegmentationStudio-setup/1"})
                    checksum, count = hashlib.sha256(), 0
                    with urllib.request.urlopen(request, timeout=30) as response:
                        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600)
                        with os.fdopen(descriptor, "wb") as output:
                            while True:
                                chunk = response.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                check_volume()
                                count += len(chunk)
                                if count > item["bytes"]:
                                    raise RuntimeError("Download exceeded its verified expected size.")
                                checksum.update(chunk)
                                output.write(chunk)
                            output.flush()
                            os.fsync(output.fileno())
                    if count != item["bytes"] or checksum.hexdigest() != item["sha256"]:
                        raise RuntimeError("Downloaded checkpoint failed size or SHA256 verification.")
                    check_volume()
                    safe(path)
                    os.replace(temporary, path)
                finally:
                    try:
                        check_volume()
                        safe(temporary).unlink(missing_ok=True)
                    except (OSError, RuntimeError):
                        pass
        atomic_json(ROOT / "setup-notes" / "hf-model-manifest.json", manifest)
        print("Pinned SAM 2 / 2.1 model setup completed. Normal inference uses only local files.")
        return 0
    except Exception as error:
        print(f"Model setup stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
