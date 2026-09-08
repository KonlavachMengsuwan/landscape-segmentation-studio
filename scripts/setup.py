#!/usr/bin/env python3
"""Install the pinned native Python wheel environment within this SSD workspace."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from platform_support import WINDOWS, python_in
from workspace import APP, ROOT, atomic_json, check_volume, local_environment, safe


def configured_base() -> Path | None:
    configuration = safe(ROOT / "runtime" / "venv" / "pyvenv.cfg")
    if configuration.is_file():
        for line in configuration.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "executable":
                return Path(value.strip())
    return None


def probe_filesystem(environment):
    check_volume()
    stem = "setup-probe-" + uuid.uuid4().hex
    executable = safe(ROOT / "tmp" / stem)
    replacement = safe(ROOT / "tmp" / (stem + ".replacement"))
    try:
        with executable.open("xb") as target:
            target.write(b"print('probe passed')\n" if WINDOWS else b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)
        subprocess.run([sys.executable, str(executable)] if WINDOWS else [str(executable)], check=True, env=environment, timeout=5)
        with replacement.open("xb") as target:
            target.write(b"atomic replacement verified\n")
        check_volume()
        os.replace(replacement, executable)
        if executable.read_bytes() != b"atomic replacement verified\n":
            raise RuntimeError("Atomic replacement probe failed.")
    finally:
        check_volume()
        executable.unlink(missing_ok=True)
        replacement.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", help="Existing native Python 3.12 (ARM64 Mac / x64 Windows) executable; read only")
    parser.add_argument("--initialize-ssd", action="store_true", help="Record this existing dedicated folder's external-volume identity when missing")
    parser.add_argument("--download-models", action="store_true", help="Download only the pinned SAM 2.1 files after package setup")
    parser.add_argument("--nvidia", action="store_true", help="Windows only: install official CUDA 13 PyTorch wheels; requires a compatible NVIDIA driver")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--rebuild-env", action="store_true", help="Preserve the old project venv under runtime and create a new one at the original path")
    args = parser.parse_args()
    try:
        if not (APP / "frontend" / "dist" / "index.html").is_file():
            raise RuntimeError("The built interface is missing. Use a platform setup ZIP from GitHub Releases, or build frontend first as described in README.md. GitHub Code > Download ZIP contains source only.")
        volume = check_volume(initialize=args.initialize_ssd)
        environment = local_environment(offline=False)
        environment["LSS_DEVICE"] = args.device
        from launch import owned_process, read_state
        running = read_state()
        if running and owned_process(running):
            raise RuntimeError("Stop Studio before changing the dependency environment.")
        base = Path(args.python).expanduser().absolute() if args.python else (configured_base() or Path(sys.executable))
        if base is None or not base.is_file() or not os.access(base, os.X_OK):
            raise RuntimeError("Provide --python with an existing native Python 3.12 (ARM64 Mac / x64 Windows) executable. No system Python will be installed or changed.")
        command = "import json,platform,sys; print(json.dumps({'architecture':platform.machine(),'version':list(sys.version_info[:3])}))"
        inspected = subprocess.run([str(base), "-I", "-c", command], check=True, text=True, capture_output=True, env=environment, timeout=15)
        python = json.loads(inspected.stdout)
        if python["architecture"].lower() not in ({"amd64", "x86_64"} if WINDOWS else {"arm64"}) or python["version"][:2] != [3, 12]:
            raise RuntimeError("The pinned environment requires native Python 3.12 (ARM64 Mac / x64 Windows), not Rosetta or system Python.")
        minimum_gib = 64 if volume["filesystem"] == "exfat" else 12
        if shutil.disk_usage(ROOT).free < minimum_gib * 1024**3:
            raise RuntimeError(f"Keep at least {minimum_gib} GiB free for the environment, wheel caches, and filesystem allocation overhead before setup.")
        probe_filesystem(environment)
        venv = safe(ROOT / "runtime" / "venv")
        if args.rebuild_env and venv.exists():
            if not (venv / "pyvenv.cfg").is_file():
                raise RuntimeError("Existing runtime/venv is not a recognized environment and was not changed.")
            check_volume()
            backup = safe(ROOT / "runtime" / ("venv.previous-" + uuid.uuid4().hex[:12]))
            os.rename(venv, backup)
            print(f"Previous project environment preserved in runtime/{backup.name}")
        if not venv.exists():
            check_volume()
            subprocess.run([str(base), "-m", "venv", "--copies", str(venv)], check=True, cwd=APP, env=environment)
        elif not (venv / "pyvenv.cfg").is_file():
            raise RuntimeError("runtime/venv is nonempty and is not a recognized environment. It was not overwritten.")
        interpreter = safe(python_in(venv))
        inspected_venv = subprocess.run([str(interpreter), "-I", "-c", command], check=True, capture_output=True, text=True, env=environment, timeout=15)
        installed_python = json.loads(inspected_venv.stdout)
        if installed_python["architecture"].lower() not in ({"amd64", "x86_64"} if WINDOWS else {"arm64"}) or installed_python["version"][:2] != [3, 12]:
            raise RuntimeError("Existing runtime/venv is not the required native Python 3.12 environment; it was not replaced.")
        if args.nvidia:
            if not WINDOWS: raise RuntimeError("NVIDIA installation is only offered on Windows.")
            subprocess.run([str(interpreter), "-m", "pip", "install", "--only-binary=:all:",
                            "torch==2.12.1+cu130", "torchvision==0.27.1+cu130", "--index-url", "https://download.pytorch.org/whl/cu130"],
                           check=True, cwd=APP, env=environment)
            args.device = "cuda"
        check_volume()
        subprocess.run([str(interpreter), "-m", "pip", "install", "--only-binary=:all:", "-r", str(APP / "requirements.lock")],
                       check=True, cwd=APP, env=environment)
        check_volume()
        if not WINDOWS:
            subprocess.run([str(interpreter), "-B", str(APP / "scripts" / "patch_transformers_exfat.py")],
                           check=True, cwd=APP, env=environment)
        subprocess.run([str(interpreter), "-m", "pip", "check"], check=True, cwd=APP, env=environment)
        device_probe = ("import sys; sys.path.insert(0,'.'); from backend.models import ModelManager; from backend.storage import Storage; "
                        "from pathlib import Path; import torch; s=Storage(Path('..').resolve()); m=ModelManager(s.root,s.guard); "
                        "x=torch.arange(16,device=m.device,dtype=torch.float32).reshape(4,4); y=x@x.T; m.sync(); "
                        "assert float(y[0,0].cpu())==14.; print('Local tensor probe passed:',m.device)")
        probe_env = local_environment(offline=True); probe_env['LSS_DEVICE'] = args.device
        subprocess.run([str(interpreter), "-B", "-c", device_probe], check=True, cwd=APP, env=probe_env)
        atomic_json(ROOT / "setup-notes" / "python-environment.json", {
            "python": ".".join(map(str, installed_python["version"])), "architecture": installed_python["architecture"],
            "environment": "runtime/venv", "base_record": "runtime/venv/pyvenv.cfg",
            "packages": "app/requirements.lock", "binary_only": True,
            "filesystem": volume["filesystem"], "executable_probe": "passed", "atomic_replace_probe": "passed",
        })
        if args.download_models:
            subprocess.run([str(interpreter), "-B", str(APP / "scripts" / "download_models.py"), "--model", "baseline"],
                           check=True, cwd=APP, env=environment)
        print("Project-local Python setup passed. Run model compatibility checks, then use Start Studio.")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Setup stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
