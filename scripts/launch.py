#!/usr/bin/env python3
"""Start/stop only this workspace's loopback server. No inference downloads."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from platform_support import WINDOWS, NOFOLLOW, python_in, process_command
if not WINDOWS:
    import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from workspace import APP, ROOT, atomic_json, check_volume, local_environment, safe

PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
STATE = ROOT / "runtime" / "studio-process.json"
SCRIPT = APP / "scripts" / "launch.py"
WORKSPACE_ID = hashlib.sha256(str(ROOT).encode()).hexdigest()


def owned_process(record: dict) -> bool:
    if record.get("root") != str(ROOT) or record.get("workspace_id") != WORKSPACE_ID:
        return False
    pid, marker = record.get("pid"), record.get("instance")
    if not isinstance(pid, int) or pid <= 1 or not isinstance(marker, str) or len(marker) != 32:
        return False
    returncode, command = process_command(pid)
    return (returncode == 0 and str(SCRIPT) in command
            and re.search(r"(?:^|\s)--studio-instance=" + re.escape(marker) + r"(?=\s|$)", command) is not None
            and re.search(r"(?:^|\s)--workspace-id=" + WORKSPACE_ID + r"(?=\s|$)", command) is not None)


@contextmanager
def management_lock():
    check_volume()
    path = safe(ROOT / "runtime" / "studio-launch.lock")
    path.parent.mkdir(exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | NOFOLLOW, 0o600)
    try:
        try:
            if WINDOWS:
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.fstat(descriptor).st_size == 0: os.write(descriptor, b'0')
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another Studio start or stop command is in progress. Retry when it finishes.") from error
        yield
    finally:
        os.close(descriptor)


def read_state() -> dict | None:
    path = safe(STATE)
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else None
    except (OSError, ValueError):
        raise RuntimeError("Application process record is unreadable. No process was stopped.")


def status_matches(instance: str) -> bool:
    try:
        # Ignore shell proxy configuration for this strictly loopback request.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(URL + "/api/status", timeout=1) as response:
            result = json.load(response)
        return result.get("instance") == instance
    except (OSError, ValueError, urllib.error.URLError):
        return False


def open_browser():
    import webbrowser
    webbrowser.open(URL)


def check_port_available() -> None:
    # Match the server's restart semantics: a closed connection in TIME_WAIT
    # is not a live listener. SO_REUSEADDR does not enable SO_REUSEPORT sharing.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE if WINDOWS else socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", PORT))
            probe.listen(1)
        except OSError as error:
            raise RuntimeError(f"Port {PORT} is already in use. No unrelated process was stopped.") from error


def start(no_open: bool) -> int:
    check_volume()
    previous = read_state()
    if previous and owned_process(previous):
        if status_matches(previous["instance"]):
            print(f"Landscape Segmentation Studio is already running at {URL}")
            if not no_open:
                open_browser()
            return 0
        raise RuntimeError("This application's process is already starting or stopping. Use Stop Studio, then retry.")
    interpreter = safe(python_in(ROOT / "runtime" / "venv"))
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise RuntimeError("The project Python environment is missing. Follow docs/MAC_SETUP.md.")
    if not safe(APP / "frontend" / "dist" / "index.html").is_file():
        raise RuntimeError("The frontend build is missing. Build it with the documented npm command before starting.")
    if not safe(APP / "backend" / "main.py").is_file():
        raise RuntimeError("The backend entry point is missing. Application implementation is incomplete.")
    for key in ("tiny", "small"):
        for name in ("model.safetensors", "config.json", "preprocessor_config.json", "processor_config.json"):
            if not safe(ROOT / "model-cache" / f"sam2.1-{key}" / name).is_file():
                raise RuntimeError(f"SAM 2.1 {key.title()} files are missing. Run scripts/download_models.py.")
    check_port_available()
    instance = uuid.uuid4().hex
    environment = local_environment(offline=True)
    environment["LSS_INSTANCE"] = instance
    command = [str(interpreter), "-B", str(SCRIPT), "serve", f"--studio-instance={instance}", f"--workspace-id={WORKSPACE_ID}"]
    check_volume()
    log = safe(ROOT / "runtime" / "studio.log")
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND | NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "ab", buffering=0) as output:
        child = subprocess.Popen(command, cwd=APP, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=not WINDOWS,
                                 creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if WINDOWS else 0)
    record = dict(pid=child.pid, instance=instance, root=str(ROOT), workspace_id=WORKSPACE_ID,
                  url=URL, started_at=time.time())
    try:
        atomic_json(STATE, record)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            check_volume()
            if child.poll() is not None:
                raise RuntimeError("The server exited during startup. Read runtime/studio.log for the exact error.")
            if status_matches(instance):
                print(f"Landscape Segmentation Studio is ready at {URL}")
                if not no_open:
                    open_browser()
                return 0
            time.sleep(.25)
        raise RuntimeError("The server did not become ready within 30 seconds. Check runtime/studio.log.")
    except Exception:
        if owned_process(record):
            os.kill(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if owned_process(record):
                    os.kill(child.pid, (signal.SIGTERM if WINDOWS else signal.SIGKILL))
        raise


def stop() -> int:
    check_volume()
    record = read_state()
    if record is None:
        print("No application process is recorded for this workspace.")
        return 0
    if not owned_process(record):
        # A stale PID is never sufficient authorization to signal a process.
        print("No matching application process is running. No process was stopped.")
        return 0
    os.kill(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not owned_process(record):
            check_volume()
            safe(STATE).unlink(missing_ok=True)
            print("Landscape Segmentation Studio stopped.")
            return 0
        time.sleep(.25)
    if owned_process(record):
        os.kill(record["pid"], (signal.SIGTERM if WINDOWS else signal.SIGKILL))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not owned_process(record):
            check_volume()
            safe(STATE).unlink(missing_ok=True)
            print("Landscape Segmentation Studio stopped after ending its unfinished work.")
            return 0
        time.sleep(.25)
    raise RuntimeError("The matching process has not exited yet. Its saved project history remains on the SSD.")


def serve(args):
    check_volume()
    if args.studio_instance != os.environ.get("LSS_INSTANCE") or args.workspace_id != WORKSPACE_ID:
        raise RuntimeError("Server instance markers do not match this workspace.")
    sys.path.insert(0, str(APP))
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=PORT, workers=1,
                access_log=False, server_header=False, timeout_graceful_shutdown=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "serve"])
    parser.add_argument("--no-open", action="store_true", help="Verify readiness without opening a browser, for testing")
    parser.add_argument("--studio-instance")
    parser.add_argument("--workspace-id")
    args = parser.parse_args()
    try:
        if args.action == "serve":
            return serve(args)
        with management_lock():
            return stop() if args.action == "stop" else start(args.no_open)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Studio: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
