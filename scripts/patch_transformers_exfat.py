"""Recorded one-line compatibility patch for Transformers 5.16.1 on macOS ExFAT.

AppleDouble resource sidecars named ._module.py are binary metadata, not Python.
The upstream lazy-import scanner reads every .py suffix and otherwise fails with
UnicodeDecodeError. Skip these entries alongside its existing __pycache__ filter.
No model math or inference output is changed. This script is idempotent and fails
closed for unexpected versions/source. Do not use it for another package version.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
sys.path.insert(0, str(APP))
from backend.storage import Storage

store = Storage(ROOT)
store.check()
assert importlib.metadata.version("transformers") == "5.16.1", "Patch requires exactly Transformers 5.16.1"
path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "transformers" / "utils" / "import_utils.py"
path = store.safe_path(path.relative_to(ROOT))
old = '            if entry.name == "__pycache__":\n'
new = '            if entry.name == "__pycache__" or entry.name.startswith("._"):\n'
source = path.read_text(encoding="utf-8")
if new in source:
    print("Transformers ExFAT patch already applied")
elif source.count(old) == 1:
    patched = source.replace(old, new)
    store._write_atomic(path.relative_to(ROOT), patched.encode())
    report = dict(package="transformers", version="5.16.1", file="utils/import_utils.py", reason="Skip binary AppleDouble sidecars in Python-module scanner on ExFAT", before_sha256=hashlib.sha256(source.encode()).hexdigest(), after_sha256=hashlib.sha256(patched.encode()).hexdigest(), old=old.strip(), new=new.strip())
    store._write_atomic(Path("setup-notes/transformers-exfat-patch.json"), json.dumps(report,indent=2).encode())
    print("Applied recorded one-line Transformers ExFAT patch")
else:
    raise SystemExit("Unexpected Transformers source; patch not applied")
