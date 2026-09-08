# Storage validation

Verified 7 September 2026 with native ARM64 Python 3.12.14 and the project-local
SSD virtual environment: Pillow 12.3.0, NumPy 2.5.3, pytest 9.1.1. The suite
reported **18 passed, 26 subtests passed** in 2.68 seconds on the connected
ExFAT SSD. Runtime varies; this is a storage check, not inference performance.

From the workspace root, the actual test command was:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=app TMPDIR="$PWD/tmp" \
  runtime/venv/bin/python -m pytest app/tests/test_storage.py \
  --basetemp tmp/test-storage -o cache_dir=tmp/pytest-cache -q
```

The tests verify original-byte preservation and hashes; canonical RGB pixels;
all eight EXIF orientations in deliberately asymmetric PNG and TIFF fixtures;
white compositing of transparency; ordinary grayscale TIFF; rejection of
scientific/multipage/invalid/oversized inputs; exact RLE and 0/255 PNG mask
membership with holes, disconnected parts, and overlapping instances; versioned
save/reopen; stale-save rejection; nonfinite JSON rejection; injected pointer
commit failure preserving the previously committed version; corrupt-source and
corrupt-project hash checks; path traversal and symlink-component rejection;
and simulated missing/wrong-volume rejection without replacement directories.

Fixtures and pytest caches remain inside the workspace's `tmp` directory.
ExFAT creates AppleDouble sidecars; history test enumeration explicitly ignores
those sidecars. Production project enumeration accepts only generated IDs.
ExFAT cannot create native symlinks, so the symlink test injects a reported
symlink component into the same production rejection branch. A native-symlink
filesystem integration test remains unperformed. Drive removal and power-loss
durability were not tested by unplugging the real SSD. No inference, frontend
alignment, style rendering, or application export rendering is established by
this storage suite.
