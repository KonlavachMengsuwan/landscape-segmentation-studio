# ExFAT environment findings

The SanDisk SSD uses ExFAT with 1 MiB allocation blocks. Executable Python copies and atomic file replacement pass. ExFAT does not support ordinary Unix symlinks, and thousands of small dependency files consume much more allocated storage than their logical sizes.

macOS writes binary AppleDouble metadata sidecars named `._filename`. Meta SAM 2's source dependencies failed to build because pip/setuptools count sidecars as extra metadata directories or attempt cleanup after their associated directory has already disappeared. Both isolated and direct source-build routes were attempted. Do not silently ignore these build failures or claim Meta SAM 2 was installed.

The verified runtime uses only prebuilt Python wheels and Hugging Face's documented SAM 2 implementation. Transformers 5.16.1 initially fails to import because its lazy-module scanner interprets `._module.py` as Python source, raising UnicodeDecodeError. `scripts/patch_transformers_exfat.py` applies one explicit change to that scanner: skip `._` entries alongside its existing `__pycache__` exclusion. It verifies the installed version and source pattern, records before/after SHA256 in setup-notes, and is idempotent. It changes no model computation or checkpoint. It must be rerun after a fresh dependency installation and must not be applied blindly to another version.

No drive formatting, repair, global Python installation, shell changes, or system privileges are used. An internal runtime location was offered during investigation but was not needed or created. Both SAM 2.1 checkpoints passed real point/box and automatic inference on the SSD-only patched runtime. A full-resolution 3072 × 2048 application run also passed.
