# Windows setup (x64 Intel/AMD; optional NVIDIA)

This release includes a Windows installation path and local CPU/CUDA adapters. **Native Windows installation, drive checks, process management, and GPU execution have not been tested on a Windows machine.** Development and inference validation were performed on the Mac mini M4 only. No other computer was accessed. Treat the Windows bundle as a test release until the checks below pass on the target PC.

1. Extract the Windows ZIP into a **new dedicated folder** on a connected local NTFS or ExFAT drive. Keep the resulting `LandscapeSegmentationStudio/app/` layout. Do not extract over another project. A local disk is required; network shares are unsupported.
2. Install Python **3.12 x64** from [python.org](https://www.python.org/downloads/windows/) yourself if it is not already available. The setup scripts use the `py -3.12` launcher. No administrator installation, system Python modification, shell configuration change, or driver installation is performed by Studio.
3. Double-click `app/Setup Windows.cmd` for CPU use. If you have an NVIDIA GPU with a CUDA 13 compatible driver, use `app/Setup Windows NVIDIA.cmd` instead. The latter installs pinned official PyTorch CUDA 13 wheels into this project's environment. It never installs a GPU driver or CUDA toolkit system-wide. Unsupported drivers cause a clear failed device probe.
4. Setup downloads the two baseline SAM 2.1 checkpoints, checks their sizes and SHA256 hashes, installs pinned binary packages, and runs bounded **real local model inference**. Internet access is needed only for this setup/download phase. Read `setup-notes/verify-*.log` if a check fails; an unverified model remains disabled.
5. Double-click `app/Start Studio.cmd`. Your browser opens `http://127.0.0.1:8765/`. Use `app/Stop Studio.cmd` to stop only this workspace's process.

The browser interface is bundled: Node.js is not needed for installation or normal use. Python remains a prerequisite; this is a setup bundle, not a signed single-executable Windows installer. The bundle contains no user images, research projects, gated checkpoints, tokens, Mac environment, or machine-specific verification results.

## Additional checkpoints

From a terminal in `app/`:

```bat
..\runtime\venv\Scripts\python.exe -B scripts\download_models.py --model all
..\runtime\venv\Scripts\python.exe -B scripts\verify_models.py --model all
```

This adds all eight SAM 2 / SAM 2.1 sizes, one model in memory at a time. Any individual ID from `config/model-manifest.json` can replace `all`. Large models need more RAM/VRAM; keep Tiny as the default. No out-of-memory retry silently changes device or precision. To explicitly test CPU, pass `--device cpu` to `verify_models.py`, and set `LSS_DEVICE=cpu` in the terminal used to launch Studio. Normal automatic selection prefers CUDA on a supported Windows GPU, otherwise CPU.

For your already supplied original SAM ViT-H and SAM 3.1 checkpoints, see [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md). Obtain gated weights personally; do not redistribute them inside the installer. Their local conversion uses strict shape/key checks and preserves source files.

## Native validation checklist

Before calling a Windows installation verified, check setup completion; real point, box, positive/negative, and automatic results; UTF-8 project names and paths containing spaces; image import and exact original hashes; save/reopen/export/restore; drive disconnect/reconnect; launch, immediate restart, and safe shutdown with an unrelated process on port 8765; and NVIDIA memory handling on the actual GPU. A successful Mac test or Windows wheel download is not evidence that these Windows checks passed.

All image processing stays on that computer. There are no accounts, cloud inference, remote model servers, LAN listeners, or video workflows.
