# Mac setup and local launch

The development machine checked on 7 September 2026 is a Mac mini with Apple M4, 16 GiB unified memory, macOS 26.6.2, and native ARM64 Python 3.12.14. The dedicated project is on the connected SanDisk Extreme SSD. The mounted volume is named `Extreme SSD`; the local identity record is in `setup-notes/volume.json` outside the Git repository.

The SSD is ExFAT. Executable files and atomic replacement were tested. macOS AppleDouble sidecars disrupted source dependency builds, so the delivered Python environment uses prebuilt wheels and a recorded one-line Transformers import-scanner patch. Its 1 MiB allocation blocks make small package files expensive: the installed environment currently occupies about **49.8 GiB of allocated SSD storage**, plus about 2.0 GiB of pip cache. These are filesystem allocation measurements, not RAM usage or checkpoint download sizes. Fresh ExFAT setup requires at least 64 GiB free. The actual Tiny and Small model tests passed with this environment. See [ExFAT details](EXFAT_COMPATIBILITY.md) and [model results](MODEL_COMPATIBILITY.md). No internal development environment or system-wide installation was needed.

## New installation from the Mac bundle

Extract into a new dedicated folder on the connected external SSD, keeping `LandscapeSegmentationStudio/app/`. With native ARM64 Python **3.12** installed and available as `python3.12`, double-click `app/Setup Mac.command`. It creates a project-local environment, installs pinned binary dependencies, downloads baseline Tiny and Small, and runs bounded real local compatibility checks. Then open `Start Studio.command`. The bundle already contains the frontend build; Node is needed only to develop/rebuild it. The current installation is already set up; do not reinstall it for these improvements.

For all eight SAM 2 / SAM 2.1 sizes, run `scripts/download_models.py --model all` and `scripts/verify_models.py --model all` with the project Python. For the supplied original SAM and SAM3.1 files, see [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md). The source files in SAM1, SAM2, and SAM3.1 are never moved, renamed, or overwritten.

## Everyday use

1. Connect the same SSD.
2. Open `app/Start Studio.command` in Finder, or run `"./Start Studio.command"` from the `app` directory.
3. The launcher verifies the SSD, the environment, both installed checkpoints, the frontend build, and port 8765. It opens [the local app](http://127.0.0.1:8765) only after the matching backend responds successfully.
4. Save your work and open `app/Stop Studio.command` when finished. Closing the browser alone does not stop Python.

The stop script validates a unique process marker, workspace identity, and PID. It never searches for and kills arbitrary Python processes. It requests shutdown, then ends only its matching process if unfinished work prevents exit. Unsaved browser changes remain unsaved. Application persistence uses earlier valid versions and atomic save pointers.

The backend binds only to `127.0.0.1:8765`. There is no LAN binding, tunnel, hosted inference, camera input, or video workflow. If the port belongs to another application, the launcher reports the conflict and leaves that process alone. Startup errors are recorded in `runtime/studio.log`; that file is outside Git and may contain local paths.

## Storage layout

All paths here are relative to the dedicated SSD folder containing `app`:

| Location | Purpose |
| --- | --- |
| `app/` | Git repository, source, scripts, locally bundled UI assets |
| `runtime/venv/` | Native Python environment with copied interpreter |
| `runtime/` | Python, pip, npm, and other project caches; process record and local log |
| `model-cache/sam2.1-tiny/`, `model-cache/sam2.1-small/` | Active pinned Transformers checkpoints |
| `projects/` | Project versions, imported original bytes, canonical images, run data |
| `exports/` | Explicitly generated exports |
| `tmp/` | Project-local temporary files |
| `setup-notes/` | Volume identity, integrity manifests, real test evidence, patch record |

The launcher sets project-local `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`, `PYTHONPYCACHEPREFIX`, pip/npm caches, and temporary paths. Normal inference enforces offline loading and disables implicit MPS CPU fallback. PyTorch memory safeguards retain their defaults. It does not change shell startup files, `HOME`, global network settings, or Codex configuration. macOS may still manage system libraries, browser storage, process logs, code-signing and Metal driver caches, and Finder metadata outside these paths.

torchvision is a shared upstream library containing image and video utilities; this app requires its image processing and NMS operations only. No PyAV, Decord, FFmpeg package, or video extras are installed for this release.

The current virtual environment uses the existing, read-only Python base bundled with Codex. `runtime/venv/pyvenv.cfg` records its local location. `venv --copies` copies the executable but does not bundle the entire base standard library. If a Codex update removes that base, rebuild using an existing native Python 3.12 executable you select. Do not treat the virtual environment as a portable Python distribution.

## Reproduce the Python environment

Use an existing native ARM64 Python **3.12** executable. The scripts will not install a system Python or administrator prerequisites. Run from the dedicated SSD folder, replacing the example executable with the selected existing one:

```sh
"/path/to/native/python3.12" -B app/scripts/setup.py \
  --python "/path/to/native/python3.12" --initialize-ssd --download-models
```

`--initialize-ssd` creates an identity record only if absent, only for the existing writable external volume containing this dedicated root. It never creates a missing `/Volumes` directory or modifies a drive. Existing identity mismatches fail closed. The script checks native architecture/version, executable file behavior, atomic replacement, and free storage; it creates `runtime/venv` with `--copies`, installs `requirements.lock` using `pip --only-binary=:all:`, applies the recorded Transformers 5.16.1 patch, checks dependencies, and runs the small MPS tensor probe. A tensor probe alone does not verify SAM inference.

On the current machine, the selected Python base can be reused without specifying its private absolute path in documentation:

```sh
runtime/venv/bin/python -B app/scripts/setup.py --download-models
```

To rebuild after selecting a replacement base, stop the app and add `--rebuild-env` to the first command. The old project environment is preserved under a unique `runtime/venv.previous-*` name; it is not deleted. Setup does not overwrite an unrecognized nonempty environment directory. If installation fails, the previous environment remains available. Rebuilding requires internet access to obtain uncached wheels.

Model setup is separate from model selection. Only the official pinned SAM 2/SAM 2.1 files in [the portable manifest](../config/model-manifest.json) are downloaded. Each size and SHA256 is checked before atomic installation. The default Tiny/Small checkpoint payload is about 325 MiB, plus small JSON files; `--model all` installs all eight variants (about 2.91 GiB). ExFAT allocation and dependency caches need more actual disk space. Earlier native `.pt` downloads from investigation are inactive and are not redistributed.

```sh
runtime/venv/bin/python -B app/scripts/download_models.py
runtime/venv/bin/python -B app/scripts/download_models.py --verify-only
runtime/venv/bin/python -B app/scripts/smoke_models.py
```

The verification-only command uses no network and writes nothing. The smoke script performs actual image inference and saves evidence under `setup-notes`. The supplied SAM 3.1 checkpoint has passed experimental local image text inference; see [SAM3_ACCESS.md](SAM3_ACCESS.md). No further token or access approval is needed for that local file.

## Build the interface

Normal use needs the built `frontend/dist` assets, not Node.js. Development/build setup uses native Node.js 24.19.0 and npm 12.0.2 on this machine. Node is an existing bundled runtime; npm and its cache live under `runtime`. With your selected Node/npm on the current terminal's `PATH`, from `app/frontend`:

```sh
npm ci --no-bin-links --ignore-scripts --cache ../../runtime/npm-cache
npm run build
```

`package-lock.json` pins the frontend dependency tree. No global npm install or shell-profile change is needed. The generated UI uses local assets and no CDN. Browser inspection and screenshots are recorded separately in the validation report; a successful build alone is insufficient verification.

## Recovery and fresh-terminal checks

If the SSD disconnects, stop new work and reconnect the original drive. The application must not create substitute folders or report unsuccessful saves as persisted. Keep unsaved browser state until reconnection. Source identity checks prevent silently substituting another same-named file. To relocate the entire dedicated root onto another authorized SSD later, preserve the directory layout and establish its new local identity deliberately; do not merely overwrite the old identity while work is running.

After setup, test a new terminal session without inherited Python activation:

```sh
"./Start Studio.command"
"./Stop Studio.command"
```

Use `runtime/venv/bin/python -B app/scripts/launch.py start --no-open` from the dedicated root for automation that must verify readiness without opening a browser. Fresh-environment startup and shutdown, browser use, and offline inference passed. Exact evidence and the remaining verification limits are in [VALIDATION.md](VALIDATION.md).
