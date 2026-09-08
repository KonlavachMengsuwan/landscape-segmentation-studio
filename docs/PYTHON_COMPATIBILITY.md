# Python versions

The current release supports **standard CPython 3.12**. It was tested with native ARM64 Python **3.12.14** on a Mac mini M4. The setup scripts deliberately select and check Python 3.12; they do not currently accept Python 3.13 or 3.14. Windows remains a test release, including when Python 3.12 is used.

Python 3.13 or 3.14 support appears feasible from the upstream package metadata reviewed on **8 September 2026**, but it has not been verified by running this application under those interpreters. Do not describe this release as supporting “Python 3.12+.”

## If another Python version is already installed

Keep it. Install Python 3.12 alongside it for this application; there is no need to uninstall your other Python version or change another project's environment. Studio creates its own environment in `runtime/venv/` and installs packages there.

On Windows, check the interpreter selected by the setup launcher:

```bat
py -3.12 --version
```

On an Apple Silicon Mac, check the native interpreter used by its setup launcher:

```sh
python3.12 --version
python3.12 -c "import platform; print(platform.machine())"
```

The second Mac command should print `arm64`. Use x64 Python on Windows. Studio's current release does not support Intel Macs, Windows ARM64, Rosetta Python, PyPy, or free-threaded Python builds. Follow the complete [Mac setup](MAC_SETUP.md) or [Windows setup](WINDOWS_SETUP.md) instructions after selecting the correct interpreter.

Python's [Windows documentation](https://docs.python.org/3/using/windows.html) describes managing multiple Python versions. Its [virtual environment documentation](https://docs.python.org/3/library/venv.html) explains project environments and their relationship to a base interpreter. A virtual environment is not a portable installation: do not send your existing `runtime/venv/` to another computer or assume that copying it replaces installing Python.

## What was checked for newer versions

The review queried the [PyPI version JSON API](https://docs.pypi.org/api/json/) for every exact pin in `requirements.lock`, checking `Requires-Python`, non-yanked wheel files, and compatible Python/ABI/platform tags. It checked standard, GIL-enabled CPython and native macOS ARM64 or Windows x64. The macOS wheel-tag simulation used macOS 26.0; it does not establish an older operating-system minimum for the complete app.

| Python version | Mac ARM64 wheel candidates | Windows x64 wheel candidates | Current application status |
|---|---|---|---|
| 3.12 | All 49 pinned packages | All 49 pinned packages | M4 application and real inference tested; Windows execution unverified |
| 3.13 | All 49 pinned packages | All 49 pinned packages | Metadata review only; setup does not accept it |
| 3.14, excluding 3.14.1 | All 49 pinned packages | All 49 pinned packages | Metadata review only; setup does not accept it |
| 3.14.1 | Rejected by a pinned dependency | Rejected by a pinned dependency | Do not use |

The 3.14 wheel check used version 3.14.0 to evaluate package requirements. **`torchvision==0.27.1` explicitly excludes Python 3.14.1** through its `Requires-Python: !=3.14.1,>=3.10` metadata. Other 3.14 patches still require application validation. The pinned [PyTorch 2.12.1](https://pypi.org/project/torch/2.12.1/), [torchvision 0.27.1](https://pypi.org/project/torchvision/0.27.1/), and [Transformers 5.16.1](https://pypi.org/project/transformers/5.16.1/) releases are the relevant versions; compatibility of a different latest release does not establish compatibility of this project's pins.

The official CUDA 13.0 indexes also list Windows x64 wheels for the pinned `torch==2.12.1+cu130` and `torchvision==0.27.1+cu130` builds for CPython 3.12, 3.13, and 3.14. That establishes binary availability, not driver compatibility, GPU memory fit, or tested NVIDIA inference. Sources: [PyTorch CUDA 13.0 wheels](https://download.pytorch.org/whl/cu130/torch/), [torchvision CUDA 13.0 wheels](https://download.pytorch.org/whl/cu130/torchvision/).

No alternate interpreter or wheel environment was installed for this review. No dependency resolution, imports, application tests, or model inference were run under Python 3.13 or 3.14. Wheel availability alone does not prove that the complete dependency graph installs successfully or that the application behaves correctly.

## A safe route to broader support and easier installation

Before adding a newer Python version to the setup scripts, create a separate development environment and validate binary-only installation, dependency resolution, the Python test suite, launch/save/export/restore, and real local inference for each supported model family. Test MPS on Apple Silicon and CPU/CUDA on actual Windows hardware separately. Preserve the working 3.12 environment and original images during this work.

For easier distribution, a future installer could manage a project-local Python runtime so recipients would not need to choose or install Python themselves. That would require its own reproducible runtime download, integrity checks, third-party notices, update strategy, and native Mac/Windows installation tests. This is a proposed improvement, **not a feature of the current ZIP setup packages**. For this release, selecting a tested Python 3.12 installation and keeping the dependency pins is the supported path.
