# Addendum: venv + setup script (Alex, 2026-10-25)

_Updated 2026-10-26 to match the code; docs/setup.md is the user guide._ The original request was the setup script below; the install list grew as the GPU stack was tested, and `setup.ps1` / `setup.sh` are the source of truth.

"Anything python in a venv and a setup script for dependencies."

- All Python runs from `.venv` in the repo root. Nothing installs into the system Python.
- `setup.ps1` (Windows PowerShell 5.1 compatible, also fine in pwsh 7):
  - Params: `-Mock` (server deps only, no torch/GPU stack; also accepts Python 3.10 or newer), `-Python <path or py-launcher spec>` (default: `py -3.10`, fall back to `python` if it reports 3.10.x), `-Cuda cu128` (torch wheel index tag), `-Force` (recreate .venv), `-StreamDiffusionRef <git ref>` (the fork commit to install; default is the commit tested on Windows + RTX 3080, 4c90d9e437aa28cca7cae1acfab1e52157261939). `setup.sh` has the same as `--mock --python --cuda --force --streamdiffusion-ref`.
  - Checks: Python 3.10 present (error with install hint if not), `nvidia-smi` present unless -Mock (warn with driver hint), Git present (needed for the pip git+ install).
  - Creates `.venv` if missing (`<python> -m venv .venv`), upgrades pip/wheel/setuptools inside it.
  - Installs `requirements.txt` (server deps, no torch).
  - Unless -Mock: installs torch/torchvision/torchaudio for the chosen CUDA tag from https://download.pytorch.org/whl/<tag> (pinned versions as in requirements-gpu.md: torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128), then, in this order:
    1. the Daydream StreamDiffusion fork at the pinned ref with `[tensorrt,controlnet]` extras;
    2. `diffusers-ipadapter` from the livepeer fork at a pinned commit (the fork imports it on every TensorRT path, but its `ipadapter` extra would also pull insightface, which needs a C++ compiler);
    3. `peft==0.17.1` with `--no-deps` (loads the TCD LoRA for the "SD 1.5 + TCD" picture model without touching the fork's torch/transformers pins);
    4. `python -m streamdiffusion.tools.install-tensorrt`; on Linux (`setup.sh`) it then restores the cuDNN version torch needs, because install-tensorrt pins an older one (Windows wheels bundle their own);
    5. downloads the upscaler model `models/realesr-animevideov3.pth` (Real-ESRGAN compact, 2.5 MB; optional: a failed download only leaves AI upscale unavailable);
    6. verifies `torch.cuda.is_available()` and prints the GPU name, then runs a GPU convolution (cuDNN check) and imports StreamDiffusion's TensorRT modules.
  - Copies `config.example.yaml` to `config.yaml` if missing (and sets backend: mock when -Mock); warns while `admin_pin` is still the example `1234`.
  - Stops on first failure (`$ErrorActionPreference = 'Stop'` plus checking `$LASTEXITCODE` after every native command), prints clear next steps at the end (set `admin_pin`, `.\run.ps1`, open http://<ip>:7860 and /admin, the firewall rule command; the first GPU start builds TensorRT engines into `engines/`, 10 to 30 minutes, once).
- `run.ps1` uses `.venv\Scripts\python.exe -m server` only; errors with "run .\setup.ps1 first" if .venv is missing. Prints LAN URLs (kid page, admin) using the machine's IPv4 addresses. `Dreamer.cmd` (double-click) runs `run.ps1` and pauses if it fails.
- Also provide `setup.sh` / `run.sh` equivalents for Linux/WSL/dev (same flags: --mock etc.) since development and tests here run on Linux.
- The quick start (README, docs/setup.md) uses only these scripts. requirements-gpu.md stays as the manual reference that matches what setup.ps1 does; `requirements.txt` holds the server and test dependencies (no torch).
- Tests/e2e scripts must use the venv interpreter (`.venv/bin/python` on Linux).
