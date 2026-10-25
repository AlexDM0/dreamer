# Addendum: venv + setup script (Alex, 2026-10-25)

"Anything python in a venv and a setup script for dependencies."

- All Python runs from `.venv` in the repo root. Nothing installs into the system Python.
- `setup.ps1` (Windows PowerShell 5.1 compatible, also fine in pwsh 7):
  - Params: `-Mock` (server deps only, no torch/GPU stack), `-Python <path or py-launcher spec>` (default: `py -3.10`, fall back to `python` if it reports 3.10.x), `-Cuda cu128` (torch wheel index tag), `-Force` (recreate .venv).
  - Checks: Python 3.10 present (error with install hint if not), `nvidia-smi` present unless -Mock (warn with driver hint), Git present (needed for the pip git+ install).
  - Creates `.venv` if missing (`<python> -m venv .venv`), upgrades pip/wheel/setuptools inside it.
  - Installs `requirements.txt` (server deps, no torch).
  - Unless -Mock: installs torch/torchvision/torchaudio for the chosen CUDA tag from https://download.pytorch.org/whl/<tag> (pinned versions as in requirements-gpu.md: torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128), then the Daydream StreamDiffusion fork with `[tensorrt,controlnet]` extras, then `python -m streamdiffusion.tools.install-tensorrt`, then verifies `import torch; torch.cuda.is_available()` and prints the GPU name.
  - Copies `config.example.yaml` to `config.yaml` if missing (and sets backend: mock when -Mock).
  - Stops on first failure (`$ErrorActionPreference = 'Stop'` plus checking `$LASTEXITCODE` after every native command), prints clear next steps at the end (`.\run.ps1`, open http://<ip>:7860 and /admin, firewall rule command).
- `run.ps1` uses `.venv\Scripts\python.exe -m server` only; errors with "run .\setup.ps1 first" if .venv is missing. Prints LAN URLs (kid page, admin) using the machine's IPv4 addresses.
- Also provide `setup.sh` / `run.sh` equivalents for Linux/WSL/dev (same flags: --mock etc.) since development and tests here run on Linux.
- README quick start uses only these scripts. requirements-gpu.md stays as the manual reference that matches what setup.ps1 does.
- Tests/e2e scripts must use the venv interpreter (`.venv/bin/python` on Linux).
