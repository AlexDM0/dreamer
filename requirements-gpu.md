# Dreamer: GPU backend install (Windows, NVIDIA RTX)

These steps set up the `streamdiffusion` backend on the Windows PC (tested target: RTX 3080).
Without them, Dreamer still runs with `backend: mock` in `config.yaml`.

**`.\setup.ps1` runs all of these steps for you** (and `./setup.sh` does the same on Linux/WSL).
This page is the manual reference: it lists exactly what the script installs, in the same
order and with the same pinned versions, for when you want to do it by hand or a step fails.
`setup.ps1 -Cuda <tag>` swaps `cu128` below for another PyTorch wheel tag, and
`-StreamDiffusionRef <ref>` installs another commit or branch instead of the tested one below.

## 1. Prerequisites

- An up-to-date NVIDIA driver that supports CUDA 12.8 (`nvidia-smi` shows "CUDA Version: 12.8" or higher).
- **Python 3.10** (64-bit) from python.org. The StreamDiffusion README asks for 3.10.
- Git for Windows (`git --version` should work in PowerShell; pip needs it to install from GitHub).
- About 15 GB of free disk space for models, TensorRT engines and the CUDA wheels.

## 2. Virtual environment

From the Dreamer folder in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## 3. PyTorch with CUDA 12.8

StreamDiffusion needs PyTorch 2.2 or newer. Install the CUDA build *before* StreamDiffusion
so pip does not pull a CPU-only torch:

```powershell
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

The GPU check comes at the end of step 4, because installing StreamDiffusion can change packages.

## 4. StreamDiffusion (Daydream fork) with TensorRT and ControlNet

```powershell
pip install "streamdiffusion[tensorrt,controlnet] @ git+https://github.com/daydreamlive/StreamDiffusion.git@4c90d9e437aa28cca7cae1acfab1e52157261939"
pip install "diffusers-ipadapter @ git+https://github.com/livepeer/Diffusers_IPAdapter.git@405f87da42932e30bd55ee8dca3ce502d7834a99"
python -m streamdiffusion.tools.install-tensorrt
```

The second line is needed even without IP-Adapter: the fork's TensorRT code imports
`diffusers_ipadapter` on every engine build, but only its `ipadapter` extra installs it.
Do not use that extra on Windows: it also pulls `insightface`, which has no Windows wheel
and needs the MSVC Build Tools to compile. Without the package the first start fails with
"Acceleration has failed." (`No module named 'diffusers_ipadapter'` further up the log).

This is the PEP 508 form of the install; the older `...#egg=streamdiffusion[tensorrt,controlnet]`
form from the upstream README is deprecated in current pip and can silently drop the extras
that the TensorRT engine build needs. The SHA is the fork commit tested on Windows with an
RTX 3080 (2026-10-25); to try a newer one, use `@main` (`.\setup.ps1 -StreamDiffusionRef main`).

On CUDA 12 the second command installs `tensorrt==10.12.0.36` and
`nvidia-cudnn-cu12==9.7.1.26`; on Windows it also installs `pywin32` and `triton-windows`.

**Linux / WSL only:** that cuDNN is older than the 9.10.2.21 the Linux torch 2.8.0 wheel is
built against, and torch then fails at its first convolution ("cuDNN version
incompatibility"). Put torch's version back afterwards (setup.sh does this; Windows torch
wheels bundle their own cuDNN and are not affected):

```bash
pip install nvidia-cudnn-cu12==9.10.2.21
```

Admin-uploaded style pictures are screened with the wrapper's own safety checker (loaded
when `model.safety_checker: true`). If no checker is loaded, uploads are allowed and a
warning is logged; if the checker fails, the upload is refused.

Then check that PyTorch still sees the GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It must print `True` and your GPU name. Then check cuDNN and StreamDiffusion's TensorRT
imports (setup.ps1 / setup.sh run the same check):

```powershell
python -c "import torch, torch.nn.functional as F; F.conv2d(torch.ones(1,3,8,8,device='cuda'), torch.ones(4,3,3,3,device='cuda')); import tensorrt, streamdiffusion.acceleration.tensorrt.models.controlnet_models, streamdiffusion.acceleration.tensorrt.export_wrappers.unet_unified_export; print('ok')"
```

## 5. Configure and first run

Copy `config.example.yaml` to `config.yaml` if it does not exist yet (setup.ps1 does this).
In `config.yaml`:

```yaml
backend: streamdiffusion
model:
  base: stabilityai/sd-turbo
  controlnet: thibaud/controlnet-sd21-scribble-diffusers
  line_polarity: white_on_black
  engine_dir: engines
```

SD-Turbo is based on SD 2.1, so only **SD 2.1** ControlNets work with it
(for example `thibaud/controlnet-sd21-scribble-diffusers`). SD 1.5 or SDXL ControlNets will fail.
Set `controlnet: null` to turn ControlNet off.

Start Dreamer with `.\run.ps1` (it runs `.venv\Scripts\python.exe -m server`). **The first start builds TensorRT
engines** into `engines\`: this is slow (often 10 to 30 minutes) and happens only once per
model, resolution and ControlNet combination. The kid page shows "busy" until it is done.
Later starts load the cached engines in under a minute.

The model files download from Hugging Face on first run (a few GB, cached in
`%USERPROFILE%\.cache\huggingface`).

## Troubleshooting

- `RuntimeError: StreamDiffusion is not installed`: step 4 did not complete in the active venv.
- `torch.cuda.is_available()` is `False`: a CPU torch got installed; rerun step 3 with `--force-reinstall`.
- setup.ps1 says the existing `.venv` uses the wrong Python: run `.\setup.ps1 -Force` to recreate it.
- Engines are kept in `engines\plain\` (ControlNet off) and `engines\controlnet\` (ControlNet
  on). The fork names the UNet engine the same either way, so in one folder a UNet built
  without ControlNet would be reused with it and the drawing's lines would be ignored.
  Delete any old `engines\stabilityai\` folder left over from before this split.
- Engine build errors after changing `size`, `t_index_list` length, or the ControlNet model:
  delete the `engines\` folder and start again to rebuild.
- Steps: up to `model.max_batch_steps` (default 4; allowed 1, 2, 4 or 8, because the
  ControlNet engine does not build for 1-3) run together as one UNet batch; more, up to
  `model.max_steps` (default 8), and all refinement passes run one after another, so they
  need no extra GPU memory. Engine memory: batch 1 ~5.1 GB, 1-4 ~7.4 GB (plus the desktop). The UNet engine is built for batch 1 to `max_batch_steps` (about 8.4 GB in use on
  the RTX 3080), and all step changes apply live. Raising `max_batch_steps` makes 5+ steps
  faster but costs memory (batch 1-8 used ~9.6 GB) and rebuilds the engine once. If a change ever fails (for example after changing `size`), it is reverted and
  an error is shown; rebuild the engines as above.
