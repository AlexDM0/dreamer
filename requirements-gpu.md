# Dreamer: GPU backend install (Windows, NVIDIA RTX)

These steps set up the `streamdiffusion` backend on the Windows PC (tested target: RTX 3080).
Without them, Dreamer still runs with `backend: mock` in `config.yaml`.

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
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## 3. PyTorch with CUDA 12.8

StreamDiffusion needs PyTorch 2.2 or newer. Install the CUDA build *before* StreamDiffusion
so pip does not pull a CPU-only torch:

```powershell
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last line must print `True`.

## 4. StreamDiffusion (Daydream fork) with TensorRT and ControlNet

```powershell
pip install "streamdiffusion[tensorrt,controlnet] @ git+https://github.com/daydreamlive/StreamDiffusion.git@main"
python -m streamdiffusion.tools.install-tensorrt
```

This is the PEP 508 form of the install; the older `...#egg=streamdiffusion[tensorrt,controlnet]`
form from the upstream README is deprecated in current pip and can silently drop the extras
that the TensorRT engine build needs. For reproducible installs, replace `@main` with a
commit SHA you have tested.

On CUDA 12 the second command installs `tensorrt==10.12.0.36` and
`nvidia-cudnn-cu12==9.7.1.26`; on Windows it also installs `pywin32` and `triton-windows`.

Admin-uploaded style pictures are screened with the wrapper's own safety checker (loaded
when `model.safety_checker: true`). If no checker is loaded, uploads are allowed and a
warning is logged; if the checker fails, the upload is refused.

## 5. Configure and first run

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

Start Dreamer (`.\run.ps1` or `python -m server`). **The first start builds TensorRT
engines** into `engines\`: this is slow (often 10 to 30 minutes) and happens only once per
model, resolution and ControlNet combination. The kid page shows "busy" until it is done.
Later starts load the cached engines in under a minute.

The model files download from Hugging Face on first run (a few GB, cached in
`%USERPROFILE%\.cache\huggingface`).

## Troubleshooting

- `RuntimeError: StreamDiffusion is not installed`: step 4 did not complete in the active venv.
- `torch.cuda.is_available()` is `False`: a CPU torch got installed; rerun step 3 with `--force-reinstall`.
- Engine build errors after changing `size`, `t_index_list` length, or the ControlNet model:
  delete the `engines\` folder and start again to rebuild.
- Changing the number of steps on the admin page can exceed the engine's batch range; the
  change is reverted and an error is shown. Rebuild engines (as above) for a new step count.
