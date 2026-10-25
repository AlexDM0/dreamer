#!/usr/bin/env bash
# Dreamer setup for Linux / WSL / development: creates .venv and installs everything into it.
#   ./setup.sh                full GPU install (Python 3.10, CUDA PyTorch, StreamDiffusion, TensorRT)
#   ./setup.sh --mock         server only (mock backend, no torch, no GPU)
#   ./setup.sh --force        delete and recreate .venv first
#   ./setup.sh --python /usr/bin/python3.10
#   ./setup.sh --cuda cu128   PyTorch wheel index tag (https://download.pytorch.org/whl/<tag>)
# Same steps as setup.ps1 (Windows); the manual equivalent is requirements-gpu.md.
set -euo pipefail

# Pinned to match requirements-gpu.md.
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHAUDIO_VERSION="2.8.0"
STREAMDIFFUSION_REPO="https://github.com/daydreamlive/StreamDiffusion.git"
# The fork imports diffusers_ipadapter on every TensorRT path, but only its `ipadapter`
# extra installs it (and that extra also pulls insightface, which needs a C++ compiler).
# Install just this small package, at the commit the fork's setup.py pins.
DIFFUSERS_IPADAPTER="diffusers-ipadapter @ git+https://github.com/livepeer/Diffusers_IPAdapter.git@405f87da42932e30bd55ee8dca3ce502d7834a99"

MOCK=0
FORCE=0
PYTHON=""
CUDA="cu128"
SD_REF="4c90d9e437aa28cca7cae1acfab1e52157261939"   # git ref of the StreamDiffusion fork (tested on Windows + RTX 3080)

usage() {
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    echo "  --streamdiffusion-ref REF   git ref of the StreamDiffusion fork (default: the tested commit)"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --mock) MOCK=1 ;;
        --force) FORCE=1 ;;
        --python) PYTHON="${2:?--python needs a value}"; shift ;;
        --python=*) PYTHON="${1#*=}" ;;
        --cuda) CUDA="${2:?--cuda needs a value}"; shift ;;
        --cuda=*) CUDA="${1#*=}" ;;
        --streamdiffusion-ref) SD_REF="${2:?--streamdiffusion-ref needs a value}"; shift ;;
        --streamdiffusion-ref=*) SD_REF="${1#*=}" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv"
VENV_PY="$VENV/bin/python"

step() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m  %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# Runs a command and stops with a clear message if it fails.
run() {
    local title="$1"; shift
    step "$title"
    "$@" || die "$title failed (exit code $?). Fix the problem above and run ./setup.sh again."
}

py_version() {
    "$@" -c 'import platform; print(platform.python_version())' 2>/dev/null | tail -n 1
}

# Python 3.10 is required for the GPU stack; --mock also accepts a newer Python 3.
py_ok() {
    case "$1" in
        3.10.*) return 0 ;;
    esac
    if [ "$MOCK" = 1 ]; then
        local minor
        minor="$(echo "$1" | cut -d. -f2)"
        case "$1" in 3.*) [ "${minor:-0}" -ge 10 ] 2>/dev/null && return 0 ;; esac
    fi
    return 1
}

NEED="3.10"
if [ "$MOCK" = 1 ]; then NEED="3.10 or newer"; fi

printf '\033[32mDreamer setup\033[0m\n'
if [ "$MOCK" = 1 ]; then
    echo "Mode: mock (server only, no torch or GPU stack)"
else
    echo "Mode: GPU (CUDA tag $CUDA, StreamDiffusion @ $SD_REF)"
fi

# ---------------------------------------------------------------- checks
step "Checking Python $NEED"
PY=""
if [ -n "$PYTHON" ]; then
    candidates=("$PYTHON")
else
    candidates=(python3.10 python3 python)
fi
for cand in "${candidates[@]}"; do
    command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ] || continue
    v="$(py_version "$cand" || true)"
    if py_ok "$v"; then
        PY="$cand"; PY_VERSION="$v"; break
    fi
    if [ -n "$v" ]; then warn "$cand is Python $v, skipping."; fi
done
if [ -z "$PY" ]; then
    die "Python $NEED was not found. Install it (Ubuntu: sudo apt install python3.10 python3.10-venv,
       for example from the deadsnakes PPA) and run ./setup.sh again, or point to it with --python PATH."
fi
echo "  Using $PY (Python $PY_VERSION)"

if [ "$MOCK" = 0 ]; then
    step "Checking NVIDIA driver"
    if command -v nvidia-smi >/dev/null 2>&1; then
        line="$(nvidia-smi 2>/dev/null | grep -m1 'CUDA Version' || true)"
        echo "  ${line:-nvidia-smi found.}" | sed 's/  */ /g'
    else
        warn "WARNING: nvidia-smi not found. Install the NVIDIA driver (CUDA 12.8 or newer; on WSL install"
        warn "the Windows driver, not a Linux one), or use ./setup.sh --mock for a GPU-free setup."
    fi

    step "Checking Git"
    command -v git >/dev/null 2>&1 \
        || die "Git was not found; pip needs it to install StreamDiffusion from GitHub (sudo apt install git)."
    echo "  $(git --version)"
fi

# ---------------------------------------------------------------- venv
if [ "$FORCE" = 1 ] && [ -d "$VENV" ]; then
    step "Removing existing .venv (--force)"
    rm -rf "$VENV"
fi

if [ -x "$VENV_PY" ]; then
    VENV_VERSION="$(py_version "$VENV_PY" || true)"
    py_ok "$VENV_VERSION" \
        || die "The existing .venv uses Python ${VENV_VERSION:-unknown}, but this setup needs Python $NEED. Run ./setup.sh --force to recreate it."
    step "Using existing .venv (Python $VENV_VERSION)"
else
    step "Creating .venv"
    "$PY" -m venv "$VENV" \
        || die "Creating .venv failed. On Debian/Ubuntu install the venv module (sudo apt install python3-venv or python3.10-venv) and run ./setup.sh again."
    [ -x "$VENV_PY" ] || die "Creating .venv did not produce $VENV_PY."
fi

run "Upgrading pip, wheel and setuptools" \
    "$VENV_PY" -m pip install --upgrade pip wheel setuptools

run "Installing server requirements (requirements.txt)" \
    "$VENV_PY" -m pip install -r "$ROOT/requirements.txt"

# ---------------------------------------------------------------- GPU stack
if [ "$MOCK" = 0 ]; then
    INDEX_URL="https://download.pytorch.org/whl/$CUDA"
    run "Installing PyTorch $TORCH_VERSION ($CUDA)" \
        "$VENV_PY" -m pip install \
        "torch==$TORCH_VERSION+$CUDA" \
        "torchvision==$TORCHVISION_VERSION+$CUDA" \
        "torchaudio==$TORCHAUDIO_VERSION+$CUDA" \
        --index-url "$INDEX_URL"

    run "Installing StreamDiffusion (Daydream fork, tensorrt + controlnet)" \
        "$VENV_PY" -m pip install "streamdiffusion[tensorrt,controlnet] @ git+$STREAMDIFFUSION_REPO@$SD_REF"

    run "Installing diffusers-ipadapter (imported by StreamDiffusion's TensorRT code)" \
        "$VENV_PY" -m pip install "$DIFFUSERS_IPADAPTER"

    run "Installing TensorRT (streamdiffusion.tools.install-tensorrt)" \
        "$VENV_PY" -m streamdiffusion.tools.install-tensorrt

    # install-tensorrt pins an older cuDNN than the one Linux torch wheels are built
    # against (torch 2.8.0: 9.10.2.21 vs 9.7.1.26), and torch refuses an older cuDNN at
    # its first convolution. Put back the version torch asks for (TensorRT 10 is fine
    # with it). Windows torch wheels bundle their own cuDNN, so setup.ps1 skips this.
    CUDNN_REQ="$("$VENV_PY" -c "import importlib.metadata as m; \
print(next((r.split(';')[0].strip() for r in (m.requires('torch') or []) if r.startswith('nvidia-cudnn')), ''))")"
    if [ -n "$CUDNN_REQ" ]; then
        run "Restoring the cuDNN version PyTorch needs ($CUDNN_REQ)" \
            "$VENV_PY" -m pip install "$CUDNN_REQ"
    fi

    step "Checking that PyTorch can see the GPU"
    "$VENV_PY" -c "import sys, torch; ok = torch.cuda.is_available(); \
print(torch.__version__, 'CUDA', torch.version.cuda, 'available:', ok); \
print('GPU:', torch.cuda.get_device_name(0)) if ok else None; sys.exit(0 if ok else 1)" \
        || die "PyTorch cannot use the GPU (torch.cuda.is_available() is False). Update the NVIDIA driver, then reinstall the CUDA build with:
       .venv/bin/python -m pip install --force-reinstall torch==$TORCH_VERSION+$CUDA torchvision==$TORCHVISION_VERSION+$CUDA torchaudio==$TORCHAUDIO_VERSION+$CUDA --index-url $INDEX_URL"

    step "Checking cuDNN and StreamDiffusion's TensorRT imports"
    "$VENV_PY" -c "import torch, torch.nn.functional as F; \
print('cuDNN', torch.backends.cudnn.version()); \
x = torch.ones(1, 3, 8, 8, device='cuda'); F.conv2d(x, torch.ones(4, 3, 3, 3, device='cuda')); \
torch.cuda.synchronize(); print('GPU convolution ok')" \
        || die "PyTorch cannot run a convolution on the GPU (often a cuDNN version mismatch). Reinstall the cuDNN that torch needs:
       .venv/bin/python -m pip install \"$CUDNN_REQ\""
    "$VENV_PY" -c "import tensorrt, streamdiffusion.acceleration.tensorrt.models.controlnet_models, streamdiffusion.acceleration.tensorrt.export_wrappers.unet_unified_export; \
print('TensorRT', tensorrt.__version__, '- StreamDiffusion TensorRT imports ok')" \
        || die "StreamDiffusion's TensorRT code cannot be imported (see the error above). A missing
       diffusers_ipadapter is fixed with: .venv/bin/python -m pip install \"$DIFFUSERS_IPADAPTER\""
fi

# ---------------------------------------------------------------- config
if [ ! -f "$ROOT/config.yaml" ]; then
    step "Creating config.yaml from config.example.yaml"
    if [ "$MOCK" = 1 ]; then
        sed 's/^backend:[[:blank:]]*[^[:blank:]]*/backend: mock/' "$ROOT/config.example.yaml" > "$ROOT/config.yaml"
    else
        cp "$ROOT/config.example.yaml" "$ROOT/config.yaml"
    fi
else
    step "Keeping existing config.yaml"
fi
if grep -Eq '^admin_pin:[[:blank:]]*"?1234"?[[:blank:]]*(#.*)?$' "$ROOT/config.yaml"; then
    warn "Remember to change admin_pin in config.yaml (it is still the example PIN 1234)."
fi
if [ "$MOCK" = 1 ] && ! grep -Eq '^backend:[[:blank:]]*mock\b' "$ROOT/config.yaml"; then
    warn "Note: config.yaml does not say 'backend: mock'; set it to run without a GPU."
fi

# ---------------------------------------------------------------- done
printf '\n\033[32mSetup complete.\033[0m\n\n'
echo "Next steps:"
echo "  1. Set admin_pin in config.yaml (in quotes, 4 to 12 digits)."
echo "  2. Start Dreamer:   ./run.sh"
if [ "$MOCK" = 0 ]; then
    echo "     The first start builds TensorRT engines into engines/ (10 to 30 minutes, once)."
fi
echo "  3. Open  http://<this-machine-ip>:7860/  (kid page) and  http://<this-machine-ip>:7860/admin"
echo "     (run.sh prints the exact addresses). Tests: .venv/bin/python -m pytest -q"
