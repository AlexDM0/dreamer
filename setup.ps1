# Dreamer setup for Windows: creates .venv and installs everything into it.
# Run from PowerShell in the Dreamer folder:
#   .\setup.ps1              full GPU install (Python 3.10, CUDA PyTorch, StreamDiffusion, TensorRT)
#   .\setup.ps1 -Mock        server only (mock backend, no torch, no GPU)
#   .\setup.ps1 -Force       delete and recreate .venv first
#   .\setup.ps1 -Python "C:\Python310\python.exe"   or   -Python "py -3.10"
#   .\setup.ps1 -Cuda cu128  PyTorch wheel index tag (https://download.pytorch.org/whl/<tag>)
# Works in Windows PowerShell 5.1 and PowerShell 7. The manual equivalent is requirements-gpu.md.
param(
    [switch]$Mock,
    [string]$Python = "",
    [string]$Cuda = "cu128",
    [switch]$Force,
    # Git ref of the StreamDiffusion fork. Default: the commit tested on Windows + RTX 3080 (2026-10-25).
    [string]$StreamDiffusionRef = "4c90d9e437aa28cca7cae1acfab1e52157261939"
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Pinned to match requirements-gpu.md.
$TorchVersion = "2.8.0"
$TorchvisionVersion = "0.23.0"
$TorchaudioVersion = "2.8.0"
$StreamDiffusionRepo = "https://github.com/daydreamlive/StreamDiffusion.git"
# The fork imports diffusers_ipadapter on every TensorRT path, but only its `ipadapter`
# extra installs it, and that extra also pulls insightface, which needs the MSVC Build
# Tools on Windows. Install just this small package, at the commit the fork pins.
# AI upscaler for refined pictures (Real-ESRGAN compact, BSD-3-Clause), 2.5 MB.
$UpscalerUrl = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
$UpscalerPath = Join-Path $PSScriptRoot "models\realesr-animevideov3.pth"
$DiffusersIpAdapter = "diffusers-ipadapter @ git+https://github.com/livepeer/Diffusers_IPAdapter.git@405f87da42932e30bd55ee8dca3ce502d7834a99"

$VenvDir = Join-Path $PSScriptRoot ".venv"
$VenvScripts = Join-Path $VenvDir "Scripts"
$VenvPython = Join-Path $VenvScripts "python.exe"
$ConfigPath = Join-Path $PSScriptRoot "config.yaml"
$ConfigExample = Join-Path $PSScriptRoot "config.example.yaml"

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Stop-Setup([string]$Text) {
    Write-Host ""
    Write-Host "ERROR: $Text" -ForegroundColor Red
    exit 1
}

# Runs a native command and stops the script if it fails. $ErrorActionPreference is relaxed
# inside this function only, so tools that write progress to stderr (pip, git) do not abort
# Windows PowerShell 5.1 when output is redirected; the exit code decides success.
function Invoke-Native([string]$Title, [string]$Exe, [string[]]$Arguments) {
    Write-Step $Title
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Stop-Setup "$Title failed (exit code $code). Fix the problem above and run .\setup.ps1 again."
    }
}

# Returns "3.10.11"-style version for an interpreter, or $null if it cannot be run.
function Get-PythonVersion([string]$Exe, [string[]]$PreArgs) {
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Exe @PreArgs -c "import platform; print(platform.python_version())" 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $line = ($out | Select-Object -Last 1)
        if ($line -match '^\d+\.\d+\.\d+') { return $Matches[0] }
    } catch {
        return $null
    }
    return $null
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# Python 3.10 is required for the GPU stack; -Mock also accepts a newer Python 3.
function Test-PythonOk([string]$Version) {
    if (-not $Version) { return $false }
    if ($Version -match '^3\.10\.') { return $true }
    if ($Mock) {
        $minor = [int]($Version.Split('.')[1])
        if ($Version.StartsWith("3.") -and $minor -ge 10) { return $true }
    }
    return $false
}

$need = "3.10"
if ($Mock) { $need = "3.10 or newer" }

Write-Host "Dreamer setup" -ForegroundColor Green
if ($Mock) {
    Write-Host "Mode: mock (server only, no torch or GPU stack)"
} else {
    Write-Host "Mode: GPU (CUDA tag $Cuda, StreamDiffusion @ $StreamDiffusionRef)"
}

# ---------------------------------------------------------------- checks
Write-Step "Checking Python $need"
$pyExe = $null
$pyArgs = @()
$pyVersion = $null
$candidates = @()
if ($Python) {
    if (Test-Path -LiteralPath $Python) {
        $candidates += , @($Python)
    } else {
        $candidates += , @($Python.Trim() -split '\s+')
    }
} else {
    $candidates += , @("py", "-3.10")
    $candidates += , @("python")
}
foreach ($cand in $candidates) {
    $exe = $cand[0]
    $rest = @()
    if ($cand.Count -gt 1) { $rest = @($cand[1..($cand.Count - 1)]) }
    if (-not (Test-Path -LiteralPath $exe) -and -not (Test-Command $exe)) { continue }
    $v = Get-PythonVersion $exe $rest
    if (Test-PythonOk $v) {
        $pyExe = $exe
        $pyArgs = $rest
        $pyVersion = $v
        break
    }
    if ($v) { Write-Host "  $($cand -join ' ') is Python $v, skipping." -ForegroundColor Yellow }
}
if (-not $pyExe) {
    Stop-Setup ("Python $need was not found. Install Python 3.10 (64-bit) from " +
        "https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH' and keep " +
        "the 'py launcher' option), then run .\setup.ps1 again. Or point to it with " +
        "-Python 'C:\path\to\python.exe'.")
}
Write-Host "  Using $((@($pyExe) + $pyArgs) -join ' ') (Python $pyVersion)"

if (-not $Mock) {
    Write-Step "Checking NVIDIA driver"
    if (Test-Command "nvidia-smi") {
        $ErrorActionPreference = "Continue"
        $smi = & nvidia-smi 2>$null | Select-String -Pattern "CUDA Version" | Select-Object -First 1
        $ErrorActionPreference = "Stop"
        if ($smi) { Write-Host ("  " + $smi.Line.Trim()) } else { Write-Host "  nvidia-smi found." }
    } else {
        Write-Host ("  WARNING: nvidia-smi not found. Install the latest NVIDIA Game Ready or Studio " +
            "driver (CUDA 12.8 or newer) from https://www.nvidia.com/Download/index.aspx, " +
            "or use .\setup.ps1 -Mock for a GPU-free setup.") -ForegroundColor Yellow
    }

    Write-Step "Checking Git"
    if (-not (Test-Command "git")) {
        Stop-Setup ("Git was not found; pip needs it to install StreamDiffusion from GitHub. " +
            "Install Git for Windows from https://git-scm.com/download/win (or: winget install Git.Git), " +
            "open a new PowerShell window and run .\setup.ps1 again.")
    }
    $ErrorActionPreference = "Continue"
    $gitVersion = & git --version 2>$null
    $ErrorActionPreference = "Stop"
    Write-Host "  $gitVersion"
}

# ---------------------------------------------------------------- venv
if ($Force -and (Test-Path -LiteralPath $VenvDir)) {
    Write-Step "Removing existing .venv (-Force)"
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}

if (Test-Path -LiteralPath $VenvPython) {
    $venvVersion = Get-PythonVersion $VenvPython @()
    if (-not (Test-PythonOk $venvVersion)) {
        Stop-Setup ("The existing .venv uses Python $venvVersion, but this setup needs Python $need. " +
            "Run .\setup.ps1 -Force to recreate it.")
    }
    Write-Step "Using existing .venv (Python $venvVersion)"
} else {
    Invoke-Native "Creating .venv" $pyExe ($pyArgs + @("-m", "venv", $VenvDir))
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Stop-Setup "Creating .venv did not produce $VenvPython."
    }
}

Invoke-Native "Upgrading pip, wheel and setuptools" $VenvPython @(
    "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools")

Invoke-Native "Installing server requirements (requirements.txt)" $VenvPython @(
    "-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "requirements.txt"))

# ---------------------------------------------------------------- GPU stack
if (-not $Mock) {
    $indexUrl = "https://download.pytorch.org/whl/$Cuda"
    Invoke-Native "Installing PyTorch $TorchVersion ($Cuda)" $VenvPython @(
        "-m", "pip", "install",
        "torch==$TorchVersion+$Cuda",
        "torchvision==$TorchvisionVersion+$Cuda",
        "torchaudio==$TorchaudioVersion+$Cuda",
        "--index-url", $indexUrl)

    $sdSpec = "streamdiffusion[tensorrt,controlnet] @ git+$StreamDiffusionRepo@$StreamDiffusionRef"
    Invoke-Native "Installing StreamDiffusion (Daydream fork, tensorrt + controlnet)" $VenvPython @(
        "-m", "pip", "install", $sdSpec)

    Invoke-Native "Installing diffusers-ipadapter (imported by StreamDiffusion's TensorRT code)" $VenvPython @(
        "-m", "pip", "install", $DiffusersIpAdapter)

    Invoke-Native "Installing TensorRT (streamdiffusion.tools.install-tensorrt)" $VenvPython @(
        "-m", "streamdiffusion.tools.install-tensorrt")

    if (Test-Path -LiteralPath $UpscalerPath) {
        Write-Step "Upscaler model already present (models\realesr-animevideov3.pth)"
    } else {
        Write-Step "Downloading the upscaler model (realesr-animevideov3.pth, 2.5 MB)"
        New-Item -ItemType Directory -Force -Path (Split-Path $UpscalerPath) | Out-Null
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $UpscalerUrl -OutFile $UpscalerPath
        } catch {
            # Optional: Dreamer runs without it (the admin page shows the upscaler as unavailable).
            Write-Host "  Could not download it ($($_.Exception.Message)); AI upscale stays unavailable." -ForegroundColor Yellow
        }
    }

    Write-Step "Checking that PyTorch can see the GPU"
    $check = "import sys, torch; ok = torch.cuda.is_available(); " +
        "print(torch.__version__, 'CUDA', torch.version.cuda, 'available:', ok); " +
        "print('GPU:', torch.cuda.get_device_name(0)) if ok else None; sys.exit(0 if ok else 1)"
    $ErrorActionPreference = "Continue"
    & $VenvPython -c $check
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) {
        Stop-Setup ("PyTorch cannot use the GPU (torch.cuda.is_available() is False). Update the NVIDIA " +
            "driver, then reinstall the CUDA build with: .venv\Scripts\python.exe -m pip install " +
            "--force-reinstall torch==$TorchVersion+$Cuda torchvision==$TorchvisionVersion+$Cuda " +
            "torchaudio==$TorchaudioVersion+$Cuda --index-url $indexUrl")
    }

    Write-Step "Checking cuDNN and StreamDiffusion's TensorRT imports"
    $check = "import torch, torch.nn.functional as F; " +
        "print('cuDNN', torch.backends.cudnn.version()); " +
        "x = torch.ones(1, 3, 8, 8, device='cuda'); F.conv2d(x, torch.ones(4, 3, 3, 3, device='cuda')); " +
        "torch.cuda.synchronize(); print('GPU convolution ok'); " +
        "import tensorrt, streamdiffusion.acceleration.tensorrt.models.controlnet_models, streamdiffusion.acceleration.tensorrt.export_wrappers.unet_unified_export; " +
        "print('TensorRT', tensorrt.__version__, '- StreamDiffusion TensorRT imports ok')"
    $ErrorActionPreference = "Continue"
    & $VenvPython -c $check
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) {
        Stop-Setup ("The GPU stack is installed but does not work (see the error above). If it says " +
            "No module named 'diffusers_ipadapter', run: .venv\Scripts\python.exe -m pip install " +
            "`"$DiffusersIpAdapter`"")
    }
}

# ---------------------------------------------------------------- config
$utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Write-Step "Creating config.yaml from config.example.yaml"
    $text = [System.IO.File]::ReadAllText($ConfigExample)
    if ($Mock) {
        $text = [regex]::Replace($text, '(?m)^backend:[^\S\r\n]*\S+', 'backend: mock')
    }
    [System.IO.File]::WriteAllText($ConfigPath, $text, $utf8NoBom)
} else {
    Write-Step "Keeping existing config.yaml"
}
$configText = [System.IO.File]::ReadAllText($ConfigPath)
if ($configText -match '(?m)^admin_pin:\s*"?1234"?\s*(#.*)?$') {
    Write-Host "  Remember to change admin_pin in config.yaml (it is still the example PIN 1234)." -ForegroundColor Yellow
}
if ($Mock -and ($configText -notmatch '(?m)^backend:\s*mock\b')) {
    Write-Host "  Note: config.yaml does not say 'backend: mock'; set it to run without a GPU." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- done
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Set admin_pin in config.yaml (in quotes, 4 to 12 digits)."
Write-Host "  2. Start Dreamer:   .\run.ps1"
if (-not $Mock) {
    Write-Host "     The first start builds TensorRT engines into engines\ (10 to 30 minutes, once)."
}
Write-Host "  3. On the iPad open  http://<pc-ip>:7860/   and the parent page  http://<pc-ip>:7860/admin"
Write-Host "     (run.ps1 prints the exact addresses)."
Write-Host "  4. Allow the port through Windows Firewall once, from an administrator PowerShell:"
Write-Host '     New-NetFirewallRule -DisplayName "Dreamer" -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow -Profile Private'
exit 0
