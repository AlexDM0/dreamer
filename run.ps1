# Dreamer launcher for Windows. Run from PowerShell:  .\run.ps1   (or: .\run.ps1 -Config my.yaml)
param(
    [string]$Config = "config.yaml"
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv found. Create it first (see README):" -ForegroundColor Yellow
    Write-Host "  py -3.10 -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path $Config)) {
    Write-Host "No $Config found; copy config.example.yaml to $Config and set admin_pin." -ForegroundColor Yellow
}

$env:SKETCH_CONFIG = $Config
& $python -m server
exit $LASTEXITCODE
