# Dreamer launcher for Windows. Run from PowerShell:  .\run.ps1   (or: .\run.ps1 -Config my.yaml)
# Uses only the interpreter in .venv; create it first with .\setup.ps1
param(
    [string]$Config = "config.yaml"
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "No .venv found: run .\setup.ps1 first (or .\setup.ps1 -Mock for a GPU-free setup)." -ForegroundColor Yellow
    exit 1
}

$port = 7860
if (Test-Path -LiteralPath $Config) {
    $configText = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $Config).ProviderPath)
    $m = [regex]::Match($configText, '(?m)^port:\s*(\d+)')
    if ($m.Success) { $port = [int]$m.Groups[1].Value }
} else {
    Write-Host "No $Config found; using defaults. Run .\setup.ps1 to create config.yaml, then set admin_pin." -ForegroundColor Yellow
}

# LAN addresses for the iPad and the parent's phone (best effort).
$ips = @()
try {
    $ips = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
        Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.IPAddress -notlike "169.254.*" } |
        ForEach-Object { $_.IPAddress })
} catch {
    try {
        $ips = @([System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
            Where-Object { $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork -and
                -not [System.Net.IPAddress]::IsLoopback($_) } |
            ForEach-Object { $_.ToString() })
    } catch {
        $ips = @()
    }
}
Write-Host ""
Write-Host "Dreamer is starting. Open on the iPad / phone:" -ForegroundColor Green
if ($ips.Count -eq 0) {
    Write-Host "  kid page:  http://<pc-ip>:$port/"
    Write-Host "  admin:     http://<pc-ip>:$port/admin"
}
foreach ($ip in $ips) {
    Write-Host "  kid page:  http://${ip}:$port/      admin: http://${ip}:$port/admin"
}
Write-Host "Press Ctrl+C to stop."
Write-Host ""

$env:SKETCH_CONFIG = $Config
# The server logs to stderr. With output redirected (a log file, a scheduled task), Windows
# PowerShell 5.1 turns each stderr line into an error record, and "Stop" would kill the
# server on its first log line; the exit code decides success instead.
$ErrorActionPreference = "Continue"
# Exit code 75: the GPU crashed and Dreamer asked to be started again (a crashed CUDA context
# can't recover inside the process). Restart a few times, then give up.
$restarts = 0
while ($true) {
    & $python -m server
    $code = $LASTEXITCODE
    if ($code -ne 75 -or $restarts -ge 5) { exit $code }
    $restarts++
    Write-Host "The graphics card crashed; restarting Dreamer ($restarts of 5)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}
