param([string]$PythonVersion = "3.12")
$ErrorActionPreference = "Stop"
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Previous command failed ($LASTEXITCODE)" } }
py "-$PythonVersion" -m venv .venv
Check-Exit
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
Check-Exit
# Frozen, mutually compatible PyTorch/torchvision pair. NVIDIA driver must support CUDA 12.8.
& $Python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
Check-Exit
& $Python -m pip install -e ".[train,dev]"
Check-Exit
& $Python -m spinelocus doctor
Check-Exit
Write-Host "Ready. Run .\.venv\Scripts\python.exe -m spinelocus --help"
