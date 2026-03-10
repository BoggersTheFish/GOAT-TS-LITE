# GOAT-TS Lite — Windows bootstrap (PowerShell)
# Run from project root: .\run_windows.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

$venvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$venvPip = Join-Path $ProjectRoot "venv\Scripts\pip.exe"

# Create venv if missing
if (-not (Test-Path $venvPython)) {
    Write-Host "[GOAT-TS Lite] Creating venv..."
    python -m venv venv
}

# Install into venv using venv's Python (avoids "user installation" and wrong interpreter)
Write-Host "[GOAT-TS Lite] Installing dependencies into venv..."
& $venvPython -m pip install --upgrade pip -q
& $venvPython -m pip install numpy pandas networkx pyyaml streamlit -q

# Optional: CPU PyTorch if not already present
$torchOk = & $venvPython -c "import torch; exit(0)" 2>$null
if (-not $LASTEXITCODE -eq 0) {
    Write-Host "[GOAT-TS Lite] Installing CPU PyTorch (may take a minute)..."
    & $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
}

Write-Host "[GOAT-TS Lite] Running reasoning loop (dry-run)..."
& $venvPython reasoning/demo_loop.py --ticks 3 --dry-run

Write-Host ""
Write-Host "[GOAT-TS Lite] Starting Streamlit UI..."
& $venvPython -m streamlit run streamlit_app.py
