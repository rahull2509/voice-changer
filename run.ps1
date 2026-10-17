# VaniFlow PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "     VaniFlow - Live Real-Time AI Voice Changer" -ForegroundColor Yellow
Write-Host "     Windows 10 CPU - Python 3.12" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Verify Python
try {
    $pyVer = python --version
    Write-Host "[INFO] Detected Python: $pyVer" -ForegroundColor Green
} catch {
    Write-Error "Python is not found in PATH! Please install Python 3.12 64-bit."
    exit 1
}

# Verify packages
Write-Host "[INFO] Checking Python dependencies..." -ForegroundColor Gray
python -c "import sounddevice, pyworld, torch, fastapi, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host "[INFO] Starting VaniFlow Voice Changer server..." -ForegroundColor Green
Write-Host "[INFO] Open your browser at: http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host ""

python -m app.main
