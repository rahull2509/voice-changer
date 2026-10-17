@echo off
setlocal
echo ========================================================
echo      VaniFlow - Live Real-Time AI Voice Changer
echo      Windows 10 CPU - Python 3.12
echo ========================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.12 64-bit and add it to PATH.
    pause
    exit /b 1
)

:: Ensure requirements installed
echo [INFO] Verifying Python dependencies...
python -c "import sounddevice, pyworld, torch, fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required packages...
    pip install -r requirements.txt
)

echo [INFO] Starting VaniFlow Voice Changer server...
echo [INFO] Web UI available at: http://127.0.0.1:8000
echo.

python -m app.main

pause
