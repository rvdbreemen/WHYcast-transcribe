@echo off
REM WHYcast-transcribe Windows installer

REM Check for Chocolatey and install if missing
where choco >nul 2>nul
if %errorlevel% neq 0 (
    echo Chocolatey not found. Installing Chocolatey...
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
    set PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin
)

REM Check for Python 3.9+ and install if missing
python --version 2>NUL | findstr /R "3\.[9-9]" >nul
if %errorlevel% neq 0 (
    echo Python 3.9+ not found. Installing Python 3.11 via Chocolatey...
    choco install -y python --version=3.11.0
    refreshenv
)

REM Check again for Python
python --version 2>NUL | findstr /R "3\.[9-9]" >nul
if %errorlevel% neq 0 (
    echo Python 3.9+ is required. Please upgrade your Python installation.
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/4] Upgrading pip...
pip install --upgrade pip

echo [3/4] Installing dependencies...
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo [4/4] Checking for ffmpeg...
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo ffmpeg not found. Installing via Chocolatey...
    choco install -y ffmpeg
) else (
    echo ffmpeg is already installed.
)

echo.
echo [Done] Activate your environment with: call venv\Scripts\activate
