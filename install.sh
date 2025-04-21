#!/bin/bash
set -e

echo "[WHYcast-transcribe] Platform-specific installer (macOS/Linux)"

# Check for Homebrew and install if missing (macOS only)
if [[ "$(uname)" == "Darwin" ]]; then
  if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi
  # Always ensure Homebrew's bin is first in PATH
  export PATH="/opt/homebrew/bin:$PATH"

  # Install build dependencies
  echo "Installing build tools and libraries (cmake, pkg-config, ffmpeg)..."
  brew install cmake pkg-config ffmpeg || true
  export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
fi

# On macOS, always install and use Homebrew’s Python 3.11 explicitly
if [[ "$(uname)" == "Darwin" ]]; then
  echo "Installing Homebrew Python@3.11..."
  brew install python@3.11 || true
  # Use the Homebrew Python 3.11 binary directly
  export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"
  PYTHON_BIN="/opt/homebrew/opt/python@3.11/bin/python3.11"
else
  PYTHON_BIN="python3"
fi

echo "Using Python version: $($PYTHON_BIN --version)"

echo "[1/4] Creating virtual environment..."
"$PYTHON_BIN" -m venv venv
source venv/bin/activate

# Ensure the `python` command exists in venv
VENV_BIN="$(pwd)/venv/bin"
echo "Ensuring 'python' points to python3 in venv..."
ln -sf "$VENV_BIN/python3" "$VENV_BIN/python"

echo "[2/4] Upgrading pip..."
# Use venv pip
"$VENV_BIN/pip" install --upgrade pip

# Create a modified requirements file without problematic packages
if [[ "$(uname)" == "Darwin" ]]; then
  echo "[3/4] Installing dependencies for macOS..."
  
  # Install PyTorch with MPS support first
  echo "Installing PyTorch with MPS support for Apple Silicon..."
  "$VENV_BIN/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  
  # Install sentencepiece binary wheel (pin version with pre-built macOS universal2 wheels)
  echo "Installing pre-built sentencepiece binary wheel..."
  "$VENV_BIN/pip" install "sentencepiece==0.1.98"
  
  # Now install the rest of the requirements
  grep -v "sentencepiece" requirements.txt | grep -v "torch" | grep -v "torchaudio" | "$VENV_BIN/pip" install -r /dev/stdin
else
  echo "[3/4] Installing core dependencies..."
  "$VENV_BIN/pip" install -r requirements.txt
fi

echo "[4/4] Checking for ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
  echo "ffmpeg not found. Installing via package manager..."
  if [[ "$(uname)" == "Darwin" ]]; then
    if command -v brew &> /dev/null; then
      brew install ffmpeg
    else
      echo "Homebrew not found. Please install ffmpeg manually: https://ffmpeg.org/download.html"
    fi
  elif command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
  elif command -v yum &> /dev/null; then
    sudo yum install -y ffmpeg
  else
    echo "Could not install ffmpeg automatically. Please install manually: https://ffmpeg.org/download.html"
  fi
else
  echo "ffmpeg is already installed."
fi

echo "\n[Testing installation...]"
# Use the venv python for testing
"$VENV_BIN/python" -m whycast_transcribe.cli --help &> /dev/null && {
  echo "\n[Installation successful!] WHYcast-transcribe is ready to use."
} || {
  echo "\n[Warning] Installation completed, but CLI test failed."
  echo "Run: $VENV_BIN/python -m whycast_transcribe.cli --help"
}

echo "Activate your environment with: source venv/bin/activate"
echo "Then run: venv/bin/python -m whycast_transcribe.cli --help for usage."
