#!/usr/bin/env bash
# bootstrap_muesli_mac.sh
# -------------------------------------------------------------
# One-shot bootstrap script that prepares a clean macOS system to run the
# Muesli desktop application *without* bundling it in an executable.  It:
#   1. Installs Homebrew if missing
#   2. Uses brew to install native deps: portaudio, ffmpeg, whisper.cpp, etc.
#   3. Installs the Ollama LLM runtime when absent
#   4. Creates a Python virtual-env (./.venv) and pip-installs requirements.txt
#   5. Fetches a Whisper model (medium-en by default) into ~/.muesli/models/
#   6. Prints how to launch the application afterwards
#
# Usage:
#   chmod +x scripts/bootstrap_muesli_mac.sh
#   ./scripts/bootstrap_muesli_mac.sh             # default medium model
#   MODEL_SIZE=small ./scripts/bootstrap_muesli_mac.sh   # tiny|base|small|medium|large|large-v3
#
# Tested on: macOS 15.5 (Apple Silicon)
# -------------------------------------------------------------
set -Eeuo pipefail

APP_NAME="Muesli"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
MODEL_SIZE="${MODEL_SIZE:-medium}"
MODELS_DIR="$HOME/.muesli/models/whisper"

log() { echo -e "\033[1;32m[bootstrap]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warning]\033[0m $*"; }
err() { echo -e "\033[1;31m[error]\033[0m $*"; }

# -------------------------------------------------------------
# 1. Homebrew + native packages
# -------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  log "Homebrew not found – installing …"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$($(command -v brew) shellenv)"
else
  eval "$($(command -v brew) shellenv)"
fi

BREW_PKGS=(portaudio ffmpeg cmake whisper-cpp)
for pkg in "${BREW_PKGS[@]}"; do
  if brew list "$pkg" &>/dev/null; then
    log "brew package '$pkg' already installed"
  else
    log "Installing brew package '$pkg'"
    brew install "$pkg"
  fi
done

# -------------------------------------------------------------
# 2. Ollama runtime (LLM provider)
# -------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  log "Installing Ollama runtime"
  curl -fsSL https://ollama.com/install.sh | sh
else
  log "Ollama already present ($(ollama --version))"
fi

# -------------------------------------------------------------
# 3. Python virtual environment + pip deps
# -------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating Python venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

log "Upgrading pip"
pip install --disable-pip-version-check --upgrade pip >/dev/null

log "Installing Python dependencies (requirements.txt)"
pip install -r requirements.txt
# Optional – whispercpp python bindings (non-critical)
pip install --quiet whispercpp || true

# -------------------------------------------------------------
# 4. Download Whisper model (if missing)
# -------------------------------------------------------------
MODEL_FILENAME="ggml-${MODEL_SIZE}.en.bin"
MODEL_PATH="$MODELS_DIR/$MODEL_FILENAME"
if [[ -f "$MODEL_PATH" ]]; then
  log "Whisper model already present ($MODEL_FILENAME)"
else
  log "Downloading Whisper model '$MODEL_SIZE' (~might be large)…"
  mkdir -p "$MODELS_DIR"
  curl -L --progress-bar \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL_FILENAME" \
    -o "$MODEL_PATH.tmp"
  mv "$MODEL_PATH.tmp" "$MODEL_PATH"
  log "Model saved to $MODEL_PATH"
fi

# -------------------------------------------------------------
# 5. Pull Ollama model (if needed)
# -------------------------------------------------------------
log "Ensuring LLM model is available in Ollama"
# Start Ollama service if not running
if ! pgrep -x "ollama" > /dev/null; then
  log "Starting Ollama service"
  ollama serve > /dev/null 2>&1 &
  sleep 2  # Give it a moment to start
fi

# Pull the default model (llama3.1:8b as per main.py config)
if ! ollama list | grep -q "llama3.1:8b"; then
  log "Pulling LLM model (llama3.1:8b) - this may take a while..."
  ollama pull llama3.1:8b
else
  log "LLM model already available"
fi

# -------------------------------------------------------------
# 6. Summary + next steps
# -------------------------------------------------------------
log "Bootstrap complete!  To run the application:\n"
cat <<EOF
  source $VENV_DIR/bin/activate
  python main.py
EOF

log "Enjoy your breakfast 🥣."
