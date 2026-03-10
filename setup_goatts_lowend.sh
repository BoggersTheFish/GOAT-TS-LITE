#!/usr/bin/env bash
# GOAT-TS Lite — automated setup for low-end CPU-only hardware
# Supports Linux and macOS. No Docker, Redis, Spark, Kubernetes, or cloud deps.

set -e

echo "[GOAT-TS Lite] Starting automated setup..."

# --- STEP 1: Detect OS ---
OS_TYPE=""
if [[ "$(uname -s)" == "Linux" ]]; then
  OS_TYPE="linux"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  OS_TYPE="macos"
else
  echo "[ERROR] Unsupported OS. This script supports Linux and macOS only."
  exit 1
fi
echo "[GOAT-TS Lite] Detected OS: $OS_TYPE"

# --- Check RAM and warn if below 4GB ---
check_ram() {
  local ram_gb=0
  if [[ "$OS_TYPE" == "linux" ]]; then
    ram_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0")
  elif [[ "$OS_TYPE" == "macos" ]]; then
    ram_gb=$(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024))
  fi
  if [[ -n "$ram_gb" && "$ram_gb" -lt 4 ]]; then
    echo "[WARNING] RAM is below 4GB (detected: ${ram_gb}GB). GOAT-TS Lite will enforce reduced ticks and batch sizes."
  fi
}
check_ram

# --- Install system dependencies ---
install_deps() {
  if command -v apt-get &>/dev/null; then
    echo "[GOAT-TS Lite] Installing dependencies (apt)..."
    sudo apt-get update -qq
    sudo apt-get install -y git curl python3.11 python3.11-venv 2>/dev/null || true
  elif command -v brew &>/dev/null; then
    echo "[GOAT-TS Lite] Installing dependencies (Homebrew)..."
    brew install git curl python@3.11 2>/dev/null || true
  else
    echo "[GOAT-TS Lite] Please install manually: git, curl, python3.11, python3.11-venv (or python@3.11 on macOS)"
  fi
}
install_deps

# --- Clone repository if not present ---
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[GOAT-TS Lite] Not a git repo; skipping clone. Run this script from a clone of GOAT-TS."
else
  echo "[GOAT-TS Lite] Repository present at $REPO_DIR"
fi

# --- Create virtual environment ---
cd "$REPO_DIR"
if [[ ! -d "venv" ]]; then
  echo "[GOAT-TS Lite] Creating virtual environment..."
  python3.11 -m venv venv || python3 -m venv venv
fi
echo "[GOAT-TS Lite] Activating venv..."
# shellcheck source=/dev/null
source venv/bin/activate

# --- Upgrade pip ---
pip install --upgrade pip -q

# --- STEP 2: Install CPU-only PyTorch ---
echo "[GOAT-TS Lite] Installing CPU-only PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q

echo "[GOAT-TS Lite] Installing Intel CPU optimizations..."
pip install intel-extension-for-pytorch -q 2>/dev/null || echo "[GOAT-TS Lite] intel-extension-for-pytorch optional; continuing without."

# --- Install repository dependencies ---
if [[ ! -f "requirements.txt" ]]; then
  echo "[GOAT-TS Lite] Creating requirements.txt..."
  cat > requirements.txt << 'REQ'
# torch/torchvision/torchaudio installed above (CPU-only)
intel-extension-for-pytorch
numpy
pandas
networkx
pyyaml
streamlit
REQ
fi
pip install -r requirements.txt -q

# --- STEP 3: Optional Memgraph ---
read -r -p "Install Memgraph graph database? (y/n) " INSTALL_MEMGRAPH
if [[ "$INSTALL_MEMGRAPH" == "y" || "$INSTALL_MEMGRAPH" == "Y" ]]; then
  if [[ "$OS_TYPE" == "linux" ]]; then
    echo "[GOAT-TS Lite] Install Memgraph Community Edition manually from https://memgraph.com/docs/install"
    echo "  Then start Memgraph and use config/graph.yaml (host: localhost, port: 7687)."
  else
    echo "[GOAT-TS Lite] Memgraph Community is Linux-only. Use Docker or a Linux VM for Memgraph."
  fi
  pip install pymgclient -q 2>/dev/null || echo "[GOAT-TS Lite] pymgclient install failed; install Memgraph and retry."
  mkdir -p config
  if [[ ! -f "config/graph.yaml" ]]; then
    cat > config/graph.yaml << 'YAML'
graph_backend: memgraph
host: localhost
port: 7687
YAML
    echo "[GOAT-TS Lite] Created config/graph.yaml"
  fi
else
  echo "[GOAT-TS Lite] Skipping Memgraph."
  mkdir -p config
  if [[ ! -f "config/graph.yaml" ]]; then
    cat > config/graph.yaml << 'YAML'
graph_backend: memory
host: localhost
port: 7687
YAML
    echo "[GOAT-TS Lite] Created config/graph.yaml (memory backend)"
  fi
fi

# --- STEP 10: Test PyTorch installation ---
echo "[GOAT-TS Lite] Testing PyTorch (CPU)..."
python -c "
import torch
x = torch.randn(1000, 1000)
y = torch.matmul(x, x)
print('[GOAT-TS Lite] PyTorch CPU test OK.')
"

# --- STEP 11: Streamlit ---
if ! python -c "import streamlit" 2>/dev/null; then
  pip install streamlit -q
fi
echo "[GOAT-TS Lite] Streamlit available. Run: streamlit run streamlit_app.py"

echo "[GOAT-TS Lite] Setup complete. Activate with: source venv/bin/activate"
echo "  Then: python reasoning/demo_loop.py --ticks 3 --dry-run"
echo "  And:  streamlit run streamlit_app.py"
