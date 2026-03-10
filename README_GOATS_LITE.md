# GOAT-TS Lite

Minimal-cost CPU-only runtime for low-end hardware (e.g. Intel Pentium Silver, 4–8GB RAM). Keeps the GOAT-TS architecture with small reasoning cycles, wave-state nodes, and optional Memgraph.

## First time (Windows, PowerShell)

**Important:** Use the **venv’s Python** for all installs and runs so packages stay inside the project (avoid "Defaulting to user installation" and wrong interpreter).

1. **Create venv and install into it** (from project root):

   ```powershell
   cd C:\Users\BoggersTheFish\Desktop\GOATS
   python -m venv venv
   .\venv\Scripts\python.exe -m pip install --upgrade pip
   .\venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   .\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   *(requirements.txt no longer includes `intel-extension-for-pytorch` — that package is Linux-only and was failing on Windows.)*

2. **Run the reasoning loop:**

   ```powershell
   .\venv\Scripts\python.exe reasoning/demo_loop.py --ticks 3 --dry-run
   ```

3. **Start the Streamlit UI** (use the venv’s Python so `streamlit` is found):

   ```powershell
   .\venv\Scripts\python.exe -m streamlit run streamlit_app.py
   ```

**One-liner script:** From project root you can run `.\run_windows.ps1` to install deps (if needed), run a dry-run, then start Streamlit.

## Quick start (Linux / macOS)

```bash
bash setup_goatts_lowend.sh
source venv/bin/activate
python reasoning/demo_loop.py --ticks 3 --dry-run
streamlit run streamlit_app.py
```

## Verification

1. **Setup**: `bash setup_goatts_lowend.sh` — installs venv, CPU PyTorch, deps; optional Memgraph prompt; runs a quick PyTorch CPU test.
2. **Reasoning loop**: `python reasoning/demo_loop.py --ticks 3 --dry-run` — 3 ticks, dry-run (no heavy model inference).
3. **UI**: `streamlit run streamlit_app.py` — Streamlit dashboard.

## Options

- `--ticks N` — reasoning ticks (default 3, max 5).
- `--dry-run` — skip model inference; simulate cycles and update wave nodes only (for testing on low hardware).

If RAM &lt; 4GB, the runtime automatically enables GOAT-TS Lite mode (ticks=3, reduced batch sizes, reduced graph updates) and prints:  
`Low RAM detected — GOAT-TS Lite mode enabled.`

## Config

- **config/graph.yaml** — graph backend (`memgraph` or `memory`), host, port (7687 for Memgraph).

No Docker, Redis, Spark, Kubernetes, or cloud dependencies. CPU-only execution.
