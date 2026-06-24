#!/usr/bin/env bash
# reBot setup. Builds the torch-free core venv (.venv) on the system Python, and,
# when Python 3.11 is available, a separate .venv-train for the lerobot/torch extras.
# Training itself runs on a cloud GPU box, not locally.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[setup] core venv (.venv) on $(python3 --version)"
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[vision,dev]"
python -c "import mujoco; print('[setup] mujoco', mujoco.__version__)"
deactivate

# The ML stack lives in its own Python 3.11 venv because lerobot v0.5 targets
# Python 3.12+ and torch arm64 wheels on 3.13 are spotty. The subprocess and
# separate-venv boundary keeps the core torch-free.
if command -v python3.11 >/dev/null 2>&1; then
  echo "[setup] train venv (.venv-train) on $(python3.11 --version)"
  python3.11 -m venv .venv-train
  # shellcheck disable=SC1091
  . .venv-train/bin/activate
  python -m pip install --upgrade pip
  pip install -e ".[train]"
  deactivate
else
  echo "[setup] python3.11 not found: skipping .venv-train (training is deferred)."
  echo "[setup] install Python 3.11 then re-run, or create .venv-train on the GPU box."
fi

echo "[setup] done. Activate the core env with: . .venv/bin/activate"
