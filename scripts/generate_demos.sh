#!/usr/bin/env bash
# Generate scripted reseat demos under domain randomization and validate the dataset.
# Usage: scripts/generate_demos.sh [n_episodes] [out_dir] [image_size]
# On the GPU box set MUJOCO_GL=egl before running for fast offscreen rendering.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && . .venv/bin/activate

N="${1:-200}"
OUT="${2:-datasets/rebot_scripted}"
HW="${3:-128}"

python -m data.record_scripted --n-episodes "$N" --out-dir "$OUT" --image-hw "$HW" "$HW" --record-every 3
python -m data.dataset_validator "$OUT"
