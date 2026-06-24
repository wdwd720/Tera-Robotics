#!/usr/bin/env bash
# RUN ON THE GPU BOX. Trains ACT on the reBot reseat dataset.
# Prereqs: the [train] extra installed (Python 3.11 .venv-train), the dataset on the Hub
# or synced locally, and HF_USER set. ACT trains in ~1-2 h on an A100 under 8 GB VRAM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MUJOCO_GL="${MUJOCO_GL:-egl}"   # GPU-box offscreen rendering
HF_USER="${HF_USER:?set HF_USER to your Hugging Face username}"
REPO_ID="${1:-$HF_USER/rebot_reseat}"
STEPS="${2:-100000}"

# shellcheck disable=SC1091
[ -f .venv-train/bin/activate ] && . .venv-train/bin/activate

lerobot-train \
  --dataset.repo_id="$REPO_ID" \
  --policy.type=act \
  --output_dir=outputs/train/act_rebot \
  --job_name=act_rebot \
  --policy.device=cuda \
  --batch_size=8 \
  --steps="$STEPS" \
  --wandb.enable=true \
  --policy.repo_id="$HF_USER/act_rebot"
