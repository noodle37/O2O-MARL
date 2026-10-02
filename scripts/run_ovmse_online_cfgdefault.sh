#!/bin/bash
# Online-only fine-tune from existing offline best with CONFIG DEFAULTS (lr=1e-3, batch=128).
# Usage: run_ovmse_online_cfgdefault.sh <Medium|Medium-Replay>
set -euo pipefail

DATASET="${1:?usage: $0 <Medium|Medium-Replay>}"
case "$DATASET" in
  Medium)
    TAG="medium"
    SRC="checkpoints/ovmse_5m_vs_6m_medium/ovmse_paper_seed1_best.pt"
    EXTRA=()
    ;;
  Medium-Replay)
    TAG="medium_replay"
    SRC="checkpoints/ovmse_5m_vs_6m_medium_replay/ovmse_paper_seed1_best.pt"
    EXTRA=(system.mixing_ratio=0.0 system.lambda_memory_end=0.05)
    ;;
  *)
    echo "usage: $0 <Medium|Medium-Replay>" >&2
    exit 1
    ;;
esac

cd /home/jxy/multiAgentRL
export CUDA_VISIBLE_DEVICES=0
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
PY=/home/jxy/miniforge3/envs/marlr/bin/python

RUN_DIR="checkpoints/ovmse_5m_vs_6m_${TAG}_bs128"
mkdir -p "$RUN_DIR"

"$PY" train.py --config configs/5m_vs_6m.yaml \
  task.dataset="$DATASET" \
  phase=online \
  checkpoint_path="$SRC" \
  checkpoint_dir="$RUN_DIR" \
  log_path="$RUN_DIR/online.jsonl" \
  "${EXTRA[@]}"
