#!/bin/bash
# mac-flow O2O on 2s3z (aligned with the original repo): 500K offline + 500K online.
set -euo pipefail

cd /home/jxy/multiAgentRL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

RUN_DIR="checkpoints/macflow_2s3z_aligned"
mkdir -p "$RUN_DIR"

echo "=== [$(date '+%F %T')] offline start RUN_DIR=$RUN_DIR GPU=$CUDA_VISIBLE_DEVICES ==="
"$PY" train.py --config configs/macflow_2s3z.yaml \
  checkpoint_dir="$RUN_DIR" \
  log_path="$RUN_DIR/offline.jsonl"

echo "=== [$(date '+%F %T')] online start ==="
"$PY" train.py --config configs/macflow_2s3z.yaml \
  phase=online \
  checkpoint_dir="$RUN_DIR" \
  checkpoint_path="$RUN_DIR/mac_flow_seed1.pt" \
  online.log_path="$RUN_DIR/online.jsonl"

echo "=== [$(date '+%F %T')] ALL DONE RUN_DIR=$RUN_DIR ==="
