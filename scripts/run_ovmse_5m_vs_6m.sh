#!/bin/bash
# Usage: run_ovmse_5m_vs_6m.sh <Medium|Medium-Replay>
set -euo pipefail

DATASET="${1:?usage: $0 <Medium|Medium-Replay>}"
case "$DATASET" in
  Medium)
    TAG="medium"
    EXTRA=()
    ;;
  Medium-Replay)
    TAG="medium_replay"
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

RUN_DIR="checkpoints/ovmse_5m_vs_6m_${TAG}_aligned"
mkdir -p "$RUN_DIR"

COMMON=(
  task.dataset="$DATASET"
  checkpoint_dir="$RUN_DIR"
  behaviour_cache_dir="$RUN_DIR/behaviour"
)

echo "=== [$(date '+%F %T')] offline start RUN_DIR=$RUN_DIR ==="
"$PY" train.py --config configs/5m_vs_6m.yaml \
  "${COMMON[@]}" "${EXTRA[@]}" \
  log_path="$RUN_DIR/offline.jsonl"

echo "=== [$(date '+%F %T')] online start ==="
"$PY" train.py --config configs/5m_vs_6m.yaml \
  "${COMMON[@]}" "${EXTRA[@]}" \
  phase=online \
  checkpoint_path="$RUN_DIR/ovmse_paper_seed1_best.pt" \
  log_path="$RUN_DIR/online.jsonl"

FINAL_CKPT="$RUN_DIR/ovmse_paper_online_ovmse_seed1.pt"
echo "=== [$(date '+%F %T')] eval640 start ckpt=$FINAL_CKPT ==="
"$PY" scripts/eval_640.py --config configs/5m_vs_6m.yaml \
  task.dataset="$DATASET" \
  --checkpoint "$FINAL_CKPT" \
  --episodes 640 --chunk 32 \
  --out "$RUN_DIR/eval640_online_final.jsonl" \
  --tag "${TAG}_aligned_online_final"
echo "=== [$(date '+%F %T')] ALL DONE RUN_DIR=$RUN_DIR ==="
