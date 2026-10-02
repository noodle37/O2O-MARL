#!/bin/bash
# Evaluate online-final and offline-best checkpoints for both 5m_vs_6m datasets, 640 episodes each.
# Usage: nohup bash scripts/run_eval_640.sh > checkpoints/eval640_nohup.log 2>&1 &
set -uo pipefail

cd /home/jxy/multiAgentRL
export CUDA_VISIBLE_DEVICES=0
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
PY=/home/jxy/miniforge3/envs/marlr/bin/python

OUT=checkpoints/eval640
mkdir -p "$OUT"

run_one () {
  local DATASET="$1" TAG="$2" CKPT="$3" LABEL="$4"
  echo "=== [$(date '+%F %T')] START $TAG/$LABEL dataset=$DATASET ckpt=$CKPT ==="
  "$PY" scripts/eval_640.py --config configs/5m_vs_6m.yaml \
    task.dataset="$DATASET" \
    --checkpoint "$CKPT" \
    --episodes 640 --chunk 32 \
    --out "$OUT/${TAG}_${LABEL}.jsonl" \
    --tag "${TAG}_${LABEL}"
  echo "=== [$(date '+%F %T')] DONE $TAG/$LABEL (exit=$?) ==="
}

run_one Medium         medium         checkpoints/ovmse_5m_vs_6m_medium/ovmse_paper_online_ovmse_seed1.pt online_final
run_one Medium         medium         checkpoints/ovmse_5m_vs_6m_medium/ovmse_paper_seed1_best.pt          offline_best
run_one Medium-Replay  medium_replay  checkpoints/ovmse_5m_vs_6m_medium_replay/ovmse_paper_online_ovmse_seed1.pt online_final
run_one Medium-Replay  medium_replay  checkpoints/ovmse_5m_vs_6m_medium_replay/ovmse_paper_seed1_best.pt          offline_best

echo "=== [$(date '+%F %T')] ALL EVALS COMPLETE ==="
