#!/bin/bash
# Step 0 of the matched-protocol plan: evaluate the three candidate offline
# checkpoints under one common 640-episode protocol.
# Usage: nohup bash scripts/run_step0_ckpt_eval.sh > logs/step0_ckpt_eval.out 2>&1 &
set -uo pipefail

cd /home/jxy/multiAgentRL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

OUT=checkpoints/protocol_audit_20261020
mkdir -p "$OUT" logs

run_one () {
  local tag="$1" ckpt="$2"
  echo "=== [$(date '+%F %T')] START ${tag} ckpt=${ckpt} ==="
  "$PY" scripts/eval_640.py --config configs/5m_vs_6m.yaml \
    task.dataset=Medium \
    --checkpoint "$ckpt" \
    --episodes 640 --chunk 32 \
    --out "$OUT/eval640_${tag}.jsonl" \
    --tag "$tag"
  echo "=== [$(date '+%F %T')] DONE ${tag} (exit=$?) ==="
}

run_one final10m checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1.pt
run_one best9m   checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1_best.pt
run_one best5m   checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt

echo "=== [$(date '+%F %T')] STEP0 COMPLETE ==="
