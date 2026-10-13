#!/bin/bash
# Experiment 1: offline Q_tot accuracy (bias + rank concordance) on GPU 3.
# Chain A measures the existing 5m_vs_6m seed1 checkpoint immediately.
# Chain B downloads the 3s5z_vs_3s6z vault, trains seeds 1-3, and measures
# each checkpoint right after its training finishes.
# Usage: nohup bash scripts/run_offline_q_measure.sh > logs/measure_offline_q_master.log 2>&1 &
set -uo pipefail

cd /home/jxy/multiAgentRL
export CUDA_VISIBLE_DEVICES=3
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

OUT=checkpoints/measure_offline_q
mkdir -p "$OUT" logs

measure_pair () {  # $1=config $2=checkpoint $3=tag
  local cfg=$1 ckpt=$2 tag=$3
  echo "=== [$(date '+%F %T')] dataset-Q measure $tag ckpt=$ckpt ==="
  "$PY" scripts/measure_offline_q.py \
    --config "$cfg" --checkpoint "$ckpt" \
    --out "$OUT/${tag}_dataset_q.json"
  echo "=== [$(date '+%F %T')] local rank inversion $tag ==="
  "$PY" scripts/verify_local_rank_inversion.py \
    --config "$cfg" --checkpoint "$ckpt" \
    --out "$OUT/${tag}_local_inversion.json"
  echo "=== [$(date '+%F %T')] measure done $tag ==="
}

# Chain A: existing 5m_vs_6m seed1 offline-best checkpoint.
(
  measure_pair configs/5m_vs_6m.yaml \
    checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1_best.pt \
    5m_vs_6m_seed1
) > logs/measure_5m_vs_6m_seed1.out 2>&1 &
PID_A=$!
echo "chain A (5m_vs_6m seed1 measure) pid=$PID_A"

# Chain B: 3s5z_vs_3s6z — download vault once, then train+measure 3 seeds in parallel.
(
  echo "=== [$(date '+%F %T')] pre-download 3s5z_vs_3s6z vault ==="
  "$PY" -c "from data.vault_download import download_and_unzip_vault; download_and_unzip_vault('og_marl', 'smac_v1', '3s5z_vs_3s6z')"
  echo "=== [$(date '+%F %T')] vault ready, launching training ==="
  for seed in 1 2 3; do
    RUN_DIR="checkpoints/ovmse_3s5z_vs_3s6z_seed${seed}"
    mkdir -p "$RUN_DIR"
    (
      "$PY" train.py --config configs/3s5z_vs_3s6z.yaml \
        seed="${seed}" \
        checkpoint_dir="$RUN_DIR" \
        behaviour_cache_dir="$RUN_DIR/behaviour" \
        checkpoint_path="$RUN_DIR/ovmse_paper_seed${seed}_best.pt" \
        log_path="$RUN_DIR/offline.jsonl"
      measure_pair configs/3s5z_vs_3s6z.yaml \
        "$RUN_DIR/ovmse_paper_seed${seed}_best.pt" \
        "3s5z_vs_3s6z_seed${seed}"
    ) > "logs/ovmse_3s5z_vs_3s6z_seed${seed}.out" 2>&1 &
    echo "3s5z_vs_3s6z seed${seed} train+measure pid=$!"
  done
  wait
) > logs/measure_3s5z_chain.out 2>&1 &
PID_B=$!
echo "chain B (3s5z_vs_3s6z train x3 + measure) pid=$PID_B"

wait "$PID_A"
echo "=== [$(date '+%F %T')] chain A done (exit=$?) ==="
wait "$PID_B"
echo "=== [$(date '+%F %T')] chain B done (exit=$?) ==="
echo "=== [$(date '+%F %T')] ALL DONE ==="
