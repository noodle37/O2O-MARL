#!/bin/bash
# Sim2O on SMAC 5m_vs_6m: offline OMIGA -> online CBS fine-tuning -> 640-episode eval.
#
# Same protocol as the other 5m_vs_6m O2O runs: Medium vault dataset, 10M offline
# filled transitions, 3M online env steps, 8 envs, batch 128 episodes, eval every
# 10k steps over 32 episodes, epsilon 1.0 -> 0.05 over 100k.
#
# Usage:
#   nohup bash scripts/run_sim2o_5m_vs_6m.sh \
#     > logs/sim2o_5m_vs_6m_seed1_master.log 2>&1 &
#
# Optional: GPU=3 SEED=1 OFFLINE_T_MAX=10000000 T_MAX=3000000 TEST_INTERVAL=10000
set -uo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY="${PY:-/home/jxy/miniconda3/envs/multiAgentRL/bin/python}"
GPU="${GPU:-3}"
SEED="${SEED:-1}"
OFFLINE_T_MAX="${OFFLINE_T_MAX:-10000000}"
T_MAX="${T_MAX:-3000000}"
TEST_INTERVAL="${TEST_INTERVAL:-10000}"
RUN_ROOT="${RUN_ROOT:-checkpoints/sim2o_5m_vs_6m_seed${SEED}}"
CFG=configs/sim2o_5m_vs_6m.yaml

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$RUN_ROOT" logs

BEST_CKPT="$RUN_ROOT/sim2o_seed${SEED}_best.pt"
ONLINE_CKPT="$RUN_ROOT/sim2o_online_none_seed${SEED}.pt"

echo "=== Sim2O 5m_vs_6m gpu=$GPU seed=$SEED ==="
echo "code HEAD: $(git rev-parse --short HEAD)"
echo "code diff sha256: $(git diff | sha256sum | cut -d' ' -f1)"
echo "offline_t_max=$OFFLINE_T_MAX online_t_max=$T_MAX test_interval=$TEST_INTERVAL"
echo "run_root=$RUN_ROOT"

COMMON=(
  "seed=$SEED"
  "task.dataset=Medium"
  "checkpoint_dir=$RUN_ROOT"
  "checkpoint_path=$BEST_CKPT"
)

echo "=== [$(date '+%F %T')] offline OMIGA start ==="
"$PY" train.py --config "$CFG" \
  "${COMMON[@]}" \
  phase=offline \
  t_max="$OFFLINE_T_MAX" \
  test_interval="$TEST_INTERVAL" \
  log_path="$RUN_ROOT/offline.jsonl" > "$RUN_ROOT/offline.out" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "=== [$(date '+%F %T')] offline FAILED rc=$rc; see $RUN_ROOT/offline.out ===" >&2
  exit $rc
fi
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "=== offline finished but best checkpoint is missing: $BEST_CKPT ===" >&2
  exit 1
fi
echo "=== [$(date '+%F %T')] offline done best=$BEST_CKPT ==="

echo "=== [$(date '+%F %T')] online Sim2O start ==="
"$PY" train.py --config "$CFG" \
  "${COMMON[@]}" \
  phase=online \
  t_max="$T_MAX" \
  online.t_max="$T_MAX" \
  test_interval="$TEST_INTERVAL" \
  online.test_interval="$TEST_INTERVAL" \
  log_path="$RUN_ROOT/online.jsonl" > "$RUN_ROOT/online.out" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "=== [$(date '+%F %T')] online FAILED rc=$rc; see $RUN_ROOT/online.out ===" >&2
  exit $rc
fi
if [[ ! -f "$ONLINE_CKPT" ]]; then
  echo "=== online finished but checkpoint is missing: $ONLINE_CKPT ===" >&2
  exit 1
fi
echo "=== [$(date '+%F %T')] online done ckpt=$ONLINE_CKPT ==="

echo "=== [$(date '+%F %T')] eval640 start ==="
"$PY" scripts/eval_640.py --config "$CFG" \
  task.dataset=Medium \
  --checkpoint "$ONLINE_CKPT" \
  --episodes 640 --chunk 32 \
  --out "$RUN_ROOT/eval640_online_final.jsonl" \
  --tag "sim2o_seed${SEED}_online_final" > "$RUN_ROOT/eval640.out" 2>&1
rc=$?
echo "=== [$(date '+%F %T')] eval640 done rc=$rc ==="
exit $rc
