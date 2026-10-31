#!/bin/bash
set -euo pipefail

SEED="${1:?usage: run_joint_reuse_seed.sh <seed> <gpu>}"
GPU="${2:?usage: run_joint_reuse_seed.sh <seed> <gpu>}"

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
CFG=configs/pv_base_5m_vs_6m.yaml
BASE_CKPT="checkpoints/pv_exp1/5m_vs_6m/C_seed${SEED}/periodic/ovm_diag_ckpt_t250000.pt"
ROOT="checkpoints/joint_reuse_online_5m_vs_6m/C_t250k/seed${SEED}"

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "FATAL: start checkpoint missing: $BASE_CKPT" >&2
  exit 3
fi
mkdir -p "$ROOT"

if [[ ! -f "$ROOT/eval640_initial.jsonl" ]]; then
  echo "=== [$(date '+%F %T')] seed=${SEED} initial 640-episode online evaluation ==="
  "$PY" scripts/eval_640.py --config "$CFG" seed="$SEED" \
    --checkpoint "$BASE_CKPT" \
    --out "$ROOT/eval640_initial.jsonl" \
    --tag "joint_reuse_C_t250k_seed${SEED}_initial"
fi

for BRANCH in joint independent none; do
  case "$BRANCH" in
    joint)
      case "$SEED" in
        1) OFFLINE_AGENTS="0,4" ;;
        2) OFFLINE_AGENTS="1,4" ;;
        3) OFFLINE_AGENTS="2,4" ;;
      esac
      ;;
    independent)
      case "$SEED" in
        1) OFFLINE_AGENTS="1,4" ;;
        2) OFFLINE_AGENTS="0,1" ;;
        3) OFFLINE_AGENTS="1,4" ;;
      esac
      ;;
    none)
      OFFLINE_AGENTS=""
      ;;
  esac

  RUN_DIR="${ROOT}/${BRANCH}"
  CKPT="${RUN_DIR}/joint_reuse_online_${BRANCH}_seed${SEED}.pt"
  mkdir -p "$RUN_DIR"

  echo "=== [$(date '+%F %T')] seed=${SEED} gpu=${GPU} branch=${BRANCH} retained=${OFFLINE_AGENTS:-none} ==="
  if [[ ! -f "$CKPT" ]]; then
    "$PY" scripts/train_joint_reuse.py \
      --config "$CFG" \
      --seed "$SEED" \
      --branch "$BRANCH" \
      --offline-agents "$OFFLINE_AGENTS" \
      --start-checkpoint "$BASE_CKPT" \
      --run-dir "$RUN_DIR" \
      --steps 250000 \
      --eval-episodes 32 \
      --test-interval 50000 \
      --save-every 100000 \
      system.use_ovm=false \
      system.mixing_ratio=0.0 \
      system.online_cql_weight=0.0 \
      transfer.mode=full_td \
      pv_probe.enabled=false \
      ovm_diag.ckpt_every=0
  else
    echo "skip training: final checkpoint exists at $CKPT"
  fi

  if [[ ! -f "${RUN_DIR}/eval640_final.jsonl" ]]; then
    echo "=== [$(date '+%F %T')] seed=${SEED} branch=${BRANCH} final 640-episode online evaluation ==="
    "$PY" scripts/eval_640.py --config "$CFG" seed="$SEED" \
      --checkpoint "$CKPT" \
      --out "${RUN_DIR}/eval640_final.jsonl" \
      --tag "joint_reuse_C_t250k_seed${SEED}_${BRANCH}"
  else
    echo "skip final eval: ${RUN_DIR}/eval640_final.jsonl exists"
  fi
  echo "=== [$(date '+%F %T')] seed=${SEED} branch=${BRANCH} DONE ==="
done

echo "=== [$(date '+%F %T')] seed=${SEED} all branches DONE ==="
