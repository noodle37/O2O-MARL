#!/bin/bash
# PEX-M online pilot on 5m_vs_6m, then endpoint evaluations.
#
# Stage 1: train the online arm for PEXM_T_MAX env steps on GPU 2 (default),
#          one frozen offline arm and an episode-level Thompson selector.
# Stage 2: 640-episode greedy evaluation of the online arm, the offline arm,
#          and Thompson / posterior-mean selectors (Wilson 95% CI).
# Stage 3: optional 640-episode endpoints for the UOT and OVMSE seed-1
#          baselines that this pilot is compared against.
#
# Env:
#   PEXM_GPU=2              physical GPU
#   PEXM_T_MAX=3000000      training steps
#   PEXM_SEED=1             training seed
#   PEXM_FORCE=1            retrain even if the final checkpoint exists
#   PEXM_EVAL_BASELINES=0   skip UOT/OVMSE endpoint evaluations
#   PEXM_SKIP_EVAL=1        train only
set -euo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${PEXM_GPU:-2}"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

CONFIG=configs/pexm_5m_vs_6m.yaml
OUT_DIR=checkpoints/pexm_5m_vs_6m
SEED="${PEXM_SEED:-1}"
T_MAX="${PEXM_T_MAX:-3000000}"
OFFLINE_CKPT=checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1.pt
ONLINE_CKPT="$OUT_DIR/pexm_online_seed${SEED}.pt"
SELECTOR="$OUT_DIR/pexm_selector_seed${SEED}.json"

mkdir -p "$OUT_DIR"
DRIVER_LOG="$OUT_DIR/driver.log"
log() { echo "=== [$(date '+%F %T')] $* ===" | tee -a "$DRIVER_LOG"; }

[[ -f "$OFFLINE_CKPT" ]] || { log "FATAL missing offline checkpoint $OFFLINE_CKPT"; exit 1; }

log "PEX-M start seed=$SEED t_max=$T_MAX gpu=$CUDA_VISIBLE_DEVICES"
if [[ -f "$ONLINE_CKPT" && "${PEXM_FORCE:-0}" != "1" ]]; then
  log "skip training: $ONLINE_CKPT exists (PEXM_FORCE=1 to rerun)"
else
  "$PY" scripts/train_pexm.py --config "$CONFIG" \
    "seed=$SEED" "t_max=$T_MAX" 2>&1 | tee "$OUT_DIR/train_seed${SEED}.log"
fi

if [[ "${PEXM_SKIP_EVAL:-0}" == "1" ]]; then
  log "PEXM_SKIP_EVAL=1, stopping after training"
  exit 0
fi

[[ -f "$ONLINE_CKPT" ]] || { log "FATAL missing online checkpoint $ONLINE_CKPT"; exit 1; }
[[ -f "$SELECTOR" ]] || { log "FATAL missing selector $SELECTOR"; exit 1; }

log "eval640 PEX-M (online/offline/thompson/mean)"
"$PY" scripts/eval_pexm_640.py --config "$CONFIG" \
  --online-checkpoint "$ONLINE_CKPT" \
  --offline-checkpoint "$OFFLINE_CKPT" \
  --selector "$SELECTOR" \
  --episodes 640 --chunk 32 \
  --out "$OUT_DIR/eval640_pexm_seed${SEED}.jsonl" \
  2>&1 | tee "$OUT_DIR/eval640_pexm_seed${SEED}.log"

if [[ "${PEXM_EVAL_BASELINES:-1}" == "1" ]]; then
  UOT_CKPT=checkpoints/uot_temporal_5m_vs_6m/uot_online_none_seed1.pt
  OVMSE_CKPT=checkpoints/ovmse_batch_20261014/5m_vs_6m_online/ovmse_paper_online_ovmse_seed1.pt

  if [[ -f "$UOT_CKPT" ]]; then
    log "eval640 UOT seed1 endpoint"
    "$PY" scripts/eval_640.py --config configs/uot_temporal_5m_vs_6m.yaml \
      "seed=1" "task.dataset=Medium" \
      --checkpoint "$UOT_CKPT" --episodes 640 --chunk 32 \
      --out "$OUT_DIR/eval640_uot_seed1.jsonl" --tag "uot_seed1" \
      2>&1 | tee "$OUT_DIR/eval640_uot_seed1.log"
  else
    log "WARNING missing UOT checkpoint $UOT_CKPT; skipping"
  fi

  if [[ -f "$OVMSE_CKPT" ]]; then
    log "eval640 OVMSE online seed1 endpoint"
    "$PY" scripts/eval_640.py --config configs/5m_vs_6m.yaml \
      "seed=1" "task.dataset=Medium" "system.moderate_lambda=false" \
      --checkpoint "$OVMSE_CKPT" --episodes 640 --chunk 32 \
      --out "$OUT_DIR/eval640_ovmse_seed1.jsonl" --tag "ovmse_seed1" \
      2>&1 | tee "$OUT_DIR/eval640_ovmse_seed1.log"
  else
    log "WARNING missing OVMSE checkpoint $OVMSE_CKPT; skipping"
  fi
fi

log "PEX-M ALL DONE seed=$SEED out=$OUT_DIR"
