#!/usr/bin/env bash
# OVMSE online on SMACv1 8m (OG-MARL Medium), one seed per worker, all on one GPU.
#
# Protocol matches PV exp1 arm C on 5m_vs_6m: shared offline checkpoint
# (ovmse_paper_seed1_best), 1M online steps, fixed evaluation seed, periodic
# checkpoints every 25k steps. Only the online seed changes.
#
# Usage:
#   nohup bash scripts/launch_ovmse_8m_seeds.sh \
#     > logs/ovmse_8m_seeds_20261106/master.log 2>&1 &
# Optional: SEEDS="1 2 3"  GPU=2  ROOT=...  LOGS=...  PLAN_ONLY=1
set -uo pipefail

cd "$(dirname "$0")/.."
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

SEEDS="${SEEDS:-1 2 3}"
GPU="${GPU:-2}"
CFG=configs/pv_base_8m.yaml
OFFLINE=checkpoints/ovmse_paper_8m/ovmse_paper_seed1_best.pt
ROOT="${ROOT:-checkpoints/pv_exp1/8m}"
LOGS="${LOGS:-logs/ovmse_8m_seeds_20261106}"
STATUS="$ROOT/job_status.jsonl"

mkdir -p "$ROOT" "$LOGS"
log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$LOGS/launcher.log"; }

if [[ ! -f "$OFFLINE" ]]; then
  echo "FATAL: shared offline checkpoint missing: $OFFLINE" >&2
  exit 3
fi

log "code HEAD: $(git rev-parse --short HEAD)"
log "map=8m OVMSE(arm C) seeds=[$SEEDS] gpu=$GPU"
log "offline ckpt: $OFFLINE"
log "config: $CFG (sha $(sha256sum "$CFG" | cut -c1-16))"

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  log "PLAN_ONLY=1, nothing launched"
  exit 0
fi

for seed in $SEEDS; do
  (
    dir="$ROOT/C_seed${seed}"
    mkdir -p "$dir"
    log "[start] gpu=$GPU seed=$seed dir=$dir"
    printf '{"tag":"ovmse_8m_seed%s","gpu":"%s","start":"%s"}\n' \
      "$seed" "$GPU" "$(date -Iseconds)" >> "$STATUS"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --config "$CFG" \
      seed="$seed" \
      checkpoint_dir="$dir" \
      checkpoint_path="$OFFLINE" \
      log_path="$dir/online.jsonl" \
      transfer.mode=ovmse \
      system.use_ovm=true \
      online.use_ovm=true \
      ovm_diag.ckpt_every=25000 \
      > "$LOGS/ovmse_8m_seed${seed}.out" 2>&1
    rc=$?
    if [[ -f "$dir/ovmse_paper_online_ovmse_seed${seed}.pt" ]]; then
      log "[done] seed=$seed exit=$rc verified"
    else
      log "[FAIL] seed=$seed exit=$rc; see $LOGS/ovmse_8m_seed${seed}.out"
    fi
    printf '{"tag":"ovmse_8m_seed%s","gpu":"%s","end":"%s","exit":%s}\n' \
      "$seed" "$GPU" "$(date -Iseconds)" "$rc" >> "$STATUS"
  ) &
  log "launched seed=$seed pid=$!"
done

wait
log "[all done]"
