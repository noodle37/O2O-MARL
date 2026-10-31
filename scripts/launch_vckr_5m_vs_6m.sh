#!/usr/bin/env bash
# VCKR on 5m_vs_6m: three online seeds on GPUs 2 and 3.
#
# All seeds share the same offline checkpoint and the standard 1M-step online
# protocol; only the online seed changes. Seed 1 and seed 3 share GPU 2 (the
# job is env-bound, and each process needs only a few GB), seed 2 uses GPU 3.
#
# Usage:
#   nohup bash scripts/launch_vckr_5m_vs_6m.sh > logs/vckr_5m_vs_6m_master.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.."
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
CFG=configs/vckr_5m_vs_6m.yaml
ROOT=checkpoints/vckr_5m_vs_6m
LOGS=logs/vckr_5m_vs_6m
STATUS="$ROOT/job_status.jsonl"

mkdir -p "$ROOT" "$LOGS"

log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$ROOT/launcher.log"; }

run_one() {
  local seed="$1" gpu="$2"
  local dir="$ROOT/seed${seed}"
  local out="$dir/vckr_online_full_td_seed${seed}.pt"
  local logf="$LOGS/seed${seed}_gpu${gpu}.out"
  mkdir -p "$dir"
  if [[ -f "$out" ]]; then
    log "[skip] seed=${seed} final checkpoint exists: $out"
    return 0
  fi
  log "[start] seed=${seed} gpu=${gpu} log=${logf}"
  printf '{"tag":"vckr_seed%s","gpu":"%s","start":"%s","dir":"%s"}\n' \
    "$seed" "$gpu" "$(date -Iseconds)" "$dir" >> "$STATUS"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py \
    --config "$CFG" \
    phase=online \
    seed="$seed" \
    checkpoint_dir="$dir" \
    log_path="$dir/online.jsonl" \
    > "$logf" 2>&1 &
  local pid=$!
  echo "$pid" > "$dir/run.pid"
  log "[pid] seed=${seed} gpu=${gpu} pid=${pid}"
}

log "code HEAD: $(git rev-parse --short HEAD)"
log "config: $CFG | offline ckpt: checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt"
log "protocol: t_max=1M, 3 seeds, GPUs 2/3"

run_one 1 2
run_one 2 3
run_one 3 2

for seed in 1 2 3; do
  pid_file="$ROOT/seed${seed}/run.pid"
  if [[ -f "$pid_file" ]]; then
    pid=$(cat "$pid_file")
    log "[wait] seed=${seed} pid=${pid}"
  fi
done
wait
rc=$?
log "all VCKR jobs finished (wait rc=${rc})"
for seed in 1 2 3; do
  printf '{"tag":"vckr_seed%s","end":"%s","final":"%s"}\n' \
    "$seed" "$(date -Iseconds)" "$ROOT/seed${seed}/vckr_online_full_td_seed${seed}.pt" >> "$STATUS"
done
