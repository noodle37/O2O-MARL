#!/usr/bin/env bash
# SAPR (Smooth Adaptive Policy Regularization) online runs on 5m_vs_6m,
# three seeds, one seed per GPU.
#
# Starting point is identical to the OVMSE / UOT arms on this map: the
# shared offline checkpoint under
#   checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt
# Only the online seed changes. SAPR runs online only: it loads that offline Q
# network as the frozen prior pi_off, keeps plain QMIX TD updates, and adapts
# the KL weight lambda_k from random local-intervention probes.
#
# Usage:
#   SEEDS="1 2 3" GPUS="1 2 3" \
#     nohup bash scripts/launch_sapr.sh > logs/sapr_batch_master.log 2>&1 &
#
# Optional:
#   SEEDS="1"  GPUS="2"  ROOT=...  LOGS=...  PLAN_ONLY=1
#
# GPU 0 is never used: it hosts another user's long-running job.
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

SEEDS="${SEEDS:-1 2 3}"
GPUS="${GPUS:-1 2 3}"
MAP="${MAP:-5m_vs_6m}"
ROOT="${ROOT:-logs/sapr_batch}"
RUN_ROOT="${RUN_ROOT:-checkpoints/sapr_5m_vs_6m}"
STATUS="$ROOT/job_status.jsonl"
FINAL_STEM="ovmse_paper_online_full_td_seed"

CKPT="checkpoints/ovmse_batch_20261014/${MAP}_offline/ovmse_paper_seed1_best.pt"
CFG="configs/sapr_${MAP}.yaml"

mkdir -p "$ROOT" "$RUN_ROOT"

log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$ROOT/launcher.log"; }

n_seeds=$(echo $SEEDS | wc -w)
n_gpus=$(echo $GPUS | wc -w)
if (( n_seeds > n_gpus )); then
  echo "refusing: $n_seeds seeds but only $n_gpus gpus ($GPUS)" >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "missing shared offline checkpoint: $CKPT" >&2
  exit 2
fi
if [[ ! -f "$CFG" ]]; then
  echo "missing config: $CFG" >&2
  exit 2
fi

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  i=0
  for seed in $SEEDS; do
    gpu=$(echo $GPUS | cut -d' ' -f $((i + 1)))
    echo "gpu=$gpu map=$MAP seed=$seed dir=$RUN_ROOT/seed$seed ckpt=$CKPT"
    i=$((i + 1))
  done
  exit 0
fi

log "code HEAD: $(git rev-parse --short HEAD)"
log "seeds=[$SEEDS] gpus=[$GPUS] map=$MAP"
log "offline ckpt: $CKPT"

pids=()
tags=()

start_job() {  # start_job <gpu> <seed>
  local gpu="$1" seed="$2"
  local dir="$RUN_ROOT/seed$seed"
  local tag="sapr_${MAP}_seed${seed}"
  local final="$dir/${FINAL_STEM}${seed}.pt"

  if [[ -f "$final" ]]; then
    log "[skip] $tag (finished: $final)"
    return 0
  fi
  mkdir -p "$dir"
  printf '{"tag":"%s","gpu":"%s","map":"%s","seed":%d,"start":"%s"}\n' \
    "$tag" "$gpu" "$MAP" "$seed" "$(date -Iseconds)" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$CFG" \
    phase=online \
    "seed=$seed" \
    "system_name=sapr $MAP seed$seed" \
    "checkpoint_dir=$dir" \
    "log_path=$dir/online.jsonl" \
    > "$ROOT/${tag}.out" 2>&1 &
  local pid=$!
  echo "$pid" > "$ROOT/${tag}.pid"
  pids+=("$pid")
  tags+=("$tag")
  log "[start] gpu=$gpu tag=$tag pid=$pid dir=$dir"
}

i=0
for seed in $SEEDS; do
  gpu=$(echo $GPUS | cut -d' ' -f $((i + 1)))
  start_job "$gpu" "$seed"
  i=$((i + 1))
done

if (( ${#pids[@]} == 0 )); then
  log "nothing to run"
  exit 0
fi

log "launched ${#pids[@]} job(s): ${tags[*]}"
log "monitor: tail -f $ROOT/*.out ; status: $STATUS"

fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    log "[done] ${tags[$idx]}"
  else
    rc=$?
    log "[FAIL] ${tags[$idx]} exit=$rc (see $ROOT/${tags[$idx]}.out)"
    printf '{"tag":"%s","end":"%s","exit":%d}\n' \
      "${tags[$idx]}" "$(date -Iseconds)" "$rc" >> "$STATUS"
    fail=1
  fi
done

log "all jobs finished (fail=$fail)"
exit $fail
