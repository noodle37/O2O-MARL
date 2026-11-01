#!/usr/bin/env bash
# ROAD (adaptive offline/online data mixing) online runs on 2s3z and 5m_vs_6m,
# one seed per GPU. Both maps of a seed share that seed's GPU.
#
# Starting point is identical to the OVMSE / UOT arms on each map: the
# shared offline checkpoint under checkpoints/ovmse_batch_20261014/<map>_offline/.
# The online recipe is plain QMIX TD; ROAD only chooses the offline episode
# fraction for each period (see documentation/road_o2o_marl_migration.md).
#
# Usage:
#   SEEDS="1 2 3" GPUS="1 2 3" MAPS="2s3z 5m_vs_6m" \
#     nohup bash scripts/launch_road.sh > logs/road_batch_master.log 2>&1 &
#
# Optional:
#   SEEDS="1"  GPUS="0"  MAPS="2s3z"  ROOT=...  LOGS=...  PLAN_ONLY=1
#
# The caller chooses the GPUs; this script does not reserve cards.
set -uo pipefail

cd "$(dirname "$0")/.."
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
# Keep the per-process thread count modest when several jobs share a card.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

SEEDS="${SEEDS:-1}"
GPUS="${GPUS:-0}"
MAPS="${MAPS:-2s3z 5m_vs_6m}"
ROOT="${ROOT:-checkpoints/road_batch}"
LOGS="${LOGS:-logs/road_batch}"
STATUS="$ROOT/job_status.jsonl"
# Final online checkpoint written by run_online: stem "ovmse_paper" (default for
# PaperQMIXLearner) + mode "full_td" + the run seed.
FINAL_STEM="ovmse_paper_online_full_td_seed"

mkdir -p "$ROOT" "$LOGS"

log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$ROOT/launcher.log"; }

n_seeds=$(echo $SEEDS | wc -w)
n_gpus=$(echo $GPUS | wc -w)
if (( n_seeds > n_gpus )); then
  echo "refusing: $n_seeds seeds but only $n_gpus gpus ($GPUS)" >&2
  exit 2
fi

# Every run must start from the same offline weights OVMSE / UOT used.
for map in $MAPS; do
  ckpt="checkpoints/ovmse_batch_20261014/${map}_offline/ovmse_paper_seed1_best.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "missing shared offline checkpoint for $map: $ckpt" >&2
    exit 2
  fi
done

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  i=0
  for seed in $SEEDS; do
    gpu=$(echo $GPUS | cut -d' ' -f $((i + 1)))
    for map in $MAPS; do
      echo "gpu=$gpu map=$map seed=$seed dir=checkpoints/road_${map}/seed${seed}"
    done
    i=$((i + 1))
  done
  exit 0
fi

log "code HEAD: $(git rev-parse --short HEAD)"
log "seeds=[$SEEDS] gpus=[$GPUS] maps=[$MAPS]"
log "offline ckpt per map: checkpoints/ovmse_batch_20261014/<map>_offline/ovmse_paper_seed1_best.pt"

pids=()
tags=()

start_job() {  # start_job <gpu> <map> <seed>
  local gpu="$1" map="$2" seed="$3"
  local cfg="configs/road_${map}.yaml"
  local dir="checkpoints/road_${map}/seed${seed}"
  local tag="road_${map}_seed${seed}"
  local final="$dir/${FINAL_STEM}${seed}.pt"

  if [[ ! -f "$cfg" ]]; then
    log "[skip] $tag (no config $cfg)"
    return 0
  fi
  if [[ -f "$final" ]]; then
    log "[skip] $tag (finished: $final)"
    return 0
  fi
  mkdir -p "$dir"
  printf '{"tag":"%s","gpu":"%s","map":"%s","seed":%d,"start":"%s"}\n' \
    "$tag" "$gpu" "$map" "$seed" "$(date -Iseconds)" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$cfg" \
    phase=online \
    "seed=$seed" \
    "system_name=MaROAD $map seed$seed" \
    "checkpoint_dir=$dir" \
    "log_path=$dir/online.jsonl" \
    > "$LOGS/${tag}.out" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOGS/${tag}.pid"
  pids+=("$pid")
  tags+=("$tag")
  log "[start] gpu=$gpu tag=$tag pid=$pid dir=$dir"
}

i=0
for seed in $SEEDS; do
  gpu=$(echo $GPUS | cut -d' ' -f $((i + 1)))
  for map in $MAPS; do
    start_job "$gpu" "$map" "$seed"
  done
  i=$((i + 1))
done

if (( ${#pids[@]} == 0 )); then
  log "nothing to run"
  exit 0
fi

log "launched ${#pids[@]} job(s): ${tags[*]}"
log "monitor: tail -f $LOGS/*.out ; status: $STATUS"

fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    log "[done] ${tags[$idx]}"
  else
    rc=$?
    log "[FAIL] ${tags[$idx]} exit=$rc (see $LOGS/${tags[$idx]}.out)"
    printf '{"tag":"%s","end":"%s","exit":%d}\n' \
      "${tags[$idx]}" "$(date -Iseconds)" "$rc" >> "$STATUS"
    fail=1
  fi
done

log "all jobs finished (fail=$fail)"
exit $fail
