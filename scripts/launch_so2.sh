#!/usr/bin/env bash
# MaSO2 online runs. One seed, one map per GPU.
#
# Default pairing matches the requested 2s3z / 5m_vs_6m split:
#   GPU 2 -> 2s3z seed 1
#   GPU 3 -> 5m_vs_6m seed 1
#
# Usage:
#   nohup bash scripts/launch_so2.sh > logs/so2_batch_master.log 2>&1 &
#
# Overrides (space-separated, zipped):
#   MAPS="2s3z 5m_vs_6m" GPUS="2 3" SEED=1
set -uo pipefail

cd "$(dirname "$0")/.."
export SC2PATH="${SC2PATH:-/home/jxy/StarCraftII}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

PY="${PY:-/home/jxy/miniconda3/envs/multiAgentRL/bin/python}"
MAPS="${MAPS:-2s3z 5m_vs_6m}"
GPUS="${GPUS:-2 3}"
SEED="${SEED:-1}"
LOGS="${LOGS:-logs/so2_batch}"

mkdir -p "$LOGS"
log() { echo "=== [$(date '+%F %T')] $*"; }

n_maps=$(echo $MAPS | wc -w)
n_gpus=$(echo $GPUS | wc -w)
if (( n_maps != n_gpus )); then
  echo "refusing: $n_maps maps but $n_gpus gpus (zip one map onto one gpu)" >&2
  exit 2
fi

i=1
for map in $MAPS; do
  ckpt="checkpoints/ovmse_batch_20261014/${map}_offline/ovmse_paper_seed1_best.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "missing shared offline checkpoint: $ckpt" >&2
    exit 2
  fi
  gpu=$(echo $GPUS | cut -d' ' -f "$i")
  cfg="configs/so2_${map}.yaml"
  dir="checkpoints/so2_${map}/seed${SEED}"
  tag="so2_${map}_seed${SEED}"
  final="$dir/so2_online_full_td_seed${SEED}.pt"
  if [[ -f "$final" ]]; then
    log "[skip] $tag (finished: $final)"
    i=$((i + 1))
    continue
  fi
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" train.py --config "$cfg" \
    phase=online \
    "seed=$SEED" \
    "system_name=MaSO2 $map seed$SEED" \
    "checkpoint_dir=$dir" \
    "checkpoint_path=$ckpt" \
    "log_path=$dir/online.jsonl" \
    > "$LOGS/${tag}.out" 2>&1 &
  pid=$!
  echo "$pid" > "$LOGS/${tag}.pid"
  log "[start] gpu=$gpu tag=$tag pid=$pid dir=$dir"
  i=$((i + 1))
done

log "launched; waiting"
fail=0
i=1
for map in $MAPS; do
  tag="so2_${map}_seed${SEED}"
  if [[ ! -f "$LOGS/${tag}.pid" ]]; then
    i=$((i + 1))
    continue
  fi
  pid=$(cat "$LOGS/${tag}.pid")
  if wait "$pid"; then
    log "[done] $tag pid=$pid"
  else
    log "[fail] $tag pid=$pid exit=$?"
    fail=1
  fi
  i=$((i + 1))
done
exit $fail
