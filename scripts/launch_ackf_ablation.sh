#!/usr/bin/env bash
# ACKF core-module ablations on 5m_vs_6m, one seed each, run concurrently.
#
# Arms:
#   qmix_offline_init    QMIX initialised from (theta_off, phi_off); no frozen
#                        encoder, adapter or gate.
#   ackf_no_adapter      frozen encoder + gate, delta_q_K = q_off.
#   ackf_no_gate         frozen encoder + adapter, q = q_on + delta_q_K.
#   ackf_unfrozen_encoder
#                        full structure, but F_off receives online TD gradients.
#
# The Full ACKF reference is scripts/launch_ackf.sh (same map, seed 1).
#
# Usage:
#   nohup bash scripts/launch_ackf_ablation.sh > logs/ackf_ablation_master.log 2>&1 &
# Optional:
#   SEED=1 GPUS="2 3" ARMS="..." ROOT=... LOGS=... PLAN_ONLY=1
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

SEED="${SEED:-1}"
GPUS="${GPUS:-2 3}"
ARMS="${ARMS:-qmix_offline_init ackf_no_adapter ackf_no_gate ackf_unfrozen_encoder}"
MAP="${MAP:-5m_vs_6m}"
ROOT="${ROOT:-checkpoints/ackf_ablation_${MAP}}"
LOGS="${LOGS:-logs/ackf_ablation_${MAP}}"
STATUS="$ROOT/job_status.jsonl"
BASE_CFG="configs/ackf_${MAP}.yaml"
OFFLINE_CKPT="checkpoints/ovmse_batch_20261014/${MAP}_offline/ovmse_paper_seed1_best.pt"

mkdir -p "$ROOT" "$LOGS"
log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$ROOT/launcher.log"; }

if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "missing shared offline checkpoint: $OFFLINE_CKPT" >&2
  exit 2
fi
if [[ ! -f "$BASE_CFG" ]]; then
  echo "missing base config: $BASE_CFG" >&2
  exit 2
fi

arm_overrides() {  # arm_overrides <arm>
  case "$1" in
    qmix_offline_init)
      echo "learner_type=ovmse_paper transfer.load_offline_nets=false"
      ;;
    ackf_no_adapter)
      echo "ackf.variant=no_adapter"
      ;;
    ackf_no_gate)
      echo "ackf.variant=no_gate"
      ;;
    ackf_unfrozen_encoder)
      echo "ackf.variant=unfrozen_encoder"
      ;;
    *)
      echo "unknown arm: $1" >&2
      return 1
      ;;
  esac
}

arm_final_stem() {  # arm_final_stem <arm>
  case "$1" in
    qmix_offline_init) echo "ovmse_paper_online_full_td_seed" ;;
    *) echo "ackf_online_full_td_seed" ;;
  esac
}

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  i=0
  n_gpus=$(echo $GPUS | wc -w)
  for arm in $ARMS; do
    gpu=$(echo $GPUS | cut -d' ' -f $(( i % n_gpus + 1 )))
    echo "gpu=$gpu arm=$arm seed=$SEED dir=$ROOT/$arm/seed$SEED"
    i=$((i + 1))
  done
  exit 0
fi

log "code HEAD: $(git rev-parse --short HEAD)"
log "seed=$SEED gpus=[$GPUS] arms=[$ARMS] map=$MAP"

pids=()
tags=()

start_job() {  # start_job <gpu> <arm> <seed>
  local gpu="$1" arm="$2" seed="$3"
  local dir="$ROOT/$arm/seed$seed"
  local tag="${arm}_${MAP}_seed${seed}"
  local stem
  stem=$(arm_final_stem "$arm")
  local final="$dir/${stem}${seed}.pt"
  local overrides
  overrides=$(arm_overrides "$arm") || return 1

  if [[ -f "$final" ]]; then
    log "[skip] $tag (finished: $final)"
    return 0
  fi
  mkdir -p "$dir"
  printf '{"tag":"%s","gpu":"%s","arm":"%s","map":"%s","seed":%d,"start":"%s"}\n' \
    "$tag" "$gpu" "$arm" "$MAP" "$seed" "$(date -Iseconds)" >> "$STATUS"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$BASE_CFG" \
    phase=online \
    "seed=$seed" \
    "system_name=ACKF $arm $MAP seed$seed" \
    "checkpoint_dir=$dir" \
    "log_path=$dir/online.jsonl" \
    $overrides \
    > "$LOGS/${tag}.out" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOGS/${tag}.pid"
  pids+=("$pid")
  tags+=("$tag")
  log "[start] gpu=$gpu tag=$tag pid=$pid dir=$dir overrides=[$overrides]"
}

i=0
n_gpus=$(echo $GPUS | wc -w)
for arm in $ARMS; do
  gpu=$(echo $GPUS | cut -d' ' -f $(( i % n_gpus + 1 )))
  start_job "$gpu" "$arm" "$SEED"
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
