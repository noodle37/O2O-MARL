#!/bin/bash
# 5m_vs_6m ranking-loss fork experiment: one seed.
#
#   1. run the UOT parent until the fork point and freeze a full trainer snapshot
#   2. resume three branches from that snapshot, differing only in the ranking
#      loss weight after the fork (continue / off / decay)
#   3. evaluate the fork checkpoint and every branch endpoint with 640 episodes
#
# Usage: run_uot_fork_5m_vs_6m.sh <seed> [fork_at] [t_max]
#   fork_at  env steps at which to fork          (default 1000000)
#   t_max    total online budget in env steps    (default 3000000)
#
# Designed for nohup: every step is timestamped into the seed log, and a failure
# aborts the seed instead of silently continuing with a stale snapshot.
set -euo pipefail

SEED="${1:?usage: $0 <seed> [fork_at] [t_max]}"
FORK_AT="${2:-1000000}"
T_MAX="${3:-3000000}"

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

# GPU policy: this box shares cards 0 and 2 with other jobs, so the fork experiment
# is confined to the cards listed in FORK_ALLOWED_GPUS. FORK_GPU is the parent's card
# and the default for branches; FORK_BRANCH_GPU_<NAME> pins an individual branch.
FORK_ALLOWED_GPUS="${FORK_ALLOWED_GPUS:-1 3}"
FORK_GPU="${FORK_GPU:-1}"

check_gpu() {
  local want="$1" g
  for g in $FORK_ALLOWED_GPUS; do
    [[ "$g" == "$want" ]] && return 0
  done
  echo "FATAL: GPU '$want' is not in FORK_ALLOWED_GPUS='$FORK_ALLOWED_GPUS'" >&2
  exit 1
}

branch_gpu() {
  local name="$1"
  local var="FORK_BRANCH_GPU_$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')"
  local gpu="${!var:-$FORK_GPU}"
  check_gpu "$gpu"
  printf '%s' "$gpu"
}

# FORK_RUN_DIR exists so a formal run can use its own directory: reusing a directory
# that already holds branch logs would make _seed_branch_log refuse to re-inject the
# parent prefix, and a stale manifest could be picked instead of the new fork point.
RUN_DIR="${FORK_RUN_DIR:-checkpoints/uot_fork_5m_vs_6m_seed${SEED}}"
mkdir -p "$RUN_DIR"
SEED_LOG="$RUN_DIR/driver.log"

log() { echo "=== [$(date '+%F %T')] $* ===" | tee -a "$SEED_LOG"; }

# The offline checkpoint each seed's parent starts from (same file the plain UOT
# run uses, so the pre-fork curve stays comparable with existing UOT runs).
# Every seed uses the FINAL checkpoint of a complete 10M-step offline run, so the
# three seeds are started from the same kind of artefact. (seed 2 previously used
# a 902k-step truncated "_best" file; that run was abandoned and re-done properly.)
case "$SEED" in
  1) OFFLINE_CKPT="checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1.pt" ;;
  *) OFFLINE_CKPT="checkpoints/ovmse_paper_5m_vs_6m_seed${SEED}/ovmse_paper_seed${SEED}.pt" ;;
esac
if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: offline checkpoint missing: $OFFLINE_CKPT" >&2
  echo "       run the offline phase for seed ${SEED} first" >&2
  exit 1
fi

PARENT_CFG=configs/uot_5m_vs_6m_fork_parent.yaml
BRANCH_CFG=configs/uot_5m_vs_6m_fork_branch.yaml
COMMON=(
  "seed=$SEED"
  "task.dataset=Medium"
  "checkpoint_dir=$RUN_DIR"
  "checkpoint_path=$OFFLINE_CKPT"
)

# ---------------------------------------------------------------- 1. parent run
# Validate the whole GPU plan before the multi-hour parent run starts.
check_gpu "$FORK_GPU"
for _b in continue off decay; do
  branch_gpu "$_b" >/dev/null
done
log "gpu plan: parent=$FORK_GPU allowed='$FORK_ALLOWED_GPUS' parallel=${FORK_BRANCH_PARALLEL:-0}"

log "parent start seed=$SEED fork_at=$FORK_AT t_max=$T_MAX ckpt=$OFFLINE_CKPT"
CUDA_VISIBLE_DEVICES="$FORK_GPU" "$PY" train.py --config "$PARENT_CFG" \
  "${COMMON[@]}" \
  "fork_at=$FORK_AT" \
  "online.t_max=$T_MAX" \
  "log_path=$RUN_DIR/parent.jsonl" 2>&1 | tee -a "$SEED_LOG"

# The parent exits at the first eval point with t_env >= fork_at; its manifest
# records the exact fork step, which the branches must resume from.
MANIFEST="$(ls -t "$RUN_DIR"/fork_snapshot_main_t*.json 2>/dev/null | head -1 || true)"
if [[ -z "${MANIFEST}" || ! -f "${MANIFEST}" ]]; then
  echo "FATAL: no fork manifest in $RUN_DIR (parent did not reach fork_at=$FORK_AT)" >&2
  exit 1
fi
FORK_T="$("$PY" -c "import json;print(json.load(open('$MANIFEST'))['t_env'])")"
SNAP="$("$PY" -c "import json;print(json.load(open('$MANIFEST'))['snapshot'])")"
if [[ ! -f "$SNAP" ]]; then
  echo "FATAL: snapshot file missing: $SNAP" >&2
  exit 1
fi
REMAIN=$((T_MAX - FORK_T))
log "fork point t_env=$FORK_T remaining=$REMAIN snapshot=$SNAP"

# ------------------------------------------------------------- 2. three branches
# Serial by default: branches only need identical state at the fork instant (the
# snapshot guarantees that), so serial runs cost wall-clock but not validity.
# Set FORK_BRANCH_PARALLEL=1 to run all three at once, spread over the allowed cards.
run_branch() {
  local name="$1"; shift
  local gpu
  gpu="$(branch_gpu "$name")"
  log "branch $name start gpu=$gpu eta_overrides: $*"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$BRANCH_CFG" \
    "${COMMON[@]}" \
    "resume_fork=$SNAP" \
    "fork_branch='$name'" \
    "fork_parent_log=$RUN_DIR/parent.jsonl" \
    "online.t_max=$T_MAX" \
    "log_path=$RUN_DIR/branch_${name}.jsonl" \
    "$@" 2>&1 | tee -a "$SEED_LOG"
  log "branch $name done"
}

# Validate every GPU up front so a misconfiguration fails before the 4-6h parent run
# instead of after it.
check_gpu "$FORK_GPU"
for _b in continue off decay; do
  branch_gpu "$_b" >/dev/null
done
log "gpu plan: parent=$FORK_GPU allowed='$FORK_ALLOWED_GPUS' parallel=${FORK_BRANCH_PARALLEL:-0}"

if [[ "${FORK_BRANCH_PARALLEL:-0}" == "1" ]]; then
  run_branch continue &
  run_branch off "system.uot_eta_drop_at=$FORK_T" &
  run_branch decay "system.uot_eta_decay_start=$FORK_T" "system.uot_eta_decay_steps=$REMAIN" &
  wait
else
  run_branch continue
  run_branch off "system.uot_eta_drop_at=$FORK_T"
  run_branch decay "system.uot_eta_decay_start=$FORK_T" "system.uot_eta_decay_steps=$REMAIN"
fi

# ------------------------------------------------------------------- 3. eval 640
# Paired endpoint evaluation on the same env seed as training: 640 episodes gives a
# Wilson half-width near 0.035 at p~0.75, which is what separates the branches.
EVAL_COMMON=(
  "seed=$SEED"
  "task.dataset=Medium"
  "checkpoint_dir=$RUN_DIR"
  "checkpoint_path=$OFFLINE_CKPT"
)
log "eval640 fork checkpoint"
"$PY" scripts/eval_640.py --config "$BRANCH_CFG" "${EVAL_COMMON[@]}" \
  --checkpoint "$SNAP" --episodes 640 --chunk 32 \
  --out "$RUN_DIR/eval640_fork.jsonl" --tag "fork_t${FORK_T}" 2>&1 | tee -a "$SEED_LOG"

for NAME in continue off decay; do
  CKPT="$RUN_DIR/uot_online_fork_resume_${NAME}_seed${SEED}.pt"
  if [[ ! -f "$CKPT" ]]; then
    log "WARNING missing endpoint checkpoint for $NAME: $CKPT"
    continue
  fi
  "$PY" scripts/eval_640.py --config "$BRANCH_CFG" "${EVAL_COMMON[@]}" \
    --checkpoint "$CKPT" --episodes 640 --chunk 32 \
    --out "$RUN_DIR/eval640_${NAME}.jsonl" --tag "${NAME}_seed${SEED}" 2>&1 | tee -a "$SEED_LOG"
done

log "ALL DONE seed=$SEED RUN_DIR=$RUN_DIR"
