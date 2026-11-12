#!/usr/bin/env bash
# 2s3z three-algorithm comparison on three cards, two jobs per card.
#
#   GPU1  MaROAD   Medium + Medium-Replay   (online 1M env steps each)
#   GPU2  OVMSE    Medium + Medium-Replay   (paper recipe: offline 3M -> online 1M)
#   GPU3  MaCal-QL Medium + Medium-Replay   (joint run: 1M online env steps with
#                                            50% offline episodes mixed in every
#                                            batch, trained from scratch)
#
# MaROAD starts from the shared OVMSE offline checkpoint, exactly as
# configs/road_2s3z.yaml. The Medium-Replay arm has no such checkpoint yet, so
# its job waits for the OVMSE Medium-Replay chain to finish its offline phase
# and reuses that frozen checkpoint (offline_pretrained_frozen.pt). This keeps
# the initial weights matched to the offline data for both MaROAD arms.
#
# Modes:
#   SMOKE=1  tiny budgets, same launch paths and GPU layout; used to validate
#            the whole pipeline before the long run.
#   SMOKE=0  full budgets (default).
#
# Usage:
#   SMOKE=1 bash scripts/run_2s3z_three_algos.sh
#   nohup bash scripts/run_2s3z_three_algos.sh \
#     > logs/2s3z_three_algos_master.log 2>&1 &
#
# Overrides: SMOKE ROOT LOGS SEED ROAD_GPU OVMSE_GPU MACAL_GPU SHARED_CKPT
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

SMOKE="${SMOKE:-0}"
ROAD_GPU="${ROAD_GPU:-1}"
OVMSE_GPU="${OVMSE_GPU:-2}"
MACAL_GPU="${MACAL_GPU:-3}"
SEED="${SEED:-1}"
SHARED_CKPT="${SHARED_CKPT:-checkpoints/ovmse_batch_20261014/2s3z_offline/ovmse_paper_seed1_best.pt}"

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "$SMOKE" == "1" ]]; then
  ROOT="${ROOT:-checkpoints/2s3z_three_algos_smoke_${STAMP}}"
  LOGS="${LOGS:-logs/2s3z_three_algos_smoke_${STAMP}}"
  OFFLINE_TMAX=3000
  ONLINE_TMAX=2000
  BATCH=8
  BUFFER=32
  TEST_INTERVAL=1000
  NUM_EVAL=4
  ROAD_PERIOD=2
  ROAD_DELTA=2
  ROAD_WINDOW=4
  WAIT_TIMEOUT=1800
else
  ROOT="${ROOT:-checkpoints/2s3z_three_algos_20261105}"
  LOGS="${LOGS:-logs/2s3z_three_algos_20261105}"
  OFFLINE_TMAX=3000000
  ONLINE_TMAX=1000000
  BATCH=128
  BUFFER=5000
  TEST_INTERVAL=10000
  NUM_EVAL=32
  ROAD_PERIOD=10
  ROAD_DELTA=32
  ROAD_WINDOW=200
  WAIT_TIMEOUT=28800
fi

mkdir -p "$ROOT" "$LOGS"
STATUS="$ROOT/job_status.jsonl"

log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$ROOT/launcher.log"; }

if [[ ! -f "$SHARED_CKPT" ]]; then
  log "FATAL: shared OVMSE Medium checkpoint missing: $SHARED_CKPT"
  exit 2
fi

log "mode=$([[ "$SMOKE" == "1" ]] && echo smoke || echo full) root=$ROOT logs=$LOGS"
log "gpus road=$ROAD_GPU ovmse=$OVMSE_GPU macal=$MACAL_GPU seed=$SEED"
log "budgets offline_tmax=$OFFLINE_TMAX online_tmax=$ONLINE_TMAX batch=$BATCH test_interval=$TEST_INTERVAL"
log "shared MaROAD Medium ckpt: $SHARED_CKPT"

wait_for_file() {  # $1=file $2=timeout_s $3=owner_pid
  local f="$1" timeout_s="$2" owner="$3" elapsed=0
  while [[ ! -f "$f" ]]; do
    if ! kill -0 "$owner" 2>/dev/null; then
      return 1
    fi
    if (( elapsed >= timeout_s )); then
      return 2
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  return 0
}

run_ovmse_chain() {  # $1=gpu $2=dataset $3=tag
  local gpu="$1" dataset="$2" tag="$3"
  local off="$ROOT/ovmse_2s3z_${tag}_offline"
  local on="$ROOT/ovmse_2s3z_${tag}_online"
  local best="$off/ovmse_paper_seed1_best.pt"
  local frozen="$off/offline_pretrained_frozen.pt"
  local out="$LOGS/ovmse_${tag}.out"
  mkdir -p "$off" "$on"
  printf '{"job":"ovmse:%s","phase":"offline","gpu":"%s","start":"%s"}\n' \
    "$tag" "$gpu" "$(date -Iseconds)" >> "$STATUS"

  log "[ovmse:$tag] offline start gpu=$gpu dataset=$dataset dir=$off"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config configs/2s3z.yaml \
    phase=offline \
    "seed=$SEED" "task.dataset=$dataset" \
    "t_max=$OFFLINE_TMAX" "batch_size=$BATCH" "buffer_size=$BUFFER" \
    "test_interval=$TEST_INTERVAL" "num_eval_episodes=$NUM_EVAL" \
    save_best_checkpoint=true \
    "checkpoint_dir=$off" "log_path=$off/offline.jsonl" \
    >> "$out" 2>&1
  local rc=$?
  log "[ovmse:$tag] offline done rc=$rc"

  if (( rc != 0 )); then
    return "$rc"
  fi
  if [[ ! -f "$best" ]]; then
    log "[ovmse:$tag] expected offline best missing: $best"
    return 1
  fi
  cp -p "$best" "$frozen"

  printf '{"job":"ovmse:%s","phase":"online","gpu":"%s","start":"%s"}\n' \
    "$tag" "$gpu" "$(date -Iseconds)" >> "$STATUS"
  log "[ovmse:$tag] online start from=$best dir=$on"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config configs/2s3z.yaml \
    phase=online \
    "seed=$SEED" "task.dataset=$dataset" \
    "online.t_max=$ONLINE_TMAX" "t_max=$ONLINE_TMAX" \
    "batch_size=$BATCH" "buffer_size=$BUFFER" \
    "test_interval=$TEST_INTERVAL" "num_eval_episodes=$NUM_EVAL" \
    "checkpoint_dir=$on" "checkpoint_path=$best" "log_path=$on/online.jsonl" \
    >> "$out" 2>&1
  rc=$?
  log "[ovmse:$tag] online done rc=$rc"
  return "$rc"
}

run_road() {  # $1=gpu $2=dataset $3=tag $4=init_ckpt
  local gpu="$1" dataset="$2" tag="$3" init="$4"
  local dir="$ROOT/road_2s3z_${tag}"
  mkdir -p "$dir"
  printf '{"job":"road:%s","phase":"online","gpu":"%s","start":"%s"}\n' \
    "$tag" "$gpu" "$(date -Iseconds)" >> "$STATUS"
  log "[road:$tag] online start gpu=$gpu dataset=$dataset init=$init dir=$dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config configs/road_2s3z.yaml \
    phase=online \
    "seed=$SEED" "task.dataset=$dataset" \
    "online.t_max=$ONLINE_TMAX" "t_max=$ONLINE_TMAX" \
    "batch_size=$BATCH" "buffer_size=$BUFFER" \
    "test_interval=$TEST_INTERVAL" "num_eval_episodes=$NUM_EVAL" \
    "checkpoint_dir=$dir" "checkpoint_path=$init" "log_path=$dir/online.jsonl" \
    "road.period_steps=$ROAD_PERIOD" "road.delta_batch=$ROAD_DELTA" "road.window=$ROAD_WINDOW" \
    "system_name=MaROAD 2s3z $dataset seed$SEED" \
    >> "$LOGS/road_${tag}.out" 2>&1
  local rc=$?
  log "[road:$tag] done rc=$rc"
  return "$rc"
}

run_macal() {  # $1=gpu $2=dataset $3=tag
  local gpu="$1" dataset="$2" tag="$3"
  local dir="$ROOT/macal_2s3z_${tag}/joint"
  mkdir -p "$dir"
  printf '{"job":"macal:%s","phase":"joint","gpu":"%s","start":"%s"}\n' \
    "$tag" "$gpu" "$(date -Iseconds)" >> "$STATUS"
  log "[macal:$tag] joint start gpu=$gpu dataset=$dataset dir=$dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config configs/macal_2s3z_joint.yaml \
    phase=online \
    "seed=$SEED" "task.dataset=$dataset" \
    "online.t_max=$ONLINE_TMAX" "t_max=$ONLINE_TMAX" \
    "batch_size=$BATCH" "buffer_size=$BUFFER" \
    "test_interval=$TEST_INTERVAL" "num_eval_episodes=$NUM_EVAL" \
    "checkpoint_dir=$dir" "log_path=$dir/online.jsonl" \
    "system_name=MaCal-QL 2s3z $dataset joint seed$SEED" \
    >> "$LOGS/macal_${tag}.out" 2>&1
  local rc=$?
  log "[macal:$tag] joint done rc=$rc"
  return "$rc"
}

# ---------------------------------------------------------------- launch ---
( run_ovmse_chain "$OVMSE_GPU" Medium medium ) & P_OM=$!
( run_ovmse_chain "$OVMSE_GPU" "Medium-Replay" medium-replay ) & P_OR=$!
( run_macal "$MACAL_GPU" Medium medium ) & P_CM=$!
( run_macal "$MACAL_GPU" "Medium-Replay" medium-replay ) & P_CR=$!
( run_road "$ROAD_GPU" Medium medium "$SHARED_CKPT" ) & P_RM=$!

# MaROAD Medium-Replay waits for the OVMSE Medium-Replay offline phase.
(
  frozen="$ROOT/ovmse_2s3z_medium-replay_offline/offline_pretrained_frozen.pt"
  log "[road:medium-replay] waiting for $frozen"
  wait_for_file "$frozen" "$WAIT_TIMEOUT" "$P_OR"
  rc=$?
  if (( rc != 0 )); then
    log "[road:medium-replay] wait failed rc=$rc (ovmse chain exited or timeout)"
    exit 1
  fi
  run_road "$ROAD_GPU" "Medium-Replay" medium-replay "$frozen"
) & P_RR=$!

log "launched: ovmse_medium=$P_OM ovmse_replay=$P_OR macal_medium=$P_CM macal_replay=$P_CR road_medium=$P_RM road_replay=$P_RR"

rc_total=0
wait_one() {  # $1=pid $2=label
  wait "$1"
  local rc=$?
  log "[exit] $2 rc=$rc"
  if (( rc != 0 )); then
    rc_total=1
  fi
}
wait_one "$P_OM" "ovmse:medium"
wait_one "$P_OR" "ovmse:medium-replay"
wait_one "$P_CM" "macal:medium"
wait_one "$P_CR" "macal:medium-replay"
wait_one "$P_RM" "road:medium"
wait_one "$P_RR" "road:medium-replay"

# -------------------------------------------------------------- verify ----
missing=0
check_file() {  # $1=path $2=label
  if [[ -f "$1" ]]; then
    log "[ok] $2 -> $1"
  else
    log "[MISSING] $2 -> $1"
    missing=1
  fi
}
check_jsonl() {  # $1=path $2=label
  if [[ -s "$1" ]] && grep -q '"evaluation/mean_episode_return"' "$1"; then
    log "[ok] $2 -> $1"
  else
    log "[MISSING/EMPTY] $2 -> $1"
    missing=1
  fi
}

check_file "$ROOT/ovmse_2s3z_medium_offline/ovmse_paper_seed1_best.pt" "OVMSE medium offline best"
check_jsonl "$ROOT/ovmse_2s3z_medium_offline/offline.jsonl" "OVMSE medium offline log"
check_file "$ROOT/ovmse_2s3z_medium_online/ovmse_paper_online_ovmse_seed1.pt" "OVMSE medium online final"
check_jsonl "$ROOT/ovmse_2s3z_medium_online/online.jsonl" "OVMSE medium online log"
check_file "$ROOT/ovmse_2s3z_medium-replay_offline/ovmse_paper_seed1_best.pt" "OVMSE replay offline best"
check_file "$ROOT/ovmse_2s3z_medium-replay_online/ovmse_paper_online_ovmse_seed1.pt" "OVMSE replay online final"
check_jsonl "$ROOT/ovmse_2s3z_medium-replay_online/online.jsonl" "OVMSE replay online log"
check_file "$ROOT/road_2s3z_medium/ovmse_paper_online_full_td_seed1.pt" "MaROAD medium final"
check_jsonl "$ROOT/road_2s3z_medium/online.jsonl" "MaROAD medium log"
check_file "$ROOT/road_2s3z_medium-replay/ovmse_paper_online_full_td_seed1.pt" "MaROAD replay final"
check_jsonl "$ROOT/road_2s3z_medium-replay/online.jsonl" "MaROAD replay log"
check_file "$ROOT/macal_2s3z_medium/joint/macal_online_scratch_seed1.pt" "MaCal medium final"
check_jsonl "$ROOT/macal_2s3z_medium/joint/online.jsonl" "MaCal medium log"
check_file "$ROOT/macal_2s3z_medium-replay/joint/macal_online_scratch_seed1.pt" "MaCal replay final"
check_jsonl "$ROOT/macal_2s3z_medium-replay/joint/online.jsonl" "MaCal replay log"

for f in "$LOGS"/*.out; do
  [[ -f "$f" ]] || continue
  if grep -q "Traceback (most recent call last)" "$f"; then
    log "[ERROR] traceback in $f"
    missing=1
  fi
done

if (( missing != 0 || rc_total != 0 )); then
  log "RESULT: FAILED (missing=$missing job_failures=$rc_total)"
  log "inspect $LOGS and $ROOT/launcher.log"
  exit 1
fi

log "RESULT: OK"
log "artifacts: $ROOT ; per-job stdout: $LOGS"
