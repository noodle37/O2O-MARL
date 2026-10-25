#!/usr/bin/env bash
# Overflow training workers for the seed-replication batch.
#
# The main launcher (scripts/launch_seed_replication.sh) is SIGSTOPped while
# this runs, so it cannot start the A/E evaluation stage before these jobs and
# the original three training jobs finish.  We drain the remaining train queue
# here (the original workers are busy, so they will see an empty queue when
# they finish), run the jobs in parallel, wait for the original workers to
# become zombies, then SIGCONT the launcher.
#
# Usage:
#   LAUNCHER_PID=... ORIGINAL_WORKERS="pid pid pid" \
#     nohup setsid bash scripts/seed_replication_overflow.sh \
#     > logs/extra_seeds_20261024/overflow_master.out 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.."
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

ROOT="${ROOT:-checkpoints/extra_seeds_20261024}"
LOGS="${LOGS:-logs/extra_seeds_20261024}"
STATUS="$ROOT/job_status.jsonl"
MAP=5m_vs_6m
TRAIN_ROOT=checkpoints/pv_exp1/$MAP
LAUNCHER_PID="${LAUNCHER_PID:?missing LAUNCHER_PID}"
ORIGINAL_WORKERS="${ORIGINAL_WORKERS:-}"

mkdir -p "$ROOT" "$LOGS"

log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$ROOT/launcher.log"; }

resume_launcher() {
  if [[ -n "${LAUNCHER_PID:-}" ]]; then
    kill -CONT "$LAUNCHER_PID" 2>/dev/null || true
    log "[overflow] SIGCONT launcher pid=$LAUNCHER_PID"
  fi
}
trap resume_launcher EXIT

train_done() { [[ -f "$1/periodic/ovm_diag_ckpt_t1000000.pt" && -f "$1/online.jsonl" ]]; }

run_job() {  # run_job <gpu> <tag> <out> <cmd...>
  local gpu="$1" tag="$2" out="$3"
  shift 3
  if [[ -e "$out" ]]; then
    log "[skip] $tag ($out exists)"
    return 0
  fi
  local logf="$LOGS/${tag}.out"
  log "[start] gpu=$gpu tag=$tag"
  printf '{"tag":"%s","gpu":"%s","out":"%s","start":"%s","cmd":%s}\n' \
    "$tag" "$gpu" "$out" "$(date -Iseconds)" \
    "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$logf" 2>&1
  local rc=$?
  printf '{"tag":"%s","end":"%s","exit":%d,"log":"%s"}\n' \
    "$tag" "$(date -Iseconds)" "$rc" "$logf" >> "$STATUS"
  log "[done] tag=$tag exit=$rc"
  return $rc
}

run_one() {  # run_one <gpu> <arm> <seed>
  local gpu="$1" arm="$2" seed="$3"
  local dir="$TRAIN_ROOT/${arm}_seed${seed}"
  if train_done "$dir"; then
    log "[skip] train_${arm}_seed${seed} (finished)"
    return 0
  fi
  run_job "$gpu" "train_${arm}_seed${seed}" "$dir/periodic/ovm_diag_ckpt_t1000000.pt" \
    bash scripts/run_pv_exp1.sh "$MAP" "$arm" "$seed" "$gpu"
  if train_done "$dir"; then
    log "[overflow] train_${arm}_seed${seed} verified"
  else
    log "[overflow] train_${arm}_seed${seed} FAILED; see $LOGS/train_${arm}_seed${seed}.out"
  fi
}

# Remaining queue after the three original jobs:
#   seed 2 arm D, seed 3 arms A B C D.
# Assignment balances the original load (GPU1 A2, GPU2 B2, GPU3 C2):
#   GPU1: D2 + C3; GPU2: A3 + D3; GPU3: B3.
run_one 1 D 2 &
run_one 2 A 3 &
run_one 3 B 3 &
run_one 1 C 3 &
run_one 2 D 3 &
wait

for spec in "A 2" "B 2" "C 2"; do
  set -- $spec
  dir="$TRAIN_ROOT/${1}_seed${2}"
  if train_done "$dir"; then
    log "[overflow] original train_${1}_seed${2} verified"
  else
    log "[overflow] original train_${1}_seed${2} MISSING; see $LOGS/train_${1}_seed${2}.out"
  fi
done

# The original launcher is stopped in `wait`; its workers continue and become
# zombies when they finish.  Do not resume until they are done.
for pid in $ORIGINAL_WORKERS; do
  while [[ -e "/proc/$pid" ]]; do
    state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || echo X)"
    [[ "$state" == "Z" ]] && break
    sleep 30
  done
  log "[overflow] original worker $pid finished"
done

log "[overflow] all overflow jobs and original workers done"
