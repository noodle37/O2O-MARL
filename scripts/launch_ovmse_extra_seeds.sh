#!/usr/bin/env bash
# Extra OVMSE (arm C) seeds on 5m_vs_6m, one seed per GPU.
#
# Arm C is the full OVMSE online recipe (transfer.mode=ovmse, use_ovm=true on
# both the frozen offline net and the online net); see scripts/run_pv_exp1.sh.
# The protocol is identical to seeds 1-3: same shared offline checkpoint
# (ovmse_paper_seed1_best), same online recipe, same sequential exploration,
# same fixed evaluation seed, same 1M-step budget, periodic checkpoints every
# 25k steps.  The only thing that changes is the online seed.
#
# Usage:
#   SEEDS="4 5 6" GPUS="1 2 3" nohup bash scripts/launch_ovmse_extra_seeds.sh \
#     > logs/ovmse_extra_seeds_20261024_master.log 2>&1 &
#
# Optional:
#   SEEDS="7 8"  GPUS="2 3"  ROOT=...  LOGS=...  PLAN_ONLY=1
#
# GPU 0 is never used: it hosts another user's long-running job.
set -uo pipefail

cd "$(dirname "$0")/.."
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"

SEEDS="${SEEDS:-4 5 6}"
GPUS="${GPUS:-1 2 3}"
MAP=5m_vs_6m
ARM=C
ROOT="${ROOT:-checkpoints/ovmse_extra_seeds_20261024}"
LOGS="${LOGS:-logs/ovmse_extra_seeds_20261024}"
STATUS="$ROOT/job_status.jsonl"
TRAIN_ROOT="checkpoints/pv_exp1/$MAP"
OFFLINE_CKPT="checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt"

mkdir -p "$ROOT" "$LOGS"

log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$ROOT/launcher.log"; }

train_done() {  # train_done <dir>
  [[ -f "$1/periodic/ovm_diag_ckpt_t1000000.pt" && -f "$1/online.jsonl" ]]
}

n_seeds=$(echo $SEEDS | wc -w)
n_gpus=$(echo $GPUS | wc -w)
if (( n_seeds > n_gpus )); then
  echo "FATAL: $n_seeds seeds but only $n_gpus GPUs" >&2
  exit 2
fi
for gpu in $GPUS; do
  if [[ "$gpu" == "0" ]]; then
    echo "FATAL: GPU 0 is reserved for another user's job" >&2
    exit 2
  fi
done
if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: shared offline checkpoint missing: $OFFLINE_CKPT" >&2
  exit 3
fi

log "code HEAD: $(git rev-parse --short HEAD)"
log "map=$MAP arm=$ARM (OVMSE) seeds=[$SEEDS] gpus=[$GPUS] t_max=1000000"
log "offline ckpt: $OFFLINE_CKPT"
log "run dirs: $TRAIN_ROOT/${ARM}_seed{${SEEDS// /,}}"
log "config sha: $(sha256sum configs/pv_base_5m_vs_6m.yaml | cut -c1-16)"

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  log "PLAN_ONLY=1, nothing launched"
  exit 0
fi

i=0
for seed in $SEEDS; do
  i=$((i + 1))
  gpu=$(echo "$GPUS" | tr ' ' '\n' | sed -n "${i}p")
  (
    dir="$TRAIN_ROOT/${ARM}_seed${seed}"
    if train_done "$dir"; then
      log "[skip] train_${ARM}_seed${seed} (finished)"
      exit 0
    fi
    log "[start] gpu=$gpu seed=$seed arm=$ARM"
    printf '{"tag":"train_%s_seed%s","gpu":"%s","start":"%s","cmd":"bash scripts/run_pv_exp1.sh %s %s %s %s"}\n' \
      "$ARM" "$seed" "$gpu" "$(date -Iseconds)" "$MAP" "$ARM" "$seed" "$gpu" >> "$STATUS"
    bash scripts/run_pv_exp1.sh "$MAP" "$ARM" "$seed" "$gpu" \
      > "$LOGS/train_${ARM}_seed${seed}.out" 2>&1
    rc=$?
    if train_done "$dir"; then
      log "[done] gpu=$gpu seed=$seed arm=$ARM exit=$rc verified"
    else
      log "[FAIL] gpu=$gpu seed=$seed arm=$ARM exit=$rc; see $LOGS/train_${ARM}_seed${seed}.out"
    fi
    printf '{"tag":"train_%s_seed%s","gpu":"%s","end":"%s","exit":%s}\n' \
      "$ARM" "$seed" "$gpu" "$(date -Iseconds)" "$rc" >> "$STATUS"
  ) &
  log "launched worker $i on gpu=$gpu pid=$!"
done

wait
log "[all done]"
