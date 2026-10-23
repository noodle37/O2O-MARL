#!/bin/bash
# Matched-protocol arms A-E for additional training seeds, spread over 3 GPUs.
#
# This driver reuses the seed-1 protocol verbatim: same offline checkpoint,
# same config (configs/matched_protocol_5m_vs_6m.yaml), same per-arm overrides
# (copied from scripts/run_matched_protocol_5m_vs_6m.sh), same 3M budget, same
# eval script (scripts/eval_matched_protocol.sh).  The ONLY new thing is the
# seed, which is why these runs are comparable to A-E_seed1 and to each other.
#
# Layout: one worker per GPU; each worker pulls (seed, arm) jobs from a shared
# queue file with flock, so a fast arm never leaves a GPU idle.  Training and
# evaluation are separate stages per job, exactly as in the seed-1 driver.
#
# Usage:
#   SEEDS="2 3" GPUS="1 2 3" RUN_ROOT=checkpoints/matched_protocol_20261020 \
#     nohup bash scripts/run_matched_protocol_multiseed.sh \
#     > logs/matched_protocol_multiseed_master.log 2>&1 &
#
# Optional:
#   OFFLINE_CKPT  offline checkpoint (default: the seed-1 matched-protocol one)
#   ARMS          arms to run (default "A B C D E")
#   PLAN_ONLY=1   print the job list and exit
set -uo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

SEEDS="${SEEDS:-2 3}"
GPUS="${GPUS:-1 2 3}"
ARMS="${ARMS:-A B C D E}"
RUN_ROOT="${RUN_ROOT:-checkpoints/matched_protocol_20261020}"
T_MAX="${T_MAX:-3000000}"
TEST_INTERVAL="${TEST_INTERVAL:-10000}"
OFFLINE_CKPT="${OFFLINE_CKPT:-checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt}"
CFG=configs/matched_protocol_5m_vs_6m.yaml
EVAL_EPISODES="${EVAL_EPISODES:-640}"
EVAL_CHUNK="${EVAL_CHUNK:-32}"

if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: offline checkpoint missing: $OFFLINE_CKPT" >&2
  exit 1
fi

# ---- job queue --------------------------------------------------------------
QUEUE="$(mktemp /tmp/matched_multiseed_queue.XXXXXX)"
trap 'rm -f "$QUEUE" "$QUEUE".lock' EXIT
: > "$QUEUE".lock

# Long arms first so no GPU idles at the tail: A/B are the joint-ranking arms
# (~25h under 3-way GPU sharing in seed 1), C/D/E the OVMSE-learner arms.
for seed in $SEEDS; do
  for arm in A B C D E; do
    case " $ARMS " in *" $arm "*) echo "$seed $arm" >> "$QUEUE" ;; esac
  done
done

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  echo "offline ckpt: $OFFLINE_CKPT"
  echo "run root:     $RUN_ROOT"
  echo "seeds:        $SEEDS"
  echo "gpus:         $GPUS"
  echo "jobs ($(wc -l < "$QUEUE")):"
  cat "$QUEUE"
  exit 0
fi

echo "=== matched protocol multi-seed ==="
echo "code HEAD: $(git rev-parse --short HEAD)"
echo "code diff sha256: $(git diff | sha256sum | cut -d' ' -f1)"
echo "offline ckpt: $OFFLINE_CKPT"
echo "seeds=[$SEEDS] arms=[$ARMS] gpus=[$GPUS] t_max=$T_MAX"
echo "jobs: $(wc -l < "$QUEUE")"

arm_overrides() {
  case "$1" in
    # Identical to scripts/run_matched_protocol_5m_vs_6m.sh.
    A) printf '%s\n' "learner_type=uot" "transfer.mode=none" "system.uot_eta=0.1" "system.mixing_ratio=0.0" ;;
    B) printf '%s\n' "learner_type=uot" "transfer.mode=none" "system.uot_eta=0.0" "system.mixing_ratio=0.0" ;;
    C) printf '%s\n' "learner_type=ovmse_paper" "transfer.mode=ovmse" "system.mixing_ratio=0.1" "online.use_ovm=true" ;;
    D) printf '%s\n' "learner_type=ovmse_paper" "transfer.mode=ovmse" "system.mixing_ratio=0.0" "online.use_ovm=true" ;;
    E) printf '%s\n' "learner_type=ovmse_paper" "transfer.mode=none" "system.mixing_ratio=0.1" "online.use_ovm=false" ;;
  esac
}

has_summary() { [[ -f "$1" ]] && grep -q '"checkpoint"' "$1"; }

claim_job() {
  # Atomically pop the first line of the queue.
  local line=""
  exec 9>"$QUEUE".lock
  flock 9
  line="$(head -n 1 "$QUEUE")"
  if [[ -n "$line" ]]; then
    tail -n +2 "$QUEUE" > "$QUEUE".tmp && mv "$QUEUE".tmp "$QUEUE"
  fi
  exec 9>&-
  printf '%s' "$line"
}

worker() {
  local gpu="$1" wid="$2"
  local seed arm dir final_ckpt rc
  while true; do
    read -r seed arm < <(claim_job)
    [[ -z "${seed:-}" ]] && break
    dir="$RUN_ROOT/${arm}_seed${seed}"
    mkdir -p "$dir"

    if has_summary "$dir/eval640_final.jsonl"; then
      echo "=== [$(date '+%F %T')] gpu=$gpu SKIP ${arm}_seed${seed} (final eval already present)"
      continue
    fi

    echo "=== [$(date '+%F %T')] gpu=$gpu START ${arm}_seed${seed} train"

    mapfile -t extra < <(arm_overrides "$arm")
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$CFG" \
      seed="$seed" task.dataset=Medium \
      checkpoint_dir="$dir" checkpoint_path="$OFFLINE_CKPT" \
      log_path="$dir/online.jsonl" \
      online.t_max="$T_MAX" t_max="$T_MAX" \
      online.test_interval="$TEST_INTERVAL" test_interval="$TEST_INTERVAL" \
      "${extra[@]}" > "logs/matched_protocol_seed${seed}_${arm}.out" 2>&1
    rc=$?
    echo "=== [$(date '+%F %T')] gpu=$gpu ${arm}_seed${seed} train done rc=$rc"

    final_ckpt="$(ls -1 "$dir"/*_online_*_seed${seed}.pt 2>/dev/null | head -1)"
    if [[ $rc -ne 0 || -z "$final_ckpt" ]]; then
      echo "=== [$(date '+%F %T')] gpu=$gpu ${arm}_seed${seed} FAILED (rc=$rc, ckpt='$final_ckpt'); see logs/matched_protocol_seed${seed}_${arm}.out"
      continue
    fi

    echo "=== [$(date '+%F %T')] gpu=$gpu START ${arm}_seed${seed} eval640"
    A_GPU="$gpu" B_GPU="$gpu" C_GPU="$gpu" D_GPU="$gpu" E_GPU="$gpu" \
      SEED="$seed" RUN_ROOT="$RUN_ROOT" ARMS="$arm" \
      EVAL_EPISODES="$EVAL_EPISODES" EVAL_CHUNK="$EVAL_CHUNK" \
      bash scripts/eval_matched_protocol.sh > "logs/matched_protocol_seed${seed}_${arm}_eval.out" 2>&1
    rc=$?
    echo "=== [$(date '+%F %T')] gpu=$gpu ${arm}_seed${seed} eval done rc=$rc"
  done
  echo "=== [$(date '+%F %T')] gpu=$gpu worker $wid queue empty, exiting"
}

i=0
for gpu in $GPUS; do
  i=$((i + 1))
  worker "$gpu" "$i" &
  echo "launched worker $i on gpu=$gpu pid=$!"
done

wait
echo "=== [$(date '+%F %T')] ALL WORKERS DONE ==="
remaining="$(wc -l < "$QUEUE")"
if [[ "$remaining" != "0" ]]; then
  echo "WARNING: $remaining job(s) left unclaimed" >&2
fi
