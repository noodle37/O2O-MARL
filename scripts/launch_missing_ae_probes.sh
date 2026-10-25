#!/usr/bin/env bash
# Fill the remaining Insight A / Insight E cells on 5m_vs_6m.
#
# Insight A (teammate-shift transplant probe, scripts/run_pv_exp2a.py):
#   * arm A x seeds 2,3 at t=1M            (arm A's version ladder is seed-1 only)
#   * arms B,C,D x seeds 1,2,3 at 250k/500k (cross-arm only has t=1M)
#   * all@off baseline (offline team, eps=0.0, 200 episodes)
#   The random-shift family is NOT re-run: it puts every agent on the offline
#   policy, so the live net is unused and the result is arm/seed independent
#   (checkpoints/pv_exp2a/5m_vs_6m/seed1_random.json already covers it).
#
# Insight E (replay gain, scripts/run_replay_gain.py):
#   * arm D x seeds 1,2,3 at t=1M           (arm D was never probed)
#   * arms B,C,D x seeds 1,2,3 at t=250k, lr=1e-4, u500/e200 (matches arm A's
#     250k protocol so U is comparable across arms)
#   * arms B,C x seeds 1,2,3 at t=1M with the original lr=1e-3 protocol
#     (u100/e128 and u500/e64, matching arm A's main-report cells)
#
# Usage:
#   GPUS="1 2 3" nohup bash scripts/launch_missing_ae_probes.sh \
#     > logs/missing_ae_20261024_master.log 2>&1 &
#
# Optional: GPUS="1 2 3"  ROOT=...  LOGS=...  PLAN_ONLY=1
#
# GPU 0 is never used (another user's long-running job).  Probe jobs are CPU
# bound with <1.2 GB GPU each, so they share GPUs 1-3 with the OVMSE training
# runs that are already in flight; per-GPU queues keep at most one probe per GPU.
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

GPUS="${GPUS:-1 2 3}"
MAP=5m_vs_6m
SEEDS="${SEEDS:-1 2 3}"
ROOT="${ROOT:-checkpoints/missing_ae_20261024}"
LOGS="${LOGS:-logs/missing_ae_20261024}"
STATUS="$ROOT/job_status.jsonl"
TRAIN_ROOT="checkpoints/pv_exp1/$MAP"
mkdir -p "$ROOT" "$LOGS"

log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$ROOT/launcher.log"; }

for gpu in $GPUS; do
  [[ "$gpu" == "0" ]] && { echo "FATAL: GPU 0 is reserved for another user" >&2; exit 2; }
done

# ---- job matrix -------------------------------------------------------------
# line format: kind|arm|seed|t|updates|eval_eps|lr|tag|out
QUEUE="$(mktemp /tmp/missing_ae.XXXXXX)"
trap 'rm -f "$QUEUE" "$QUEUE".lock' EXIT
: > "$QUEUE".lock

add_a() {  # add_a <arm> <seed> <t>
  local arm="$1" seed="$2" t="$3" short tag out
  case "$t" in
    1000000) short=1M ;; 250000) short=250k ;; 500000) short=500k ;; *) short=$t ;;
  esac
  tag="A_arm${arm}_t${short}_seed${seed}"
  out="$ROOT/${tag}.json"
  printf 'A|%s|%s|%s|0|0||%s|%s\n' "$arm" "$seed" "$t" "$tag" "$out" >> "$QUEUE"
}

add_e() {  # add_e <arm> <seed> <t> <updates> <eval_eps> <lr>
  local arm="$1" seed="$2" t="$3" u="$4" e="$5" lr="$6" short lrs tag out
  case "$t" in
    1000000) short=1M ;; 250000) short=250k ;; *) short=$t ;;
  esac
  case "$lr" in
    1e-4) lrs=1e4 ;; 1e-3) lrs=1e3 ;; *) lrs=$lr ;;
  esac
  tag="E_${arm}_t${short}_lr${lrs}_u${u}_e${e}_seed${seed}"
  out="$ROOT/${tag}.json"
  printf 'E|%s|%s|%s|%s|%s|%s|%s|%s\n' "$arm" "$seed" "$t" "$u" "$e" "$lr" "$tag" "$out" >> "$QUEUE"
}

# Insight A
for seed in 2 3; do add_a A "$seed" 1000000; done
for arm in B C D; do
  for seed in $SEEDS; do
    add_a "$arm" "$seed" 250000
    add_a "$arm" "$seed" 500000
  done
done
printf 'A|OFF|1|0|0|0||A_all_off_e200|%s\n' "$ROOT/A_all_off_e200.json" >> "$QUEUE"

# Insight E
for seed in $SEEDS; do add_e D "$seed" 1000000 200 128 1e-4; done
for arm in B C D; do
  for seed in $SEEDS; do add_e "$arm" "$seed" 250000 500 200 1e-4; done
done
for arm in B C; do
  for seed in $SEEDS; do
    add_e "$arm" "$seed" 1000000 100 128 1e-3
    add_e "$arm" "$seed" 1000000 500 64 1e-3
  done
done

n_jobs=$(grep -c . "$QUEUE")
log "code HEAD: $(git rev-parse --short HEAD)"
log "map=$MAP seeds=[$SEEDS] gpus=[$GPUS] jobs=$n_jobs"
log "outputs: $ROOT"

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  log "PLAN_ONLY=1, nothing launched"
  cat "$QUEUE"
  exit 0
fi

claim_job() {  # claim_job <queue>
  local queue="$1" line=""
  exec 9>"$queue".lock
  flock 9
  line="$(head -n 1 "$queue")"
  if [[ -n "$line" ]]; then
    tail -n +2 "$queue" > "$queue".tmp && mv "$queue".tmp "$queue"
  fi
  exec 9>&-
  printf '%s' "$line"
}

worker() {  # worker <gpu> <wid>
  local gpu="$1" wid="$2" line kind arm seed t u e lr tag out rc short
  while true; do
    line="$(claim_job "$QUEUE")"
    [[ -z "${line:-}" ]] && break
    IFS='|' read -r kind arm seed t u e lr tag out <<< "$line"
    if [[ -s "$out" ]]; then
      log "[skip] $tag (output exists)"
      continue
    fi
    if [[ "$kind" == "A" && "$arm" != "OFF" && "$t" != "0" ]]; then
      if [[ ! -f "$TRAIN_ROOT/${arm}_seed${seed}/periodic/ovm_diag_ckpt_t${t}.pt" ]]; then
        log "[skip] $tag (missing checkpoint ${arm}_seed${seed} t=$t)"
        continue
      fi
    fi
    if [[ "$kind" == "E" ]]; then
      if [[ ! -f "$TRAIN_ROOT/${arm}_seed${seed}/periodic/ovm_diag_ckpt_t${t}.pt" ]]; then
        log "[skip] $tag (missing checkpoint ${arm}_seed${seed} t=$t)"
        continue
      fi
    fi
    log "[start] gpu=$gpu worker=$wid $tag"
    printf '{"tag":"%s","gpu":"%s","start":"%s"}\n' "$tag" "$gpu" "$(date -Iseconds)" >> "$STATUS"
    rc=0
    if [[ "$kind" == "A" ]]; then
      if [[ "$arm" == "OFF" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_pv_exp2a.py --map "$MAP" \
          --seed 1 --skip-natural --eps-values 0.0 --episodes 200 --save-episodes \
          --out "$out" > "$LOGS/${tag}.out" 2>&1 || rc=$?
      else
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_pv_exp2a.py --map "$MAP" \
          --seed "$seed" --arm "$arm" --t-steps "$t" --episodes 200 --save-episodes \
          --out "$out" > "$LOGS/${tag}.out" 2>&1 || rc=$?
      fi
    else
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_replay_gain.py --seed "$seed" \
        --arm "$arm" --t-steps "$t" --updates "$u" --batch-eps 32 \
        --eval-episodes "$e" --lr "$lr" --save-episodes \
        --out "$out" > "$LOGS/${tag}.out" 2>&1 || rc=$?
    fi
    if [[ $rc -eq 0 && -s "$out" ]]; then
      log "[done] gpu=$gpu $tag exit=0"
    else
      log "[FAIL] gpu=$gpu $tag exit=$rc; see $LOGS/${tag}.out"
    fi
    printf '{"tag":"%s","gpu":"%s","end":"%s","exit":%s}\n' \
      "$tag" "$gpu" "$(date -Iseconds)" "$rc" >> "$STATUS"
  done
  log "[worker=$wid gpu=$gpu] queue empty"
}

i=0
for gpu in $GPUS; do
  i=$((i + 1))
  worker "$gpu" "$i" &
  log "launched worker $i on gpu=$gpu pid=$!"
done
wait
log "[all done]"
n_fail=$(grep -c '"exit":[1-9]' "$STATUS" 2>/dev/null || echo 0)
log "failures: $n_fail"
