#!/usr/bin/env bash
# Additional training seeds for PV exp1 arms A-D, followed by the Insight A/E
# probes used in checkpoints/extra_20261024/.
#
# Protocol is identical to seed 1: same offline checkpoint, same config, same
# per-arm overrides, same 1M-step budget.  Only the online seed changes.  Arm D
# needs its seed-specific affine-scaled starting checkpoint, which is built
# before the training queue starts.
#
# Usage:
#   SEEDS="2 3" GPUS="1 2 3" nohup bash scripts/launch_seed_replication.sh \
#     > logs/extra_seeds_20261024_master.log 2>&1 &
#
# Optional:
#   ARMS="A B C D"  ROOT=...  LOGS=...  PLAN_ONLY=1
# GPU 0 is never used (it hosts the long-running 2halfcheetah PV exp1 jobs).
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

SEEDS="${SEEDS:-2 3}"
ARMS="${ARMS:-A B C D}"
GPUS="${GPUS:-1 2 3}"
ROOT="${ROOT:-checkpoints/extra_seeds_20261024}"
LOGS="${LOGS:-logs/extra_seeds_20261024}"
STATUS="$ROOT/job_status.jsonl"
MAP=5m_vs_6m
CFG=configs/pv_base_5m_vs_6m.yaml
SCALED_DIR=checkpoints/pv_exp1_D_ckpt/$MAP
TRAIN_ROOT=checkpoints/pv_exp1/$MAP

mkdir -p "$ROOT" "$LOGS" "$SCALED_DIR"

log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$ROOT/launcher.log"; }

train_done() {  # train_done <dir>
  [[ -f "$1/periodic/ovm_diag_ckpt_t1000000.pt" && -f "$1/online.jsonl" ]]
}

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

# ---- arm D starting checkpoints (sequential, before the queue) -------------
for seed in $SEEDS; do
  scaled="$SCALED_DIR/D_seed${seed}_scaled.pt"
  if [[ -f "$scaled" ]]; then
    log "[skip] prep_D_seed${seed} ($scaled exists)"
    continue
  fi
  gpu="${GPUS%% *}"
  run_job "$gpu" "prep_D_seed${seed}" "$scaled" \
    "$PY" scripts/make_scale_corrected_ckpt.py --config "$CFG" --seed "$seed" \
    --out "$scaled" || log "[warn] prep_D_seed${seed} failed; D will be skipped"
done

EFFECTIVE_ARMS=""
for arm in $ARMS; do
  if [[ "$arm" == "D" ]]; then
    ok=1
    for seed in $SEEDS; do
      [[ -f "$SCALED_DIR/D_seed${seed}_scaled.pt" ]] || ok=0
    done
    [[ "$ok" == "1" ]] || { log "[warn] missing D scaled checkpoint; dropping arm D"; continue; }
  fi
  EFFECTIVE_ARMS="$EFFECTIVE_ARMS $arm"
done
EFFECTIVE_ARMS="${EFFECTIVE_ARMS# }"

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  echo "seeds:  $SEEDS"
  echo "arms:   $EFFECTIVE_ARMS"
  echo "gpus:   $GPUS"
  echo "root:   $ROOT"
  exit 0
fi

log "code HEAD: $(git rev-parse --short HEAD)"
log "seeds=[$SEEDS] arms=[$EFFECTIVE_ARMS] gpus=[$GPUS] t_max=1000000"
log "offline ckpt: checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt"

# ---- training queue --------------------------------------------------------
TQUEUE="$(mktemp /tmp/seed_repl_train.XXXXXX)"
EQUEUE="$(mktemp /tmp/seed_repl_eval.XXXXXX)"
trap 'rm -f "$TQUEUE" "$TQUEUE".lock "$EQUEUE" "$EQUEUE".lock' EXIT
: > "$TQUEUE".lock
: > "$EQUEUE".lock

for seed in $SEEDS; do
  for arm in $EFFECTIVE_ARMS; do
    echo "$seed $arm" >> "$TQUEUE"
  done
done

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

train_worker() {  # train_worker <gpu> <wid>
  local gpu="$1" wid="$2" seed arm dir
  while true; do
    read -r seed arm < <(claim_job "$TQUEUE")
    [[ -z "${seed:-}" ]] && break
    dir="$TRAIN_ROOT/${arm}_seed${seed}"
    if train_done "$dir"; then
      log "[skip] train_${arm}_seed${seed} (finished)"
      continue
    fi
    log "[train] worker=$wid gpu=$gpu seed=$seed arm=$arm"
    run_job "$gpu" "train_${arm}_seed${seed}" "$dir/periodic/ovm_diag_ckpt_t1000000.pt" \
      bash scripts/run_pv_exp1.sh "$MAP" "$arm" "$seed" "$gpu"
    if train_done "$dir"; then
      log "[train] worker=$wid seed=$seed arm=$arm verified"
    else
      log "[train] worker=$wid seed=$seed arm=$arm FAILED; see $LOGS/train_${arm}_seed${seed}.out"
    fi
  done
  log "[train] worker=$wid gpu=$gpu queue empty"
}

i=0
for gpu in $GPUS; do
  i=$((i + 1))
  train_worker "$gpu" "$i" &
  log "launched train worker $i on gpu=$gpu pid=$!"
done
wait
log "[train] all workers done"

# ---- Insight A/E evaluation queue -----------------------------------------
for seed in $SEEDS; do
  for arm in B C D; do
    [[ " $EFFECTIVE_ARMS " == *" $arm "* ]] || continue
    echo "A $seed $arm" >> "$EQUEUE"
  done
  [[ " $EFFECTIVE_ARMS " == *" A "* ]] && {
    echo "E $seed A 250000 500 200" >> "$EQUEUE"
    echo "E $seed A 1000000 500 200" >> "$EQUEUE"
  }
  [[ " $EFFECTIVE_ARMS " == *" B "* ]] && echo "E $seed B 1000000 200 128" >> "$EQUEUE"
  [[ " $EFFECTIVE_ARMS " == *" C "* ]] && echo "E $seed C 1000000 200 128" >> "$EQUEUE"
done

fmt_t() { case "$1" in 1000000) echo 1M ;; 250000) echo 250k ;; *) echo "$1" ;; esac; }

eval_worker() {  # eval_worker <gpu> <wid>
  local gpu="$1" wid="$2" kind seed arm t u e tag out short
  while true; do
    read -r kind seed arm t u e < <(claim_job "$EQUEUE")
    [[ -z "${kind:-}" ]] && break
    if [[ "$kind" == "A" ]]; then
      tag="A_arm${arm}_t1M_seed${seed}"
      out="$ROOT/${tag}.json"
      log "[eval] worker=$wid gpu=$gpu $tag"
      run_job "$gpu" "$tag" "$out" \
        "$PY" scripts/run_pv_exp2a.py --map "$MAP" --seed "$seed" --arm "$arm" \
        --t-steps 1000000 --episodes 200 --skip-random --save-episodes --out "$out"
    else
      short="$(fmt_t "$t")"
      tag="E_${arm}_t${short}_lr1e4_u${u}_e${e}_seed${seed}"
      out="$ROOT/${tag}.json"
      log "[eval] worker=$wid gpu=$gpu $tag"
      run_job "$gpu" "$tag" "$out" \
        "$PY" scripts/run_replay_gain.py --seed "$seed" --arm "$arm" --t-steps "$t" \
        --updates "$u" --batch-eps 32 --eval-episodes "$e" --lr 1e-4 \
        --save-episodes --out "$out"
    fi
  done
  log "[eval] worker=$wid gpu=$gpu queue empty"
}

i=0
for gpu in $GPUS; do
  i=$((i + 1))
  eval_worker "$gpu" "$i" &
  log "launched eval worker $i on gpu=$gpu pid=$!"
done
wait
log "[all done]"

failures="$(python3 - "$STATUS" <<'PY'
import json, sys
bad = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    if row.get("exit", 0) != 0:
        bad.append(f"{row['tag']}:exit{row['exit']}")
print(" ".join(bad))
PY
)"
if [[ -n "$failures" ]]; then
  log "[warn] failed jobs: $failures"
else
  log "[ok] all jobs exit=0"
fi
