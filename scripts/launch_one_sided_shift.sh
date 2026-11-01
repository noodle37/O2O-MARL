#!/bin/bash
# One-sided policy-shift evaluation on SMAC 5m_vs_6m.
#
# Grid eps:     all agents keep the offline net; a designated subset takes a
#               uniformly random legal action with probability eps.
# Grid replace: the designated subset uses an online fine-tuned checkpoint.
#
# Conditions per grid:
#   O  all offline greedy
#   A  all agents deviate
#   B  focal stays offline greedy, teammates deviate
#   C  only focal deviates, teammates stay offline greedy
#
# Usage:
#   bash scripts/launch_one_sided_shift.sh smoke   # 2 episodes, GPU 1, wiring check
#   bash scripts/launch_one_sided_shift.sh pilot   # 64 episodes, signal check
#   nohup bash scripts/launch_one_sided_shift.sh full \
#     > logs/one_sided_shift/driver.log 2>&1 &
#
# GPU 0 is never used: it hosts another user's long-running job.
set -uo pipefail

cd "$(dirname "$0")/.."
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
CFG=configs/pv_base_5m_vs_6m.yaml
OFFLINE=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt
MUAPR1=checkpoints/muapr_5m_vs_6m/seed1/ovmse_paper_online_full_td_seed1.pt
MUAPR2=checkpoints/muapr_5m_vs_6m/seed2/ovmse_paper_online_full_td_seed2.pt
MUAPR3=checkpoints/muapr_5m_vs_6m/seed3/ovmse_paper_online_full_td_seed3.pt
PVC=checkpoints/pv_exp1/5m_vs_6m/C_seed1/periodic/ovm_diag_ckpt_t250000.pt

ROOT=checkpoints/one_sided_shift/5m_vs_6m
LOGS=logs/one_sided_shift
STATUS="$LOGS/job_status.jsonl"
mkdir -p "$ROOT/eps" "$ROOT/replace" "$ROOT/smoke" "$ROOT/pilot" "$LOGS"

log() { echo "=== [$(date '+%F %T')] $* ==="; }

PHASE="${1:-full}"
JOBS=()
WORKERS=3
EPISODES="${EPISODES:-640}"

build_full_jobs() {
  JOBS+=("eps|eps_O|eps|O|0|0|none")
  JOBS+=("eps|eps_O_repeat|eps|O|0|0|none")
  for spec in "005:0.05" "010:0.1" "020:0.2" "030:0.3"; do
    tag="${spec%%:*}"
    eps="${spec##*:}"
    for arm in A B C; do
      JOBS+=("eps|eps_${arm}_e${tag}_f0|eps|${arm}|0|${eps}|none")
    done
  done
  for f in 1 2 3 4; do
    JOBS+=("eps|eps_B_e010_f${f}|eps|B|${f}|0.1|none")
    JOBS+=("eps|eps_C_e010_f${f}|eps|C|${f}|0.1|none")
  done
  for src in "muapr_s1:${MUAPR1}" "muapr_s2:${MUAPR2}" "muapr_s3:${MUAPR3}" "pv_c_s1_250k:${PVC}"; do
    name="${src%%:*}"
    path="${src#*:}"
    JOBS+=("replace|replace_${name}_A|replace|A|0|0|${path}")
    JOBS+=("replace|replace_${name}_B_f0|replace|B|0|0|${path}")
    JOBS+=("replace|replace_${name}_C_f0|replace|C|0|0|${path}")
  done
}

case "$PHASE" in
  smoke)
    ACTIVE_ROOT="$ROOT/smoke"
    WORKERS=1
    EPISODES=2
    JOBS=(
      "eps|eps_O|eps|O|0|0|none"
      "eps|eps_O_repeat|eps|O|0|0|none"
      "eps|eps_A_e010_f0|eps|A|0|0.1|none"
      "eps|eps_B_e010_f0|eps|B|0|0.1|none"
      "eps|eps_C_e010_f0|eps|C|0|0.1|none"
      "replace|replace_muapr_s1_B_f0|replace|B|0|0|${MUAPR1}"
    )
    ;;
  pilot)
    ACTIVE_ROOT="$ROOT/pilot"
    WORKERS=3
    EPISODES=64
    JOBS=(
      "eps|eps_O|eps|O|0|0|none"
      "eps|eps_A_e020_f0|eps|A|0|0.2|none"
      "eps|eps_B_e020_f0|eps|B|0|0.2|none"
      "eps|eps_C_e020_f0|eps|C|0|0.2|none"
      "replace|replace_muapr_s1_A|replace|A|0|0|${MUAPR1}"
      "replace|replace_muapr_s1_B_f0|replace|B|0|0|${MUAPR1}"
      "replace|replace_muapr_s1_C_f0|replace|C|0|0|${MUAPR1}"
    )
    ;;
  full)
    ACTIVE_ROOT="$ROOT"
    EPISODES="${EPISODES:-640}"
    build_full_jobs
    ;;
  *)
    echo "usage: $0 [smoke|pilot|full]" >&2
    exit 2
    ;;
esac

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  log "phase=$PHASE root=$ACTIVE_ROOT episodes=$EPISODES jobs=${#JOBS[@]}"
  for job in "${JOBS[@]}"; do echo "  $job"; done
  exit 0
fi

has_summary() {
  local f="$1"
  [[ -f "$f" ]] && tail -n 1 "$f" | grep -q '"summary": true'
}

run_worker() {
  local w="$1"
  local gpu="$((1 + w))"
  local i=0
  for job in "${JOBS[@]}"; do
    if (( i % WORKERS == w )); then
      IFS='|' read -r grid label mode arm focal eps live <<< "$job"
      local out="$ACTIVE_ROOT/$grid/${label}.jsonl"
      if has_summary "$out"; then
        log "[skip] $label (finished)"
      else
        mkdir -p "$(dirname "$out")"
        log "[start] $label gpu=$gpu mode=$mode arm=$arm focal=$focal eps=$eps"
        printf '{"phase":"%s","label":"%s","gpu":"%s","start":"%s"}\n' \
          "$PHASE" "$label" "$gpu" "$(date -Iseconds)" >> "$STATUS"
        local args=(
          --config "$CFG"
          --offline-ckpt "$OFFLINE"
          --live-ckpt "$live"
          --mode "$mode"
          --arm "$arm"
          --focal "$focal"
          --eps "$eps"
          --episodes "$EPISODES"
          --eval-seed 424242
          --noise-seed 424242
          --device cuda
          --out "$out"
        )
        if [[ "$PHASE" == "smoke" ]]; then args+=(--smoke); fi
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_one_sided_shift.py \
          "${args[@]}" > "$LOGS/${label}.log" 2>&1
        local rc=$?
        printf '{"phase":"%s","label":"%s","gpu":"%s","exit":%d,"end":"%s"}\n' \
          "$PHASE" "$label" "$gpu" "$rc" "$(date -Iseconds)" >> "$STATUS"
        if (( rc == 0 )); then log "[done] $label"; else log "[fail] $label exit=$rc"; fi
      fi
    fi
    i=$((i + 1))
  done
}

log "phase=$PHASE root=$ACTIVE_ROOT episodes=$EPISODES jobs=${#JOBS[@]} workers=$WORKERS"
if (( WORKERS == 1 )); then
  run_worker 0
else
  for w in $(seq 0 $((WORKERS - 1))); do
    run_worker "$w" &
  done
  wait
fi
log "phase '$PHASE' finished"
