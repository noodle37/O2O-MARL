#!/bin/bash
# One-sided policy-shift evaluation on SMAC 3m and 8m.
#
# Conditions per map (focus 0):
#   O        all offline greedy
#   O_repeat same as O (determinism check)
#   A/B/C    all deviate / teammates deviate (focal offline) / only focal deviates
#            at eps in {0.05, 0.10, 0.20, 0.30}
#
# 3m runs on GPU 1, 8m on GPU 2; one map per card.
#
# Usage:
#   bash scripts/launch_one_sided_shift_3m_8m.sh smoke   # 2 episodes, wiring check
#   bash scripts/launch_one_sided_shift_3m_8m.sh full    # 14 conditions x 640
#   bash scripts/launch_one_sided_shift_3m_8m.sh all     # smoke, then full
#   nohup bash scripts/launch_one_sided_shift_3m_8m.sh all \
#     > logs/one_sided_shift_3m_8m/driver.log 2>&1 &
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
LOGS=logs/one_sided_shift_3m_8m
STATUS="$LOGS/job_status.jsonl"
mkdir -p "$LOGS"

MAPS=(3m 8m)
declare -A CFG OFFLINE ROOT GPU
CFG[3m]=configs/pv_base_3m.yaml
OFFLINE[3m]=checkpoints/ovmse_paper_3m/ovmse_paper_seed1_best.pt
ROOT[3m]=checkpoints/one_sided_shift/3m
GPU[3m]=1
CFG[8m]=configs/pv_base_8m.yaml
OFFLINE[8m]=checkpoints/ovmse_paper_8m/ovmse_paper_seed1_best.pt
ROOT[8m]=checkpoints/one_sided_shift/8m
GPU[8m]=2

log() { echo "=== [$(date '+%F %T')] $* ==="; }

build_jobs() {
  local phase="$1"
  JOBS=()
  if [[ "$phase" == "smoke" ]]; then
    JOBS+=("eps|eps_O|eps|O|0|0|none")
    JOBS+=("eps|eps_A_e010_f0|eps|A|0|0.1|none")
    JOBS+=("eps|eps_B_e010_f0|eps|B|0|0.1|none")
    JOBS+=("eps|eps_C_e010_f0|eps|C|0|0.1|none")
    return
  fi
  JOBS+=("eps|eps_O|eps|O|0|0|none")
  JOBS+=("eps|eps_O_repeat|eps|O|0|0|none")
  for spec in "005:0.05" "010:0.1" "020:0.2" "030:0.3"; do
    local tag="${spec%%:*}"
    local eps="${spec##*:}"
    for arm in A B C; do
      JOBS+=("eps|eps_${arm}_e${tag}_f0|eps|${arm}|0|${eps}|none")
    done
  done
}

has_summary() {
  local f="$1"
  [[ -f "$f" ]] && tail -n 1 "$f" | grep -q '"summary": true'
}

run_map() {
  local phase="$1"
  local map="$2"
  local cfg="${CFG[$map]}"
  local offline="${OFFLINE[$map]}"
  local gpu="${GPU[$map]}"
  local active_root="${ROOT[$map]}"
  local episodes=640
  if [[ "$phase" == "smoke" ]]; then
    active_root="$active_root/smoke"
    episodes=2
  fi
  mkdir -p "$active_root/eps" "$LOGS/$map"
  build_jobs "$phase"
  log "[$map] phase=$phase gpu=$gpu jobs=${#JOBS[@]} episodes=$episodes"
  local rc_all=0
  for job in "${JOBS[@]}"; do
    IFS='|' read -r grid label mode arm focal eps live <<< "$job"
    local out="$active_root/$grid/${label}.jsonl"
    if has_summary "$out"; then
      log "[$map][skip] $label (finished)"
      continue
    fi
    log "[$map][start] $label gpu=$gpu mode=$mode arm=$arm focal=$focal eps=$eps"
    printf '{"phase":"%s","map":"%s","label":"%s","gpu":"%s","start":"%s"}\n' \
      "$phase" "$map" "$label" "$gpu" "$(date -Iseconds)" >> "$STATUS"
    local args=(
      --config "$cfg"
      --offline-ckpt "$offline"
      --mode "$mode"
      --arm "$arm"
      --focal "$focal"
      --eps "$eps"
      --episodes "$episodes"
      --eval-seed 424242
      --noise-seed 424242
      --device cuda
      --out "$out"
    )
    if [[ "$phase" == "smoke" ]]; then args+=(--smoke); fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_one_sided_shift.py \
      "${args[@]}" > "$LOGS/$map/${label}.log" 2>&1
    local rc=$?
    printf '{"phase":"%s","map":"%s","label":"%s","gpu":"%s","exit":%d,"end":"%s"}\n' \
      "$phase" "$map" "$label" "$gpu" "$rc" "$(date -Iseconds)" >> "$STATUS"
    if (( rc == 0 )); then log "[$map][done] $label"; else log "[$map][fail] $label exit=$rc"; rc_all=1; fi
  done
  return $rc_all
}

run_phase() {
  local phase="$1"
  local pids=()
  for map in "${MAPS[@]}"; do
    run_map "$phase" "$map" &
    pids+=("$!")
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

run_analysis() {
  local map="$1"
  local root="${ROOT[$map]}"
  log "[$map] analyze"
  "$PY" scripts/analyze_one_sided_shift.py \
    --root "$root" \
    --include eps \
    --baseline eps_O \
    --out "$root/summary.json" > "$LOGS/$map/analyze.log" 2>&1
  if (( $? == 0 )); then log "[$map] summary -> $root/summary.json"; else log "[$map] analyze failed"; fi
}

PHASE="${1:-all}"
case "$PHASE" in
  smoke)
    run_phase smoke
    ;;
  full)
    run_phase full && for map in "${MAPS[@]}"; do run_analysis "$map"; done
    ;;
  all)
    if run_phase smoke; then
      log "smoke passed; starting full grid"
      if run_phase full; then
        for map in "${MAPS[@]}"; do run_analysis "$map"; done
      else
        log "full grid had failures"
      fi
    else
      log "smoke failed; not starting full grid"
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 [smoke|full|all]" >&2
    exit 2
    ;;
esac
