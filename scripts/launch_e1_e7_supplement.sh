#!/bin/bash
# E1-E7 supplement driver (GPUs 2 and 3 only).
#
# Phase "measure" (evaluation / pilot only):
#   * E2/E3: adaptive local-reuse pilot on four online checkpoints, 600 trials each;
#   * E1:    fixed-focal transplant with partner noise eps in {0, .1, .2, .3};
#   * both analyses.
# Phase "train":
#   * Stage G: mixing_ratio 0.1 x online_cql_weight 50, arms `mix` and `both`,
#     seeds 1..3 (the 0/0 and 0/50 cells are Stage D `nocql`/`uniform`);
#   * Stage G analysis.
#
# Usage:
#   nohup bash scripts/launch_e1_e7_supplement.sh all > logs/e1_e7_supplement/driver.log 2>&1 &
#   bash scripts/launch_e1_e7_supplement.sh measure
#   bash scripts/launch_e1_e7_supplement.sh train
set -uo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
ROOT=checkpoints/e1_e7_supplement
LOGS=logs/e1_e7_supplement
mkdir -p "$ROOT/e1" "$ROOT/pilots" "$LOGS"
PHASE="${1:-all}"

OFFLINE_CKPT=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt
CKPT_NAMES=(B_s1_250k C_s1_250k C_s2_250k C_s3_250k D_s1_250k C_s1_1M)
CKPT_PATHS=(
  checkpoints/pv_exp1/5m_vs_6m/B_seed1/periodic/ovm_diag_ckpt_t250000.pt
  checkpoints/pv_exp1/5m_vs_6m/C_seed1/periodic/ovm_diag_ckpt_t250000.pt
  checkpoints/pv_exp1/5m_vs_6m/C_seed2/periodic/ovm_diag_ckpt_t250000.pt
  checkpoints/pv_exp1/5m_vs_6m/C_seed3/periodic/ovm_diag_ckpt_t250000.pt
  checkpoints/pv_exp1/5m_vs_6m/D_seed1/periodic/ovm_diag_ckpt_t250000.pt
  checkpoints/pv_exp1/5m_vs_6m/C_seed1/periodic/ovm_diag_ckpt_t1000000.pt
)
CKPT_SEEDS=(1 1 2 3 1 11)

log() { echo "=== [$(date '+%F %T')] $* ==="; }

run_measure() {
  log "phase measure: local-reuse pilots (6 x 600 trials)"
  for i in "${!CKPT_NAMES[@]}"; do
    name="${CKPT_NAMES[$i]}"
    if [[ -f "$ROOT/pilots/$name/pilot.json" ]]; then
      log "skip pilot $name (exists)"
      continue
    fi
    CUDA_VISIBLE_DEVICES=3 "$PY" scripts/run_adaptive_reuse_pilot.py \
      --config configs/pv_base_5m_vs_6m.yaml \
      --seed "${CKPT_SEEDS[$i]}" \
      --policy-ckpt "${CKPT_PATHS[$i]}" \
      --offline-ckpt "$OFFLINE_CKPT" \
      --treatment-agents 1,4 \
      --trials 600 \
      --out-dir "$ROOT/pilots/$name" \
      > "$LOGS/pilot_$name.log" 2>&1
    log "pilot $name exit=$?"
  done

  log "phase measure: fixed-focal transplant (6 processes, GPUs 2/3)"
  local pids=()
  for i in "${!CKPT_NAMES[@]}"; do
    name="${CKPT_NAMES[$i]}"
    gpu=$((2 + i % 2))
    if [[ -f "$ROOT/e1/$name.json" ]]; then
      log "skip E1 $name (exists)"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_e1_teammate_causal.py \
      --checkpoint "$name=${CKPT_PATHS[$i]}" \
      --focal single_0=0 pair_1_4=1,4 all=0,1,2,3,4 \
      --eps 0 0.1 0.2 0.3 \
      --episodes 64 \
      --offline-ckpt "$OFFLINE_CKPT" \
      --device cuda \
      --out "$ROOT/e1/$name.json" \
      > "$LOGS/e1_$name.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; log "E1 pid $pid exit=$?"; done

  log "phase measure: analyses"
  "$PY" scripts/analyze_e1_teammate_causal.py \
    --input "$ROOT/e1/*.json" \
    --out "$ROOT/e1_summary.json" 2>&1 | tee "$LOGS/analyze_e1.log"
  local pilot_args=()
  for i in "${!CKPT_NAMES[@]}"; do
    name="${CKPT_NAMES[$i]}"
    if [[ -f "$ROOT/pilots/$name/pilot.json" ]]; then
      pilot_args+=(--pilot "$name=$ROOT/pilots/$name/pilot.json")
    fi
  done
  if [[ ${#pilot_args[@]} -gt 0 ]]; then
    "$PY" scripts/analyze_e2_e3_local_benefit.py \
      "${pilot_args[@]}" \
      --out "$ROOT/e2_e3_summary.json" 2>&1 | tee "$LOGS/analyze_e2_e3.log"
  fi
}

run_train() {
  log "phase train: Stage G mix/both seeds 1..3 on GPUs 2/3"
  for seed in 1 2 3; do
    bash scripts/run_stage_g_stacking.sh "$seed" 2 mix \
      > "$LOGS/stage_g_mix_seed$seed.log" 2>&1 &
    bash scripts/run_stage_g_stacking.sh "$seed" 3 both \
      > "$LOGS/stage_g_both_seed$seed.log" 2>&1 &
  done
  wait
  log "phase train: Stage G analysis"
  "$PY" scripts/analyze_stage_g_stacking.py \
    --root checkpoints/stage_g_stacking/5m_vs_6m \
    --base-root checkpoints/support_cql_stage_d/5m_vs_6m \
    --tail 3 \
    --out checkpoints/stage_g_stacking/5m_vs_6m/summary.json \
    2>&1 | tee "$LOGS/analyze_stage_g.log"
}

case "$PHASE" in
  measure) run_measure ;;
  train) run_train ;;
  all) run_measure; run_train ;;
  *) echo "usage: $0 [all|measure|train]" >&2; exit 2 ;;
esac

log "driver phase '$PHASE' finished"
