#!/bin/bash
# Stage G: can offline replay mixing and online CQL stacking be combined?
#
# Usage: run_stage_g_stacking.sh <seed> <gpu> [arms...]
#        arms default to "mix both"; "none" and "cql" are accepted for reruns.
#
# The 2x2 design is
#                online_cql_weight
#                 0            50
#   mixing 0    none          cql      (identical to Stage D nocql / uniform)
#   mixing .1   mix           both
#
# `none` and `cql` already exist as Stage D `nocql` / `uniform` with the same
# checkpoint, recipe, budget, evaluation seed and seeds 1..3, so the default only
# runs the two new cells. The analysis script reads the Stage D logs for the 0/0
# and 0/50 cells.
#
# The two mechanisms act on different parts of the batch (which transitions are
# sampled vs. what loss is applied), so they can be implemented together; the
# question this stage answers is whether their measured benefits ADD. Everything
# else is off: no OVM value memory and no teammate-support gate.
set -euo pipefail

SEED="${1:?usage: run_stage_g_stacking.sh <seed> <gpu> [arms...]}"
GPU="${2:?usage: run_stage_g_stacking.sh <seed> <gpu> [arms...]}"
shift 2
ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  ARMS=(mix both)
fi

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
CFG=configs/support_gate_5m_vs_6m.yaml
OFFLINE_CKPT=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt
BEH=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/behaviour/behaviour.pt
ROOT="checkpoints/stage_g_stacking/5m_vs_6m"
EVAL_ROWS=21
EXTRA=(${STAGE_G_OVERRIDES:-})

if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: shared offline checkpoint missing: $OFFLINE_CKPT" >&2
  exit 3
fi

COMMON=(transfer.mode=full_td system.use_ovm=false online.use_ovm=false
        system.support_gate_mode=off system.support_cql_mode=off
        system.mixing_ratio=0.0 "${EXTRA[@]}")

for ARM in "${ARMS[@]}"; do
  OVERRIDES=("${COMMON[@]}")
  case "$ARM" in
    none) OVERRIDES+=(system.mixing_ratio=0.0 system.online_cql_weight=0.0) ;;
    mix)  OVERRIDES+=(system.mixing_ratio=0.1 system.online_cql_weight=0.0) ;;
    cql)  OVERRIDES+=(system.mixing_ratio=0.0 system.online_cql_weight=50.0) ;;
    both) OVERRIDES+=(system.mixing_ratio=0.1 system.online_cql_weight=50.0) ;;
    *) echo "unknown arm '$ARM' (expected none|mix|cql|both)" >&2; exit 2 ;;
  esac
  if [[ "$ARM" == "cql" || "$ARM" == "both" ]] && [[ ! -f "$BEH" ]]; then
    echo "FATAL: behaviour prior cache missing: $BEH" >&2
    exit 3
  fi

  RUN_DIR="${ROOT}/${ARM}_seed${SEED}"
  mkdir -p "$RUN_DIR"
  if [[ -f "$RUN_DIR/online.jsonl" ]] \
     && [[ $(wc -l < "$RUN_DIR/online.jsonl") -ge $EVAL_ROWS ]]; then
    echo "skip ${ARM} seed=${SEED}: already has $EVAL_ROWS eval rows"
    continue
  fi

  echo "=== [$(date '+%F %T')] stage G arm=${ARM} seed=${SEED} gpu=${GPU} ==="
  "$PY" train.py --config "$CFG" \
    seed="$SEED" \
    checkpoint_dir="$RUN_DIR" \
    log_path="$RUN_DIR/online.jsonl" \
    "${OVERRIDES[@]}"
  echo "=== [$(date '+%F %T')] stage G arm=${ARM} seed=${SEED} DONE ==="
done

echo "=== [$(date '+%F %T')] stage G seed=${SEED} arms ${ARMS[*]} DONE ==="
