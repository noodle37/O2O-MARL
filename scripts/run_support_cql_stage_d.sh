#!/bin/bash
# Stage D: does teammate support tell us WHERE to spend conservatism?
#
# Usage: run_support_cql_stage_d.sh <seed> <gpu> [arms...]
#        arms default to "nocql uniform gated anti"
#
# Stage C killed the value-memory channel: there, the gate's placement could not
# be separated from the floor's magnitude, because max(q_off, y) binds hardest
# exactly where offline support is high. This channel does not have that defect.
# The CQL term is reduced by a weighted mean, so reweighting its placement
# leaves the total conservatism mass unchanged by construction, and the three
# weighted arms differ ONLY in where the penalty sits.
#
#   nocql    no conservative term online at all (channel reference)
#   uniform  uniform placement; this is exactly what a tuned global
#            online_cql_weight can express, so it is the matched-mass control
#   gated    penalty kept where the executed joint action is in the old team's
#            support, theta calibrated from the offline data's own quantile
#   anti     the same gate inverted; same mass, opposite placement
#
# The mechanism only holds if gated beats BOTH uniform and anti with a
# consistent sign across seeds. Beating nocql alone would only show that the
# conservative term helps, which PV exp1 arm B already established.
set -euo pipefail

SEED="${1:?usage: run_support_cql_stage_d.sh <seed> <gpu> [arms...]}"
GPU="${2:?usage: run_support_cql_stage_d.sh <seed> <gpu> [arms...]}"
shift 2
ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  ARMS=(nocql uniform gated anti)
fi

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
CFG=configs/support_gate_5m_vs_6m.yaml
OFFLINE_CKPT=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt
ROOT="checkpoints/support_cql_stage_d/5m_vs_6m"
EVAL_ROWS=21

if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: shared offline checkpoint missing: $OFFLINE_CKPT" >&2
  exit 3
fi

# The conservative term needs the cached CFCQL behaviour prior; training it here
# would add hours and differ per arm.
BEH=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/behaviour/behaviour.pt
if [[ ! -f "$BEH" ]]; then
  echo "FATAL: behaviour prior cache missing: $BEH" >&2
  exit 3
fi

# The value-memory channel is off in every arm so it cannot confound this test.
COMMON=(transfer.mode=full_td system.use_ovm=false online.use_ovm=false
        system.support_gate_mode=off)

for ARM in "${ARMS[@]}"; do
  OVERRIDES=("${COMMON[@]}")
  case "$ARM" in
    nocql)   OVERRIDES+=(system.online_cql_weight=0.0  system.support_cql_mode=off) ;;
    uniform) OVERRIDES+=(system.online_cql_weight=50.0 system.support_cql_mode=off) ;;
    gated)   OVERRIDES+=(system.online_cql_weight=50.0 system.support_cql_mode=on) ;;
    anti)    OVERRIDES+=(system.online_cql_weight=50.0 system.support_cql_mode=anti) ;;
    *) echo "unknown arm '$ARM' (expected nocql|uniform|gated|anti)" >&2; exit 2 ;;
  esac

  RUN_DIR="${ROOT}/${ARM}_seed${SEED}"
  mkdir -p "$RUN_DIR"
  if [[ -f "$RUN_DIR/online.jsonl" ]] \
     && [[ $(wc -l < "$RUN_DIR/online.jsonl") -ge $EVAL_ROWS ]]; then
    echo "skip ${ARM} seed=${SEED}: already has $EVAL_ROWS eval rows"
    continue
  fi

  echo "=== [$(date '+%F %T')] stage D arm=${ARM} seed=${SEED} gpu=${GPU} ==="
  "$PY" train.py --config "$CFG" \
    seed="$SEED" \
    checkpoint_dir="$RUN_DIR" \
    log_path="$RUN_DIR/online.jsonl" \
    "${OVERRIDES[@]}"
  echo "=== [$(date '+%F %T')] stage D arm=${ARM} seed=${SEED} DONE ==="
done

echo "=== [$(date '+%F %T')] stage D seed=${SEED} all arms DONE ==="
