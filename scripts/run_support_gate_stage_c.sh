#!/bin/bash
# Stage C: does the teammate-support gate repair the offline value memory?
#
# Usage: run_support_gate_stage_c.sh <seed> <gpu> [arms...]
#        arms default to "none ovm tcvm anti"
#
# All four arms start from the same offline checkpoint, share the recipe, the
# budget, the exploration schedule and the fixed evaluation seed. They differ
# only in how the offline value memory is weighted:
#
#   none   no offline value memory at all
#   ovm    ungated memory (the published OVMSE schedule)
#   tcvm   memory kept only where the executed joint action is in the old team's
#          support, theta calibrated from the offline data's own quantile
#   anti   the same gate with inverted ordering, spending a comparable amount of
#          memory weight on the opposite steps
#
# anti is the arm that makes the result falsifiable: if tcvm only wins because
# it applies less memory, anti wins too, and the teammate conditioning is not
# doing the work.
set -euo pipefail

SEED="${1:?usage: run_support_gate_stage_c.sh <seed> <gpu> [arms...]}"
GPU="${2:?usage: run_support_gate_stage_c.sh <seed> <gpu> [arms...]}"
shift 2
ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  ARMS=(none ovm tcvm anti)
fi

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
CFG=configs/support_gate_5m_vs_6m.yaml
OFFLINE_CKPT=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt
ROOT="checkpoints/support_gate_stage_c/5m_vs_6m"

if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: shared offline checkpoint missing: $OFFLINE_CKPT" >&2
  exit 3
fi

for ARM in "${ARMS[@]}"; do
  OVERRIDES=()
  case "$ARM" in
    none) OVERRIDES+=(transfer.mode=full_td system.use_ovm=false online.use_ovm=false
                      system.support_gate_mode=off) ;;
    ovm)  OVERRIDES+=(system.support_gate_mode=off) ;;
    tcvm) OVERRIDES+=(system.support_gate_mode=on) ;;
    anti) OVERRIDES+=(system.support_gate_mode=anti) ;;
    *) echo "unknown arm '$ARM' (expected none|ovm|tcvm|anti)" >&2; exit 2 ;;
  esac

  RUN_DIR="${ROOT}/${ARM}_seed${SEED}"
  mkdir -p "$RUN_DIR"
  if [[ -f "$RUN_DIR/online.jsonl" ]] && grep -q '"t_env": 200000' "$RUN_DIR/online.jsonl" 2>/dev/null; then
    echo "skip ${ARM} seed=${SEED}: run already reached 200000 steps"
    continue
  fi

  echo "=== [$(date '+%F %T')] stage C arm=${ARM} seed=${SEED} gpu=${GPU} ==="
  "$PY" train.py --config "$CFG" \
    seed="$SEED" \
    checkpoint_dir="$RUN_DIR" \
    log_path="$RUN_DIR/online.jsonl" \
    "${OVERRIDES[@]}"
  echo "=== [$(date '+%F %T')] stage C arm=${ARM} seed=${SEED} DONE ==="
done

echo "=== [$(date '+%F %T')] stage C seed=${SEED} all arms DONE ==="
