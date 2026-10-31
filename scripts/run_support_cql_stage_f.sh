#!/bin/bash
# Stage F: mass-normalized support-conditioned CQL placement.
#
# Same setup as Stage D, but gated/anti arms normalize their per-batch CQL weight
# so that mean weight == 1.0, exactly matching the uniform arm in total mass.
# Any benefit can then only come from WHERE the penalty is placed.
#
# Usage: run_support_cql_stage_f.sh <seed> <gpu> [arms...]
#        arms default to "nocql uniform gated_norm anti_norm"
set -euo pipefail

SEED="${1:?usage: run_support_cql_stage_f.sh <seed> <gpu> [arms...]}"
GPU="${2:?usage: run_support_cql_stage_f.sh <seed> <gpu> [arms...]}"
shift 2
ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  ARMS=(nocql uniform gated_norm anti_norm)
fi

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
CFG=configs/support_gate_5m_vs_6m.yaml
OFFLINE_CKPT=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt
ROOT="checkpoints/support_cql_stage_f/5m_vs_6m"

if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: shared offline checkpoint missing: $OFFLINE_CKPT" >&2
  exit 3
fi

for ARM in "${ARMS[@]}"; do
  OVERRIDES=()
  case "$ARM" in
    nocql)
      OVERRIDES+=(system.online_cql_weight=0 system.support_cql_mode=off system.use_ovm=false online.use_ovm=false)
      ;;
    uniform)
      OVERRIDES+=(system.online_cql_weight=50.0 system.support_cql_mode=off system.support_cql_normalize_mass=false)
      ;;
    gated_norm)
      OVERRIDES+=(system.online_cql_weight=50.0 system.support_cql_mode=on system.support_cql_normalize_mass=true)
      ;;
    anti_norm)
      OVERRIDES+=(system.online_cql_weight=50.0 system.support_cql_mode=anti system.support_cql_normalize_mass=true)
      ;;
    *) echo "unknown arm '$ARM' (expected nocql|uniform|gated_norm|anti_norm)" >&2; exit 2 ;;
  esac

  RUN_DIR="${ROOT}/${ARM}_seed${SEED}"
  mkdir -p "$RUN_DIR"
  if [[ -f "$RUN_DIR/online.jsonl" ]] && grep -q '"t_env": 200000' "$RUN_DIR/online.jsonl" 2>/dev/null; then
    echo "skip ${ARM} seed=${SEED}: run already reached 200000 steps"
    continue
  fi

  echo "=== [$(date '+%F %T')] stage F arm=${ARM} seed=${SEED} gpu=${GPU} ==="
  "$PY" train.py --config "$CFG" \
    seed="$SEED" \
    checkpoint_dir="$RUN_DIR" \
    log_path="$RUN_DIR/online.jsonl" \
    checkpoint_path="$OFFLINE_CKPT" \
    "${OVERRIDES[@]}"
  echo "=== [$(date '+%F %T')] stage F arm=${ARM} seed=${SEED} DONE ==="
done

echo "=== [$(date '+%F %T')] stage F seed=${SEED} all arms DONE ==="
