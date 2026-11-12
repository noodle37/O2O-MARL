#!/usr/bin/env bash
# Run TCPA arms sequentially on one GPU.
#
#   bash scripts/run_tcpa_arms.sh <gpu> <variant> [<variant> ...]
#
# Environment:
#   MAP   scenario key, default 5m_vs_6m   (needs configs/tcpa_<MAP>.yaml)
#   SEED  run seed, default 1
#   T_MAX online budget in env steps, default 3000000
#
# Every arm shares the same config, the same offline checkpoint and the same
# online protocol; only tcpa.variant and the output paths differ, so the arms
# are directly comparable.
set -euo pipefail

GPU="$1"
shift

MAP="${MAP:-5m_vs_6m}"
SEED="${SEED:-1}"
T_MAX="${T_MAX:-3000000}"
CONFIG="configs/tcpa_${MAP}.yaml"
# The project environment (smac / omegaconf / torch); the base interpreter does
# not have them.
PYTHON="${PYTHON:-/home/jxy/miniconda3/envs/multiAgentRL/bin/python}"

if [ ! -f "$CONFIG" ]; then
  echo "config '$CONFIG' not found" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

for variant in "$@"; do
  out="checkpoints/tcpa_${MAP}/${variant}/seed${SEED}"
  mkdir -p "$out"
  echo "[tcpa] gpu=${GPU} map=${MAP} seed=${SEED} variant=${variant} -> ${out}"
  if "$PYTHON" train.py --config "$CONFIG" \
    "tcpa.variant=${variant}" \
    "seed=${SEED}" \
    "t_max=${T_MAX}" \
    "online.t_max=${T_MAX}" \
    "checkpoint_dir=${out}" \
    "log_path=${out}/online.jsonl" \
    > "${out}/train.out" 2>&1; then
    echo "[tcpa] finished variant=${variant}"
  else
    status=$?
    echo "[tcpa] FAILED variant=${variant} exit=${status}" >&2
  fi
done
