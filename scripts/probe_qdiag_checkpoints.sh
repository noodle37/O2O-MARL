#!/usr/bin/env bash
# Fixed-seed probe of every periodic checkpoint of the four O2O diagnostic runs.
# Writes one JSON per (run, step); skips files that already exist.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/jxy/miniconda3/envs/multiAgentRL/bin/python}
ROOT=checkpoints/qdiag_20261020
OUT=$ROOT/probes
EPISODES=${EPISODES:-8}
SEED=${SEED:-12345}
mkdir -p "$OUT" logs
: > "logs/probe_qdiag_checkpoints.log"

for run in 2halfcheetah_off 2halfcheetah_on 5m_vs_6m_off 5m_vs_6m_on; do
  cfg="configs/qdiag_${run}.yaml"
  run_dir="$ROOT/$run"
  # t=0 uses the offline checkpoint named in the config
  offline_ckpt=$("$PY" - "$cfg" <<'PY'
import sys
from omegaconf import OmegaConf
print(OmegaConf.load(sys.argv[1]).checkpoint_path)
PY
)
  t0_out="$OUT/${run}_t000000.json"
  if [ ! -f "$t0_out" ]; then
    echo "[probe] $run t=0 $offline_ckpt" | tee -a "logs/probe_qdiag_checkpoints.log"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
      "$PY" scripts/measure_paper_q.py --config "$cfg" --checkpoint "$offline_ckpt" \
      --label "${run}_t0" --episodes "$EPISODES" --seed "$SEED" --out "$t0_out" \
      >> "logs/probe_qdiag_checkpoints.log" 2>&1
  fi
  shopt -s nullglob
  for ckpt in $(ls -v "$run_dir"/periodic/*.pt 2>/dev/null); do
    base=$(basename "$ckpt")
    step=$(echo "$base" | sed -n 's/.*_t\([0-9]*\)\.pt/\1/p')
    [ -n "$step" ] || continue
    out="$OUT/${run}_t$(printf '%06d' "$step").json"
    [ -f "$out" ] && continue
    echo "[probe] $run t=$step $ckpt" | tee -a "logs/probe_qdiag_checkpoints.log"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
      "$PY" scripts/measure_paper_q.py --config "$cfg" --checkpoint "$ckpt" \
      --label "${run}_t${step}" --episodes "$EPISODES" --seed "$SEED" --out "$out" \
      >> "logs/probe_qdiag_checkpoints.log" 2>&1
  done
  shopt -u nullglob
done
echo "[probe] done" | tee -a "logs/probe_qdiag_checkpoints.log"
