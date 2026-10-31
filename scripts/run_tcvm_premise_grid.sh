#!/bin/bash
# Stage A premise grid: offline-value-memory floor quality across policy versions.
set -euo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
CFG=configs/pv_base_5m_vs_6m.yaml
OUT=checkpoints/tcvm_premise/5m_vs_6m
EPISODES="${EPISODES:-256}"

mkdir -p "$OUT" logs

run_online() {
  local arm="$1" seed="$2" t="$3"
  local ckpt="checkpoints/pv_exp1/5m_vs_6m/${arm}_seed${seed}/periodic/ovm_diag_ckpt_t${t}.pt"
  local tag="${arm}_seed${seed}_t${t}"
  if [[ ! -f "$ckpt" ]]; then
    echo "SKIP missing $ckpt"
    return
  fi
  if [[ -f "$OUT/${tag}.json" ]]; then
    echo "SKIP done $tag"
    return
  fi
  echo "=== [$(date '+%F %T')] online $tag ==="
  "$PY" scripts/run_tcvm_premise.py --config "$CFG" --policy-ckpt "$ckpt" \
    --tag "$tag" --episodes "$EPISODES" --out "$OUT" 2>&1 | grep -v "^ *$"
}

run_offline_data() {
  local arm="$1" seed="$2" t="$3"
  local ckpt="checkpoints/pv_exp1/5m_vs_6m/${arm}_seed${seed}/periodic/ovm_diag_ckpt_t${t}.pt"
  local tag="offlinedata_${arm}_seed${seed}_t${t}"
  if [[ -f "$OUT/${tag}.json" ]]; then
    echo "SKIP done $tag"
    return
  fi
  echo "=== [$(date '+%F %T')] offline-data $tag ==="
  "$PY" scripts/run_tcvm_premise.py --config "$CFG" --policy-ckpt "$ckpt" \
    --source offline_data --tag "$tag" --episodes "$EPISODES" --out "$OUT" 2>&1 | grep -v "^ *$"
}

for T in 25000 100000 250000 1000000; do
  run_online A 1 "$T"
done
for T in 25000 250000 1000000; do
  run_online C 1 "$T"
done
run_online A 2 250000
run_online C 2 250000
run_offline_data A 1 25000
run_offline_data A 1 250000
run_offline_data A 1 1000000

echo "=== [$(date '+%F %T')] premise grid DONE ==="
