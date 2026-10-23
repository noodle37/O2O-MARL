#!/bin/bash
# FSR feasibility: train/calibrate the support flow from existing OVMSE QMIX,
# then online fine-tune on 2s3z (GPU 2) and 2halfcheetah (GPU 3).
# Usage: nohup bash scripts/run_fsr_feasibility.sh > logs/fsr_feasibility_master.log 2>&1 &
set -uo pipefail

cd /home/jxy/multiAgentRL
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp

mkdir -p logs checkpoints/fsr_2s3z checkpoints/fsr_2halfcheetah

run_chain () {
  local gpu=$1 cfg=$2 name=$3
  local dir="checkpoints/${name}"
  local ckpt="$dir/fsr_seed1.pt"
  mkdir -p "$dir"
  echo "=== [$(date '+%F %T')] $name offline start gpu=$gpu ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$cfg" \
    checkpoint_dir="$dir" \
    log_path="$dir/offline.jsonl"
  local rc=$?
  echo "=== [$(date '+%F %T')] $name offline done rc=$rc ==="
  if [ $rc -ne 0 ]; then echo "$name offline failed, skipping online"; return $rc; fi
  if [ ! -f "$ckpt" ]; then
    echo "$name expected FSR checkpoint missing: $ckpt"; return 1
  fi
  echo "=== [$(date '+%F %T')] $name bucket gate ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/verify_fsr_buckets.py \
    --config "$cfg" --ckpt "$ckpt" || echo "$name bucket gate failed (continuing)"
  echo "=== [$(date '+%F %T')] $name online start ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$cfg" \
    phase=online \
    checkpoint_dir="$dir" \
    checkpoint_path="$ckpt" \
    online.log_path="$dir/online.jsonl" \
    log_path="$dir/online.jsonl"
  rc=$?
  echo "=== [$(date '+%F %T')] $name online done rc=$rc ==="
  return $rc
}

(
  run_chain 2 configs/fsr_2s3z.yaml fsr_2s3z
  echo "=== [$(date '+%F %T')] 2s3z CHAIN DONE ==="
) > logs/fsr_2s3z.out 2>&1 &
PID1=$!

(
  run_chain 3 configs/fsr_2halfcheetah.yaml fsr_2halfcheetah
  echo "=== [$(date '+%F %T')] 2halfcheetah CHAIN DONE ==="
) > logs/fsr_2halfcheetah.out 2>&1 &
PID2=$!

echo "2s3z pid=$PID1 gpu=2 log=logs/fsr_2s3z.out"
echo "2halfcheetah pid=$PID2 gpu=3 log=logs/fsr_2halfcheetah.out"

wait "$PID1"; echo "=== [$(date '+%F %T')] 2s3z exit=$? ==="
wait "$PID2"; echo "=== [$(date '+%F %T')] 2halfcheetah exit=$? ==="
echo "=== [$(date '+%F %T')] ALL DONE ==="
