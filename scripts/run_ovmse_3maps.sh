#!/bin/bash
# OVMSE offline->online chains on 3 maps, one GPU each.
#   GPU1: SMAC 5m_vs_6m (cfcql Medium)
#   GPU2: SMAC 2s3z     (cfcql Medium)
#   GPU3: MaMujoco 2halfcheetah (og_marl Good)
# Fresh dirs under checkpoints/ovmse_batch_20261014/ so previous runs are untouched.
# Usage: nohup bash scripts/run_ovmse_3maps.sh > logs/ovmse_3maps_master.log 2>&1 &
set -uo pipefail

cd /home/jxy/multiAgentRL
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp

RUN_ROOT=checkpoints/ovmse_batch_20261014
mkdir -p "$RUN_ROOT" logs

# $1=gpu $2=config $3=name
run_chain () {
  local gpu=$1 cfg=$2 name=$3
  local off="$RUN_ROOT/${name}_offline"
  local on="$RUN_ROOT/${name}_online"
  local best="$off/ovmse_paper_seed1_best.pt"
  mkdir -p "$off" "$on"
  echo "=== [$(date '+%F %T')] $name offline start gpu=$gpu ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$cfg" \
    checkpoint_dir="$off" \
    behaviour_cache_dir="$off/behaviour" \
    log_path="$off/offline.jsonl"
  local rc=$?
  echo "=== [$(date '+%F %T')] $name offline done rc=$rc ==="
  if [ $rc -ne 0 ]; then echo "$name offline failed, skipping online"; return $rc; fi
  if [ ! -f "$best" ]; then
    echo "$name expected offline checkpoint missing: $best"; return 1
  fi
  cp -p "$best" "$off/offline_pretrained_frozen.pt"
  echo "=== [$(date '+%F %T')] $name online start from=$best ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$cfg" \
    phase=online \
    checkpoint_dir="$on" \
    checkpoint_path="$best" \
    online.log_path="$on/online.jsonl" \
    log_path="$on/online.jsonl"
  rc=$?
  echo "=== [$(date '+%F %T')] $name online done rc=$rc ==="
  return $rc
}

(
  run_chain 1 configs/5m_vs_6m.yaml 5m_vs_6m
  echo "=== [$(date '+%F %T')] 5m_vs_6m CHAIN DONE ==="
) > logs/ovmse_3maps_5m_vs_6m.out 2>&1 &
PID1=$!

(
  run_chain 2 configs/2s3z.yaml 2s3z
  echo "=== [$(date '+%F %T')] 2s3z CHAIN DONE ==="
) > logs/ovmse_3maps_2s3z.out 2>&1 &
PID2=$!

(
  run_chain 3 configs/2halfcheetah.yaml 2halfcheetah
  echo "=== [$(date '+%F %T')] 2halfcheetah CHAIN DONE ==="
) > logs/ovmse_3maps_2halfcheetah.out 2>&1 &
PID3=$!

echo "5m_vs_6m pid=$PID1 gpu=1 log=logs/ovmse_3maps_5m_vs_6m.out"
echo "2s3z pid=$PID2 gpu=2 log=logs/ovmse_3maps_2s3z.out"
echo "2halfcheetah pid=$PID3 gpu=3 log=logs/ovmse_3maps_2halfcheetah.out"

wait "$PID1"; echo "=== [$(date '+%F %T')] 5m_vs_6m exit=$? ==="
wait "$PID2"; echo "=== [$(date '+%F %T')] 2s3z exit=$? ==="
wait "$PID3"; echo "=== [$(date '+%F %T')] 2halfcheetah exit=$? ==="
echo "=== [$(date '+%F %T')] ALL DONE ==="
