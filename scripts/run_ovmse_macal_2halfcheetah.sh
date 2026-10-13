#!/bin/bash
# OVMSE and MaCal-QL on 2halfcheetah (og_marl vault, Good), offline then online.
#
# Each algorithm gets its own GPU and its own pair of directories: the offline
# pretraining weights stay in *_offline/ and are never written to again, while
# the online fine-tune loads them via checkpoint_path and saves into *_online/.
#
# Usage: nohup bash scripts/run_ovmse_macal_2halfcheetah.sh > logs/ovmse_macal_2halfcheetah_master.log 2>&1 &
set -uo pipefail

cd /home/jxy/multiAgentRL
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
export TMPDIR=/tmp

OVMSE_GPU="${OVMSE_GPU:-1}"
MACAL_GPU="${MACAL_GPU:-2}"

OFF_OVMSE=checkpoints/ovmse_2halfcheetah_offline
ON_OVMSE=checkpoints/ovmse_2halfcheetah_online
OFF_MACAL=checkpoints/macal_2halfcheetah_offline
ON_MACAL=checkpoints/macal_2halfcheetah_online
mkdir -p "$OFF_OVMSE" "$ON_OVMSE" "$OFF_MACAL" "$ON_MACAL" logs

# $1=gpu $2=config $3=offline dir $4=online dir $5=offline best checkpoint
run_chain () {
  local gpu=$1 cfg=$2 off=$3 on=$4 best=$5
  echo "=== [$(date '+%F %T')] offline start cfg=$cfg gpu=$gpu dir=$off ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$cfg" \
    checkpoint_dir="$off" \
    log_path="$off/offline.jsonl"
  local rc=$?
  echo "=== [$(date '+%F %T')] offline done rc=$rc ==="
  if [ $rc -ne 0 ]; then echo "offline failed, skipping online"; return $rc; fi
  if [ ! -f "$best" ]; then
    echo "expected offline checkpoint missing: $best"; return 1
  fi
  # Snapshot the offline weights so nothing downstream can touch them.
  cp -p "$best" "$off/offline_pretrained_frozen.pt"
  echo "=== [$(date '+%F %T')] online start from=$best dir=$on ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$cfg" \
    phase=online \
    checkpoint_dir="$on" \
    checkpoint_path="$best" \
    log_path="$on/online.jsonl"
  rc=$?
  echo "=== [$(date '+%F %T')] online done rc=$rc ==="
  return $rc
}

(
  run_chain "$OVMSE_GPU" configs/2halfcheetah.yaml \
    "$OFF_OVMSE" "$ON_OVMSE" "$OFF_OVMSE/ovmse_paper_seed1_best.pt"
  echo "=== [$(date '+%F %T')] OVMSE CHAIN DONE ==="
) > logs/ovmse_2halfcheetah.out 2>&1 &
PID_OVMSE=$!
echo "ovmse chain pid=$PID_OVMSE gpu=$OVMSE_GPU log=logs/ovmse_2halfcheetah.out"

(
  run_chain "$MACAL_GPU" configs/macal_2halfcheetah.yaml \
    "$OFF_MACAL" "$ON_MACAL" "$OFF_MACAL/macal_ql_seed1_best.pt"
  echo "=== [$(date '+%F %T')] MACAL CHAIN DONE ==="
) > logs/macal_2halfcheetah.out 2>&1 &
PID_MACAL=$!
echo "macal chain pid=$PID_MACAL gpu=$MACAL_GPU log=logs/macal_2halfcheetah.out"

wait "$PID_OVMSE"; echo "=== [$(date '+%F %T')] ovmse chain exit=$? ==="
wait "$PID_MACAL"; echo "=== [$(date '+%F %T')] macal chain exit=$? ==="
echo "=== [$(date '+%F %T')] ALL DONE ==="
