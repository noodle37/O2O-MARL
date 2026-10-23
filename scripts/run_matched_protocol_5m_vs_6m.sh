#!/bin/bash
# Matched-protocol arms A-E on SMAC 5m_vs_6m.
#
# All arms share one offline checkpoint, one online recipe, and one code snapshot.
# The only differences are the pre-registered switches:
#   A  uot ranking (eta=0.1), OVM off, mixing 0.0
#   B  uot eta=0 (vanilla O2O), OVM off, mixing 0.0
#   C  OVMSE protocol, OVM on, mixing 0.1
#   D  OVM only, OVM on, mixing 0.0
#   E  mixing only, OVM off, mixing 0.1
#
# Usage:
#   nohup bash scripts/run_matched_protocol_5m_vs_6m.sh \
#     > logs/matched_protocol_master_seed1.log 2>&1 &
#
# Required:
#   OFFLINE_CKPT  checkpoint selected by Step 0 (same for every arm)
# Optional:
#   ARMS="A B C D E" SEED=1 T_MAX=3000000 RUN_ROOT=checkpoints/matched_protocol_20261020
#   A_GPU=3 B_GPU=3 C_GPU=3 D_GPU=2 E_GPU=2
set -uo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

CFG=configs/matched_protocol_5m_vs_6m.yaml
OFFLINE_CKPT="${OFFLINE_CKPT:?set OFFLINE_CKPT to the Step-0 selected checkpoint}"
SEED="${SEED:-1}"
T_MAX="${T_MAX:-3000000}"
TEST_INTERVAL="${TEST_INTERVAL:-10000}"
RUN_ROOT="${RUN_ROOT:-checkpoints/matched_protocol_20261020}"
ARMS="${ARMS:-A B C D E}"

A_GPU="${A_GPU:-3}"
B_GPU="${B_GPU:-3}"
C_GPU="${C_GPU:-3}"
D_GPU="${D_GPU:-2}"
E_GPU="${E_GPU:-2}"

read -r -a ARM_ORDER <<< "$ARMS"
mkdir -p "$RUN_ROOT" logs

if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: offline checkpoint missing: $OFFLINE_CKPT" >&2
  exit 1
fi

echo "=== matched protocol ==="
echo "code HEAD: $(git rev-parse --short HEAD)"
echo "code diff sha256: $(git diff | sha256sum | cut -d' ' -f1)"
echo "offline ckpt: $OFFLINE_CKPT"
echo "seed=$SEED t_max=$T_MAX arms=[$ARMS]"
echo "gpu map A=$A_GPU B=$B_GPU C=$C_GPU D=$D_GPU E=$E_GPU"
echo "config: $CFG"

arm_gpu() {
  case "$1" in
    A) echo "$A_GPU" ;;
    B) echo "$B_GPU" ;;
    C) echo "$C_GPU" ;;
    D) echo "$D_GPU" ;;
    E) echo "$E_GPU" ;;
  esac
}

arm_overrides() {
  case "$1" in
    A)
      printf '%s\n' \
        "learner_type=uot" \
        "transfer.mode=none" \
        "system.uot_eta=0.1" \
        "system.mixing_ratio=0.0"
      ;;
    B)
      printf '%s\n' \
        "learner_type=uot" \
        "transfer.mode=none" \
        "system.uot_eta=0.0" \
        "system.mixing_ratio=0.0"
      ;;
    C)
      printf '%s\n' \
        "learner_type=ovmse_paper" \
        "transfer.mode=ovmse" \
        "system.mixing_ratio=0.1" \
        "online.use_ovm=true"
      ;;
    D)
      printf '%s\n' \
        "learner_type=ovmse_paper" \
        "transfer.mode=ovmse" \
        "system.mixing_ratio=0.0" \
        "online.use_ovm=true"
      ;;
    E)
      printf '%s\n' \
        "learner_type=ovmse_paper" \
        "transfer.mode=none" \
        "system.mixing_ratio=0.1" \
        "online.use_ovm=false"
      ;;
  esac
}

run_arm() {
  local arm="$1"
  local gpu stem dir rc
  gpu="$(arm_gpu "$arm")"
  dir="$RUN_ROOT/${arm}_seed${SEED}"
  mkdir -p "$dir"

  mapfile -t extra < <(arm_overrides "$arm")
  local common=(
    "seed=$SEED"
    "task.dataset=Medium"
    "checkpoint_dir=$dir"
    "checkpoint_path=$OFFLINE_CKPT"
    "log_path=$dir/online.jsonl"
    "online.t_max=$T_MAX"
    "t_max=$T_MAX"
    "online.test_interval=$TEST_INTERVAL"
    "test_interval=$TEST_INTERVAL"
  )

  {
    echo "# arm=$arm gpu=$gpu seed=$SEED t_max=$T_MAX"
    printf 'CUDA_VISIBLE_DEVICES=%s %s train.py --config %s' "$gpu" "$PY" "$CFG"
    printf ' %q' "${common[@]}" "${extra[@]}"
    echo
  } > "$dir/train_command.txt"

  echo "=== [$(date '+%F %T')] arm=$arm gpu=$gpu stage=train start dir=$dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$CFG" \
    "${common[@]}" "${extra[@]}" > "$dir/train.out" 2>&1
  rc=$?
  echo "=== [$(date '+%F %T')] arm=$arm stage=train done rc=$rc"
  if [[ $rc -ne 0 ]]; then
    echo "arm=$arm train failed; see $dir/train.out" >&2
    return $rc
  fi
  echo "=== [$(date '+%F %T')] arm=$arm final files:"
  ls -lh "$dir"/*.pt 2>/dev/null || true
  return 0
}

declare -A PID
for arm in "${ARM_ORDER[@]}"; do
  gpu="$(arm_gpu "$arm")"
  ( run_arm "$arm" ) > "logs/matched_protocol_seed${SEED}_${arm}.out" 2>&1 &
  PID[$arm]=$!
  echo "launched arm=$arm gpu=$gpu pid=${PID[$arm]} log=logs/matched_protocol_seed${SEED}_${arm}.out"
done

failed=0
for arm in "${ARM_ORDER[@]}"; do
  wait "${PID[$arm]}"
  rc=$?
  echo "=== [$(date '+%F %T')] arm=$arm exit=$rc ==="
  [[ $rc -ne 0 ]] && failed=1
done

if [[ $failed -ne 0 ]]; then
  echo "=== [$(date '+%F %T')] SOME ARMS FAILED ==="
  exit 1
fi
echo "=== [$(date '+%F %T')] ALL ARMS DONE ==="
