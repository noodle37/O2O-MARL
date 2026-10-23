#!/bin/bash
# Evaluate matched-protocol arms with the common 640-episode protocol.
# Runs the final checkpoint and every periodic checkpoint, resumably.
#
# Usage:
#   bash scripts/eval_matched_protocol.sh A B C D E
#   ARMS="A B" SEED=1 bash scripts/eval_matched_protocol.sh
#   SKIP_PERIODIC=1 bash scripts/eval_matched_protocol.sh A
set -uo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

CFG=configs/matched_protocol_5m_vs_6m.yaml
SEED="${SEED:-1}"
RUN_ROOT="${RUN_ROOT:-checkpoints/matched_protocol_20261020}"
EVAL_EPISODES="${EVAL_EPISODES:-640}"
EVAL_CHUNK="${EVAL_CHUNK:-32}"
SKIP_FINAL="${SKIP_FINAL:-0}"
SKIP_PERIODIC="${SKIP_PERIODIC:-0}"
ARMS="${ARMS:-$*}"

A_GPU="${A_GPU:-3}"
B_GPU="${B_GPU:-3}"
C_GPU="${C_GPU:-3}"
D_GPU="${D_GPU:-2}"
E_GPU="${E_GPU:-2}"

read -r -a ARM_ORDER <<< "$ARMS"

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
      printf '%s\n' "learner_type=uot" "transfer.mode=none" "system.uot_eta=0.1" "system.mixing_ratio=0.0"
      ;;
    B)
      printf '%s\n' "learner_type=uot" "transfer.mode=none" "system.uot_eta=0.0" "system.mixing_ratio=0.0"
      ;;
    C)
      printf '%s\n' "learner_type=ovmse_paper" "transfer.mode=ovmse" "system.mixing_ratio=0.1" "online.use_ovm=true"
      ;;
    D)
      printf '%s\n' "learner_type=ovmse_paper" "transfer.mode=ovmse" "system.mixing_ratio=0.0" "online.use_ovm=true"
      ;;
    E)
      printf '%s\n' "learner_type=ovmse_paper" "transfer.mode=none" "system.mixing_ratio=0.1" "online.use_ovm=false"
      ;;
  esac
}

has_summary() {
  [[ -f "$1" ]] && grep -q '"checkpoint"' "$1"
}

eval_ckpt() {
  local arm="$1" ckpt="$2" tag="$3" out="$4" gpu overrides
  gpu="$(arm_gpu "$arm")"
  mapfile -t overrides < <(arm_overrides "$arm")
  mkdir -p "$(dirname "$out")"
  if has_summary "$out"; then
    echo "skip existing $out"
    return 0
  fi
  echo "=== [$(date '+%F %T')] eval arm=$arm tag=$tag ckpt=$ckpt"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_640.py --config "$CFG" \
    seed="$SEED" task.dataset=Medium \
    "${overrides[@]}" \
    --checkpoint "$ckpt" --episodes "$EVAL_EPISODES" --chunk "$EVAL_CHUNK" \
    --out "$out" --tag "$tag" > "${out%.jsonl}.out" 2>&1
  local rc=$?
  echo "=== [$(date '+%F %T')] eval arm=$arm tag=$tag done rc=$rc"
  return $rc
}

for arm in "${ARM_ORDER[@]}"; do
  dir="$RUN_ROOT/${arm}_seed${SEED}"
  if [[ ! -d "$dir" ]]; then
    echo "missing arm dir: $dir" >&2
    continue
  fi
  final_ckpt="$(ls -1 "$dir"/*_online_*_seed${SEED}.pt 2>/dev/null | head -1)"
  if [[ "$SKIP_FINAL" != "1" && -n "$final_ckpt" ]]; then
    eval_ckpt "$arm" "$final_ckpt" "${arm}_seed${SEED}_final" "$dir/eval640_final.jsonl"
  elif [[ "$SKIP_FINAL" != "1" ]]; then
    echo "no final checkpoint yet for $arm ($dir)"
  fi
  if [[ "$SKIP_PERIODIC" != "1" ]]; then
    for p in "$dir"/periodic/ovm_diag_ckpt_t*.pt; do
      [[ -f "$p" ]] || continue
      step="$(basename "$p" .pt | sed 's/.*_t//')"
      eval_ckpt "$arm" "$p" "${arm}_seed${SEED}_t${step}" "$dir/eval640_t${step}.jsonl"
    done
  fi
done
echo "=== [$(date '+%F %T')] matched eval pass complete ==="
