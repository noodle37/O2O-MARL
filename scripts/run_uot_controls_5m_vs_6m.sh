#!/bin/bash
# Section-4 simple controls on SMAC 5m_vs_6m, all from the same seed-1 offline QMIX.
#
# Arms (one switch each, everything else identical to the fixed joint-ranking run):
#   fjr       fixed joint ranking, eta=0.1 for all 3M steps (the reference run
#             is checkpoints/uot_temporal_5m_vs_6m; this arm is not re-run)
#   pad       per-agent pairwise ordinal distillation, no mixer/joint candidates
#   tdr       joint ranking, eta linearly 0.1 -> 0 over the first 2M steps
#   marginal  joint ranking over per-agent marginally supported combinations
#   state     joint ranking over all legal joints, loss weighted by offline state density
#   none      joint ranking over all legal joints, no support or density weight
#
# Usage:
#   nohup bash scripts/run_uot_controls_5m_vs_6m.sh \
#     > logs/uot_controls_5m_vs_6m_master.log 2>&1 &
#
# Optional environment knobs (defaults match the formal protocol):
#   T_MAX=3000000 TEST_INTERVAL=10000 RUN_ROOT=checkpoints/uot_controls_5m_vs_6m
#   EVAL_EPISODES=640 EVAL_CHUNK=32 SKIP_EVAL=0 TDR_DECAY_STEPS=2000000
#   PAD_GPU=1 TDR_GPU=0 MARGINAL_GPU=1 STATE_GPU=0 NONE_GPU=3
#
# GPU placement uses cards 0/1/3, which are shared only with light existing jobs.
# GPU 2 is running three fork branches near full utilization and is left alone.
set -uo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

T_MAX="${T_MAX:-3000000}"
TEST_INTERVAL="${TEST_INTERVAL:-10000}"
RUN_ROOT="${RUN_ROOT:-checkpoints/uot_controls_5m_vs_6m}"
EVAL_EPISODES="${EVAL_EPISODES:-640}"
EVAL_CHUNK="${EVAL_CHUNK:-32}"
SKIP_EVAL="${SKIP_EVAL:-0}"
TDR_DECAY_STEPS="${TDR_DECAY_STEPS:-2000000}"
ARMS="${ARMS:-pad tdr marginal state none}"
PAD_GPU="${PAD_GPU:-1}"
TDR_GPU="${TDR_GPU:-0}"
MARGINAL_GPU="${MARGINAL_GPU:-1}"
STATE_GPU="${STATE_GPU:-0}"
NONE_GPU="${NONE_GPU:-3}"

CFG=configs/uot_control_5m_vs_6m.yaml
OFFLINE_CKPT=checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1.pt
read -r -a ARM_ORDER <<< "$ARMS"

mkdir -p "$RUN_ROOT" logs
if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: offline checkpoint missing: $OFFLINE_CKPT" >&2
  exit 1
fi

arm_gpu() {
  case "$1" in
    pad) echo "$PAD_GPU" ;;
    tdr) echo "$TDR_GPU" ;;
    marginal) echo "$MARGINAL_GPU" ;;
    state) echo "$STATE_GPU" ;;
    none) echo "$NONE_GPU" ;;
  esac
}

# The checkpoint stem encodes the learner switch; keep it in sync with
# UOTLearner.checkpoint_stem and training/online.py's final filename.
arm_stem() {
  case "$1" in
    pad) echo "uot_agent_joint" ;;
    tdr) echo "uot" ;;
    marginal) echo "uot_joint_marginal" ;;
    state) echo "uot_joint_state" ;;
    none) echo "uot_joint_none" ;;
  esac
}

arm_overrides() {
  case "$1" in
    pad) printf '%s\n' "system.uot_ranking_scope=agent" ;;
    tdr)
      printf '%s\n' \
        "system.uot_eta_decay_start=0" \
        "system.uot_eta_decay_steps=$TDR_DECAY_STEPS"
      ;;
    marginal) printf '%s\n' "system.uot_support_mode=marginal" ;;
    state) printf '%s\n' "system.uot_support_mode=state" ;;
    none) printf '%s\n' "system.uot_support_mode=none" ;;
  esac
}

run_arm() {
  local arm="$1"
  local gpu stem dir rc ckpt
  gpu="$(arm_gpu "$arm")"
  stem="$(arm_stem "$arm")"
  dir="$RUN_ROOT/$arm"
  mkdir -p "$dir"

  mapfile -t extra < <(arm_overrides "$arm")
  local common=(
    "seed=1"
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
    echo "# arm=$arm gpu=$gpu t_max=$T_MAX test_interval=$TEST_INTERVAL"
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

  ckpt="$dir/${stem}_online_none_seed1.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "arm=$arm endpoint checkpoint missing: $ckpt" >&2
    return 1
  fi
  echo "=== [$(date '+%F %T')] arm=$arm endpoint=$ckpt"

  if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "=== [$(date '+%F %T')] arm=$arm stage=eval skipped"
    return 0
  fi

  {
    echo "# arm=$arm gpu=$gpu episodes=$EVAL_EPISODES chunk=$EVAL_CHUNK"
    printf '%s scripts/eval_640.py --config %s' "$PY" "$CFG"
    printf ' %q' "${common[@]}" "${extra[@]}"
    printf ' --checkpoint %q --episodes %s --chunk %s --out %q --tag %q\n' \
      "$ckpt" "$EVAL_EPISODES" "$EVAL_CHUNK" "$dir/eval640.jsonl" "${arm}_seed1"
  } > "$dir/eval_command.txt"

  echo "=== [$(date '+%F %T')] arm=$arm stage=eval start episodes=$EVAL_EPISODES"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_640.py --config "$CFG" \
    "${common[@]}" "${extra[@]}" \
    --checkpoint "$ckpt" --episodes "$EVAL_EPISODES" --chunk "$EVAL_CHUNK" \
    --out "$dir/eval640.jsonl" --tag "${arm}_seed1" > "$dir/eval640.out" 2>&1
  rc=$?
  echo "=== [$(date '+%F %T')] arm=$arm stage=eval done rc=$rc"
  if [[ $rc -ne 0 ]]; then
    echo "arm=$arm eval failed; see $dir/eval640.out" >&2
    return $rc
  fi
  return 0
}

declare -A PID
for arm in "${ARM_ORDER[@]}"; do
  gpu="$(arm_gpu "$arm")"
  ( run_arm "$arm" ) > "logs/uot_control_${arm}.out" 2>&1 &
  PID[$arm]=$!
  echo "launched arm=$arm gpu=$gpu pid=${PID[$arm]} log=logs/uot_control_${arm}.out"
done

echo "=== [$(date '+%F %T')] all arms launched run_root=$RUN_ROOT t_max=$T_MAX ==="
failed=0
for arm in "${ARM_ORDER[@]}"; do
  wait "${PID[$arm]}"
  rc=$?
  echo "=== [$(date '+%F %T')] arm=$arm exit=$rc ==="
  if [[ $rc -ne 0 ]]; then
    failed=1
  fi
done

if [[ $failed -ne 0 ]]; then
  echo "=== [$(date '+%F %T')] SOME ARMS FAILED ==="
  exit 1
fi
echo "=== [$(date '+%F %T')] ALL ARMS DONE ==="
