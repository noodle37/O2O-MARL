#!/bin/bash
# TDR (joint ranking, eta 0.1 -> 0 over the first 2M steps) initialized from
# the same 5.08M-best offline checkpoint as the OVMSE 3M reference run
# (checkpoints/ovmse_batch_20261014/5m_vs_6m_online). The same checkpoint is
# also the frozen ranking teacher, matching OVMSE's OVM teacher.
set -euo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

GPU="${GPU:-2}"
T_MAX="${T_MAX:-3000000}"
TEST_INTERVAL="${TEST_INTERVAL:-10000}"
DECAY_STEPS="${DECAY_STEPS:-2000000}"
RUN_ROOT="${RUN_ROOT:-checkpoints/uot_tdr_ovmse_init_5m_vs_6m}"
RUN_DIR="$RUN_ROOT/tdr"
CFG=configs/uot_control_5m_vs_6m.yaml
OFFLINE_CKPT=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt

mkdir -p "$RUN_DIR" logs
if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: missing offline checkpoint $OFFLINE_CKPT" >&2
  exit 1
fi

common=(
  seed=1
  task.dataset=Medium
  checkpoint_dir="$RUN_DIR"
  checkpoint_path="$OFFLINE_CKPT"
  log_path="$RUN_DIR/online.jsonl"
  online.t_max="$T_MAX"
  t_max="$T_MAX"
  online.test_interval="$TEST_INTERVAL"
  test_interval="$TEST_INTERVAL"
  system.uot_eta_decay_start=0
  system.uot_eta_decay_steps="$DECAY_STEPS"
)

{
  printf 'CUDA_VISIBLE_DEVICES=%s %s train.py --config %s' "$GPU" "$PY" "$CFG"
  printf ' %q' "${common[@]}"
  echo
} > "$RUN_DIR/train_command.txt"

echo "=== [$(date '+%F %T')] TDR train start run_dir=$RUN_DIR offline_ckpt=$OFFLINE_CKPT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --config "$CFG" "${common[@]}" \
  > "$RUN_DIR/train.out" 2>&1
echo "=== [$(date '+%F %T')] TDR train done"

CKPT="$RUN_DIR/uot_online_none_seed1.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "FATAL: missing endpoint checkpoint $CKPT" >&2
  exit 1
fi

{
  printf 'CUDA_VISIBLE_DEVICES=%s %s scripts/eval_640.py --config %s' "$GPU" "$PY" "$CFG"
  printf ' %q' "${common[@]}"
  printf ' --checkpoint %q --episodes 640 --chunk 32 --out %q --tag tdr_ovmseinit_seed1\n' \
    "$CKPT" "$RUN_DIR/eval640.jsonl"
} > "$RUN_DIR/eval_command.txt"

echo "=== [$(date '+%F %T')] eval640 start ckpt=$CKPT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_640.py --config "$CFG" "${common[@]}" \
  --checkpoint "$CKPT" --episodes 640 --chunk 32 \
  --out "$RUN_DIR/eval640.jsonl" --tag tdr_ovmseinit_seed1 > "$RUN_DIR/eval640.out" 2>&1
echo "=== [$(date '+%F %T')] ALL DONE run_dir=$RUN_DIR"
