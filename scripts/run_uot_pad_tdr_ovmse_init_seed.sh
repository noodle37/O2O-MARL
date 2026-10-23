#!/bin/bash
# PAD + TDR: the Per-agent Distillation control with TDR's eta decay schedule.
#
# Same offline checkpoint, same online recipe and same eta schedule as
# scripts/run_uot_tdr_ovmse_init_seed.sh. The ONLY difference is
# system.uot_ranking_scope=agent, which switches the ranking loss from the joint
# recorded-neighbourhood hinge to per-agent pairwise ordinal distillation
# (uot/learner.py:_agent_ranking_loss -- teacher per-agent Q ordering only, no
# mixer and no joint candidate pool). eta_at() multiplies whichever loss the
# scope selected, so the decay schedule applies identically to both arms.
#
# This is the pre-registered Experiment-4 "PAD + TDR" arm. The comparison it
# supports is PAD+TDR vs TDR (joint scope) at matched seed and matched init:
# any gap is the joint-vs-individual ranking information, not the schedule.
# Note the earlier checkpoints/uot_controls_5m_vs_6m/pad arm is NOT this arm --
# it ran ranking_scope=agent with NO decay (eta flat 0.1) from an older offline
# checkpoint.
#
# Usage: SEED=1 GPU=1 nohup bash scripts/run_uot_pad_tdr_ovmse_init_seed.sh \
#          > logs/uot_pad_tdr_ovmse_init_5m_vs_6m_seed1_master.log 2>&1 &
set -euo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

SEED="${SEED:-1}"
GPU="${GPU:-1}"
T_MAX="${T_MAX:-3000000}"
TEST_INTERVAL="${TEST_INTERVAL:-10000}"
DECAY_STEPS="${DECAY_STEPS:-2000000}"
RUN_ROOT="${RUN_ROOT:-checkpoints/uot_pad_tdr_ovmse_init_5m_vs_6m}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/pad_tdr_seed$SEED}"
CFG=configs/uot_control_5m_vs_6m.yaml
OFFLINE_CKPT=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt

mkdir -p "$RUN_DIR" logs
if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: missing offline checkpoint $OFFLINE_CKPT" >&2
  exit 1
fi

common=(
  seed="$SEED"
  task.dataset=Medium
  checkpoint_dir="$RUN_DIR"
  checkpoint_path="$OFFLINE_CKPT"
  log_path="$RUN_DIR/online.jsonl"
  online.t_max="$T_MAX"
  t_max="$T_MAX"
  online.test_interval="$TEST_INTERVAL"
  test_interval="$TEST_INTERVAL"
  system.uot_ranking_scope=agent
  system.uot_eta_decay_start=0
  system.uot_eta_decay_steps="$DECAY_STEPS"
)

{
  printf 'CUDA_VISIBLE_DEVICES=%s %s train.py --config %s' "$GPU" "$PY" "$CFG"
  printf ' %q' "${common[@]}"
  echo
} > "$RUN_DIR/train_command.txt"

# checkpoint_stem is f"uot_{ranking_scope}_{support_mode}" (uot/learner.py:87),
# so the agent-scope endpoint is uot_agent_joint_online_none_seed<N>.pt, not the
# joint-scope uot_online_none_seed<N>.pt the TDR script expects.
CKPT="$RUN_DIR/uot_agent_joint_online_none_seed$SEED.pt"

echo "=== [$(date '+%F %T')] PAD+TDR seed=$SEED train start run_dir=$RUN_DIR offline_ckpt=$OFFLINE_CKPT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --config "$CFG" "${common[@]}" \
  > "$RUN_DIR/train.out" 2>&1
echo "=== [$(date '+%F %T')] PAD+TDR seed=$SEED train done"

if [[ ! -f "$CKPT" ]]; then
  echo "FATAL: missing endpoint checkpoint $CKPT" >&2
  exit 1
fi

{
  printf 'CUDA_VISIBLE_DEVICES=%s %s scripts/eval_640.py --config %s' "$GPU" "$PY" "$CFG"
  printf ' %q' "${common[@]}"
  printf ' --checkpoint %q --episodes 640 --chunk 32 --out %q --tag %q\n' \
    "$CKPT" "$RUN_DIR/eval640.jsonl" "pad_tdr_ovmseinit_seed$SEED"
} > "$RUN_DIR/eval_command.txt"

echo "=== [$(date '+%F %T')] eval640 seed=$SEED start ckpt=$CKPT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_640.py --config "$CFG" "${common[@]}" \
  --checkpoint "$CKPT" --episodes 640 --chunk 32 \
  --out "$RUN_DIR/eval640.jsonl" --tag "pad_tdr_ovmseinit_seed$SEED" > "$RUN_DIR/eval640.out" 2>&1
echo "=== [$(date '+%F %T')] ALL DONE seed=$SEED run_dir=$RUN_DIR"
