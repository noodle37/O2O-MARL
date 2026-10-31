#!/bin/bash
set -euo pipefail

cd /home/jxy/multiAgentRL
mkdir -p logs

CFG=configs/pv_base_5m_vs_6m.yaml
for SEED in 1 2 3; do
  CKPT="checkpoints/pv_exp1/5m_vs_6m/C_seed${SEED}/periodic/ovm_diag_ckpt_t250000.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "FATAL: missing start checkpoint for seed ${SEED}: $CKPT" >&2
    exit 3
  fi
done
if [[ ! -f "$CFG" ]]; then
  echo "FATAL: missing config $CFG" >&2
  exit 3
fi

launch() {
  local seed="$1"
  local gpu="$2"
  local log="logs/joint_reuse_online_seed${seed}_gpu${gpu}.out"
  local pid_file="logs/joint_reuse_online_gpu${gpu}_seed${seed}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "skip seed=${seed} gpu=${gpu}: pid $(cat "$pid_file") already running"
    return
  fi
  nohup bash scripts/run_joint_reuse_seed.sh "$seed" "$gpu" > "$log" 2>&1 &
  echo $! > "$pid_file"
  echo "started seed=${seed} gpu=${gpu} pid=$! log=${log}"
}

launch 1 0
launch 2 1
launch 3 1
