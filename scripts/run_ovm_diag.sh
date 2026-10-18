#!/bin/bash
# Controlled OVM mechanism diagnostic: 3 maps x 2 arms (on/off), nohup.
#
# Each map's two arms resume from the SAME offline init and differ only in
# transfer.mode (ovmse -> use_ovm=True, full_td -> use_ovm=False); everything
# else (seed, epsilon schedule, batch composition, target cadence, diagnostics)
# is byte-identical, enforced by scripts/make_ovm_diag_configs.py.
#
# Outputs per arm:
#   checkpoints/ovm_diag_20261017/<map>_<arm>/online.jsonl   eval + diagnostic rows
#   checkpoints/ovm_diag_20261017/<map>_<arm>/ovm_diag_final_snapshot_t*.pt
#   checkpoints/ovm_diag_20261017/<map>_<arm>/periodic/ovm_diag_ckpt_t*.pt
#   logs/ovm_diag/<map>_<arm>.out                             full stdout
#
# Usage:
#   bash scripts/run_ovm_diag.sh            # generate configs + launch all arms
#   bash scripts/run_ovm_diag.sh --status   # show progress only
set -uo pipefail

cd /home/jxy/multiAgentRL
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID

RUN_ROOT=checkpoints/ovm_diag_20261017
LOG_DIR=logs/ovm_diag
mkdir -p "$RUN_ROOT" "$LOG_DIR"

# map:gpu pairs (see MAPS in scripts/make_ovm_diag_configs.py)
GPUS="2halfcheetah:2 2s3z:1 5m_vs_6m:0"

status () {
  echo "=== arms ==="
  for spec in $GPUS; do
    map=${spec%%:*}
    for arm in on off; do
      pid_file="$LOG_DIR/${map}_${arm}.pid"
      log="$LOG_DIR/${map}_${arm}.out"
      if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        state=RUNNING
      else
        state=stopped
      fi
      rows=0
      [[ -f "$RUN_ROOT/${map}_${arm}/online.jsonl" ]] && rows=$(wc -l < "$RUN_ROOT/${map}_${arm}/online.jsonl")
      last=""
      [[ -f "$RUN_ROOT/${map}_${arm}/online.jsonl" ]] && last=$($PY - "$RUN_ROOT/${map}_${arm}/online.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
last = rows[-1]
print("t_env=%s win=%.3f q_bias=%.3f act=%.3f lift=%.3f" % (
    last.get("t_env"), last.get("evaluation/win_rate", float("nan")),
    last.get("evaluation/q_bias", float("nan")),
    last.get("ovm_diag/activation_rate", float("nan")),
    last.get("ovm_diag/lift_raw_mean", float("nan"))))
EOF
)
      printf "%-14s %-8s %-8s rows=%-4s %s\n" "$map" "$arm" "$state" "$rows" "$last"
    done
  done
}

if [[ "${1:-}" == "--status" ]]; then
  status
  exit 0
fi

echo "=== [$(date '+%F %T')] generating configs ==="
"$PY" scripts/make_ovm_diag_configs.py || exit 1

echo "=== [$(date '+%F %T')] launching arms ==="
for spec in $GPUS; do
  map=${spec%%:*}
  gpu=${spec##*:}
  for arm in on off; do
    cfg="configs/ovm_diag_${map}_${arm}.yaml"
    log="$LOG_DIR/${map}_${arm}.out"
    pid_file="$LOG_DIR/${map}_${arm}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "[skip] ${map}_${arm} already running (pid $(cat "$pid_file"))"
      continue
    fi
    echo "[launch] ${map}_${arm} gpu=$gpu cfg=$cfg"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" train.py --config "$cfg" > "$log" 2>&1 &
    echo $! > "$pid_file"
    sleep 2
  done
done

sleep 20
echo "=== [$(date '+%F %T')] early status ==="
status
echo "logs: $LOG_DIR/*.out   data: $RUN_ROOT/*/online.jsonl"
