#!/bin/bash
# Launch Stage F seeds 1 and 2 in parallel on GPU 0/1, then seed 3 on the
# first freed GPU.  Uses nohup so the runs survive session disconnect.
set -euo pipefail

cd /home/jxy/multiAgentRL
mkdir -p logs

PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

# Clear old partial outputs for a clean restart.
: > logs/stage_f_seed1_gpu0.out
: > logs/stage_f_seed2_gpu1.out
: > logs/stage_f_seed3_gpu0.out
: > logs/stage_f_seed3_gpu1.out

echo "[$(date '+%F %T')] launching seed 1 on GPU 0 and seed 2 on GPU 1"
nohup bash scripts/run_support_cql_stage_f.sh 1 0 >> logs/stage_f_seed1_gpu0.out 2>&1 &
PID1=$!
nohup bash scripts/run_support_cql_stage_f.sh 2 1 >> logs/stage_f_seed2_gpu1.out 2>&1 &
PID2=$!

echo "[$(date '+%F %T')] waiting for first seed to finish (PIDs $PID1, $PID2)"
wait -n
FIRST=$?

# Determine which GPU freed up.
if ! kill -0 $PID1 2>/dev/null; then
  FREE_GPU=0
  echo "[$(date '+%F %T')] seed 1 finished first; launching seed 3 on GPU 0"
else
  FREE_GPU=1
  echo "[$(date '+%F %T')] seed 2 finished first; launching seed 3 on GPU 1"
fi

nohup bash scripts/run_support_cql_stage_f.sh 3 "$FREE_GPU" >> "logs/stage_f_seed3_gpu${FREE_GPU}.out" 2>&1 &
PID3=$!

echo "[$(date '+%F %T')] waiting for all seeds to finish (PID3=$PID3)"
wait

echo "[$(date '+%F %T')] all Stage F seeds done"
