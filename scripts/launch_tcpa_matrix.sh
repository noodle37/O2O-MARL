#!/usr/bin/env bash
# Launch the TCPA main experiment and its ablations on the three idle GPUs.
#
#   bash scripts/launch_tcpa_matrix.sh
#
# Every arm uses the same config, the same CFCQL-D offline prior and the same
# online protocol (3M env steps, 8 parallel envs, eval every 10k steps with 32
# episodes); only tcpa.variant differs.  That makes the arms directly
# comparable to the existing O2O baselines (ovmse / sapr / ackf / ...) whose
# logs live under checkpoints/.
#
# GPU 1: full (main)          -> no_igm (control without prior retention)
# GPU 2: fixed_gate            -> abs_q (ranking vs absolute Q retention)
# GPU 3: no_residual           -> gate_anti (falsification arm)
#
# After the 5m_vs_6m queue on GPU 1 the cross-map arm on 2s3z starts, so the
# method is also checked on a second map with the same recipe.
set -uo pipefail

cd "$(dirname "$0")/.."

T_MAX="${T_MAX:-3000000}"
SEED="${SEED:-1}"
export T_MAX SEED

mkdir -p logs/tcpa

nohup bash -c "
  MAP=5m_vs_6m bash scripts/run_tcpa_arms.sh 1 full no_igm
  MAP=2s3z     bash scripts/run_tcpa_arms.sh 1 full
" > logs/tcpa/gpu1.log 2>&1 &
echo "gpu1 pid $!"

nohup bash -c "
  MAP=5m_vs_6m bash scripts/run_tcpa_arms.sh 2 fixed_gate abs_q
" > logs/tcpa/gpu2.log 2>&1 &
echo "gpu2 pid $!"

nohup bash -c "
  MAP=5m_vs_6m bash scripts/run_tcpa_arms.sh 3 no_residual gate_anti
" > logs/tcpa/gpu3.log 2>&1 &
echo "gpu3 pid $!"

echo "launched: 7 arms over 3 GPUs (5m_vs_6m: 6 arms, 2s3z: 1 arm)"
