#!/bin/bash
# Stage-3-only re-run of the seed-3 fork experiment.
#
# The seed-3 driver died right after the last branch finished (the fork script
# had been rewritten at 16:21:55 while its bash was still reading the old file
# by byte offset, so it misparsed line 150 and exited before stage 3). The three
# branch endpoints exist; only the 640-episode endpoint evaluations are missing.
#
# This replicates stage 3 of scripts/run_uot_fork_5m_vs_6m.sh exactly: same
# branch config, same overrides (env seed included), 640 episodes, chunk 32,
# same output names and tags. GPU 3 is used because the temporal main-line run
# there is env-bound (4% GPU utilisation), so the evaluations barely slow it.
set -euo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

SEED=3
RUN_DIR=checkpoints/uot_fork_5m_vs_6m_seed3
SEED_LOG="$RUN_DIR/driver.log"
OFFLINE_CKPT=checkpoints/ovmse_paper_5m_vs_6m_seed3/ovmse_paper_seed3.pt
BRANCH_CFG=configs/uot_5m_vs_6m_fork_branch.yaml
SNAP="$RUN_DIR/fork_snapshot_main_t1000325.pt"
FORK_T=1000325

log() { echo "=== [$(date '+%F %T')] $* ===" | tee -a "$SEED_LOG"; }

for f in "$SNAP" "$OFFLINE_CKPT"; do
  [[ -f "$f" ]] || { log "FATAL missing $f"; exit 1; }
done

EVAL_COMMON=(
  "seed=$SEED"
  "task.dataset=Medium"
  "checkpoint_dir=$RUN_DIR"
  "checkpoint_path=$OFFLINE_CKPT"
)

log "eval640 stage-3 re-run start (driver had died before stage 3)"
log "eval640 fork checkpoint"
"$PY" scripts/eval_640.py --config "$BRANCH_CFG" "${EVAL_COMMON[@]}" \
  --checkpoint "$SNAP" --episodes 640 --chunk 32 \
  --out "$RUN_DIR/eval640_fork.jsonl" --tag "fork_t${FORK_T}" 2>&1 | tee -a "$SEED_LOG"

for NAME in continue off decay; do
  CKPT="$RUN_DIR/uot_online_fork_resume_${NAME}_seed${SEED}.pt"
  if [[ ! -f "$CKPT" ]]; then
    log "WARNING missing endpoint checkpoint for $NAME: $CKPT"
    continue
  fi
  log "eval640 $NAME endpoint"
  "$PY" scripts/eval_640.py --config "$BRANCH_CFG" "${EVAL_COMMON[@]}" \
    --checkpoint "$CKPT" --episodes 640 --chunk 32 \
    --out "$RUN_DIR/eval640_${NAME}.jsonl" --tag "${NAME}_seed${SEED}" 2>&1 | tee -a "$SEED_LOG"
done

log "eval640 ALL DONE seed=$SEED RUN_DIR=$RUN_DIR"
