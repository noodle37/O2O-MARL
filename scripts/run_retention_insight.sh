#!/bin/bash
# Retention-insight validation H1/H2 on 5m_vs_6m: one-agent counterfactual
# deviations and K-way candidate selectability, measured by branch replay
# against a frozen greedy teacher. Nothing is trained; parameter_updates=0.
#
# Sharded by environment seed because each branch replay relaunches SC2 and is
# CPU-bound. A finished shard is a self-contained JSON, so a crashed seed can be
# rerun without touching the others (RETENTION_FORCE=1 reruns anyway).
#
# Usage: run_retention_insight.sh [n_seeds] [seed0]
#
# Env:
#   RETENTION_PARALLEL   shards in flight             (default 8)
#   RETENTION_FORCE=1    rerun shards already on disk
#   RETENTION_OUT        merged JSON                  (default logs/retention_insight.json)
#   RETENTION_SHARD_DIR  per-seed shards and logs     (default logs/retention_shards)
#   RETENTION_CONFIG     config                       (default configs/5m_vs_6m.yaml)
#   RETENTION_CKPT       frozen teacher               (default UOT final; the
#                        offline best wins only ~20% of seeds, too few successes)
#
# Read from the merged JSON: summary.selectability_by_set (H2),
# summary.deviation_by_kind and summary.random_pairs (H1), and
# summary.selectability_minus_global for the clustered paired difference.
set -euo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
PY="${RETENTION_PY:-/home/jxy/miniconda3/envs/multiAgentRL/bin/python}"

N_SEEDS="${1:-80}"
SEED0="${2:-1}"
PARALLEL="${RETENTION_PARALLEL:-8}"
FORCE="${RETENTION_FORCE:-0}"
OUT="${RETENTION_OUT:-logs/retention_insight.json}"
SHARD_DIR="${RETENTION_SHARD_DIR:-logs/retention_shards}"
CONFIG="${RETENTION_CONFIG:-configs/5m_vs_6m.yaml}"
CKPT="${RETENTION_CKPT:-checkpoints/uot_temporal_5m_vs_6m/uot_online_none_seed1.pt}"

if [[ ! -f "$CKPT" ]]; then
  echo "FATAL: frozen teacher checkpoint missing: $CKPT" >&2
  exit 1
fi
if (( PARALLEL < 1 )); then
  echo "FATAL: RETENTION_PARALLEL must be >= 1" >&2
  exit 1
fi

mkdir -p "$SHARD_DIR" "$(dirname "$OUT")"
DRIVER_LOG="$SHARD_DIR/driver.log"
log() { echo "=== [$(date '+%F %T')] $* ===" | tee -a "$DRIVER_LOG"; }

shard_done() {
  local f="$1"
  [[ -s "$f" ]] && "$PY" - "$f" <<'PYEOF' 2>/dev/null
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get("parameter_updates") == 0 and "summary" in payload else 1)
PYEOF
}

log "start n_seeds=$N_SEEDS seed0=$SEED0 parallel=$PARALLEL config=$CONFIG ckpt=$CKPT out=$OUT"

pids=()
running=()
fail=0
for seed in $(seq "$SEED0" "$((SEED0 + N_SEEDS - 1))"); do
  shard="$SHARD_DIR/shard_${seed}.json"
  if (( ! FORCE )) && shard_done "$shard"; then
    log "skip seed $seed (finished shard on disk)"
    continue
  fi
  log "seed $seed start"
  "$PY" scripts/verify_retention_insight.py \
    --config "$CONFIG" \
    --frozen-ckpt "$CKPT" \
    --seed0 "$seed" \
    --n-seeds 1 \
    --out "$shard" \
    > "$SHARD_DIR/shard_${seed}.log" 2>&1 &
  pids+=("$!")
  running+=("$seed")
  if (( ${#pids[@]} >= PARALLEL )); then
    if ! wait "${pids[0]}"; then
      log "seed ${running[0]} FAILED (see shard_${running[0]}.log)"
      fail=$((fail + 1))
    fi
    pids=("${pids[@]:1}")
    running=("${running[@]:1}")
  fi
done

while (( ${#pids[@]} > 0 )); do
  if ! wait "${pids[0]}"; then
    log "seed ${running[0]} FAILED (see shard_${running[0]}.log)"
    fail=$((fail + 1))
  fi
  pids=("${pids[@]:1}")
  running=("${running[@]:1}")
done

log "merging shards from $SHARD_DIR"
"$PY" scripts/verify_retention_insight.py \
  --merge-dir "$SHARD_DIR" \
  --out "$OUT" >> "$DRIVER_LOG" 2>&1

if (( fail > 0 )); then
  log "DONE with $fail failed shard(s); merged the completed shards into $OUT"
  exit 1
fi
log "DONE merged -> $OUT"
