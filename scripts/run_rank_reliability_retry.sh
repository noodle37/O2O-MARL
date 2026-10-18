#!/bin/bash
# Retry driver for the rank-reliability experiment (task 2).
#
# The first full run (logs/auto_driver.log) launched 12 shards in parallel; 3 of them
# (env seeds 4, 5, 8) died on SC2 websocket/connect errors before writing any shard
# JSON, so their records are missing from logs/rank_reliability_full.json
# (18/24 expected trajectory clusters). This script re-runs only the missing seeds
# with lower parallelism (3 at a time) and up to 3 attempts per seed, then re-merges
# all shards into logs/rank_reliability_full.json.
#
# Launch:  nohup bash scripts/run_rank_reliability_retry.sh > logs/auto_retry.out 2>&1 &
set -u
ROOT=/home/jxy/multiAgentRL
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
B_SNAPSHOT=$ROOT/checkpoints/seed2_continuation_B.pt
SHARD_DIR=$ROOT/logs/rank_rel_shards
LOG=$ROOT/logs/auto_retry.log
MISSING=(4 5 8)
MAX_ATTEMPTS=3
PARALLEL=3
mkdir -p "$SHARD_DIR"
log(){ echo "[retry $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_one(){  # $1 = seed  -> exit code 0 iff shard_<seed>.json exists
  local s=$1
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    if [ -f "$SHARD_DIR/shard_${s}.json" ]; then log "seed $s already has output; skip"; return 0; fi
    log "seed $s attempt $attempt/$MAX_ATTEMPTS"
    $PY "$ROOT/scripts/verify_rank_reliability.py" \
      --n-seeds 1 --seed0 "$s" --waves greedy,epsgreedy --eps 0.3 \
      --k-cand 6 --margin 0.1 --max-probes 4 --max-joints 0 --k-near 64 \
      --cont-ckpt "$B_SNAPSHOT" --device cpu \
      --out "$SHARD_DIR/shard_${s}.json" >> "$SHARD_DIR/shard_${s}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$SHARD_DIR/shard_${s}.json" ]; then
      log "seed $s OK (attempt $attempt)"; return 0
    fi
    log "seed $s FAILED attempt $attempt (rc=$rc)"
    sleep 20
  done
  return 1
}

log "retry run start: missing seeds = ${MISSING[*]}"
pids=()
for s in "${MISSING[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  run_one "$s" & pids+=($!)
  sleep 15
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || { fail=1; log "retry worker pid=$pid failed"; }; done
log "retry shards done (any_fail=$fail)"

# --- merge ---------------------------------------------------------------------
$PY - "$ROOT" "$SHARD_DIR" <<'PY'
import json,sys,glob,os
root, shard_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
from scripts.verify_rank_reliability import _summary
records=[]; meta=None
for f in sorted(glob.glob(os.path.join(shard_dir,"shard_*.json"))):
    try:
        d=json.load(open(f))
    except Exception as e:
        print("skip unreadable shard", f, e); continue
    records.extend(d.get("records",[])); meta=d
if meta is None:
    print("NO SHARDS MERGED"); sys.exit(1)
payload={"config":meta.get("config"),"frozen_checkpoint":meta.get("frozen_checkpoint"),
  "cont_checkpoint":meta.get("cont_checkpoint"),"waves":meta.get("waves"),"eps":meta.get("eps"),
  "k_cand":meta.get("k_cand"),"margin":meta.get("margin"),"k_near":meta.get("k_near"),
  "parameter_updates":0,"records":records,"summary":_summary(records)}
out=os.path.join(root,"logs","rank_reliability_full.json")
json.dump(payload, open(out,"w"), indent=2)
print("MERGED", len(records), "records ->", out)
print("POOLED:", json.dumps(payload["summary"]["pooled"]))
PY
log "re-merged -> $ROOT/logs/rank_reliability_full.json ; DONE"
