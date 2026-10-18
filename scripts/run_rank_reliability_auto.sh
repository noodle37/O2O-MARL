#!/bin/bash
# Autonomous driver for the offline joint-action ranking reliability experiment (task 2).
#
# Steps (all unattended, logs to logs/auto_driver.log):
#   1. Wait until the seed2 offline critic (continuation B) reaches a competent eval return.
#   2. Snapshot it to a fixed path so every measurement shard uses the SAME B.
#   3. Run the full A+B branch-replay measurement, sharded by env seed for parallelism
#      (A = frozen seed1 greedy, B = seed2 greedy; greedy + eps-greedy waves; K=6).
#   4. Merge shards into logs/rank_reliability_full.json with the grouped, cluster-CI summary.
#
# Launch:  nohup bash scripts/run_rank_reliability_auto.sh > logs/auto_driver.out 2>&1 &
set -u
ROOT=/home/jxy/multiAgentRL
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
SEED2_JSONL=$ROOT/checkpoints/ovmse_paper_5m_vs_6m_seed2/offline.jsonl
SEED2_BEST=$ROOT/checkpoints/ovmse_paper_5m_vs_6m_seed2/ovmse_paper_seed2_best.pt
B_SNAPSHOT=$ROOT/checkpoints/seed2_continuation_B.pt
SHARD_DIR=$ROOT/logs/rank_rel_shards
LOG=$ROOT/logs/auto_driver.log
THRESH=11.0          # competent eval return for continuation B
N_SEEDS=12           # env seeds 1..12  -> ~24 trajectories (greedy+epsgreedy) after 4 probes each
MAX_WAIT_MIN=360     # safety cap on waiting for seed2
mkdir -p "$SHARD_DIR"
log(){ echo "[driver $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- 1. wait for a competent seed2 -------------------------------------------------
log "waiting for seed2 continuation B to reach eval return >= $THRESH (cap ${MAX_WAIT_MIN}m) ..."
waited=0
while true; do
  best=$($PY - "$SEED2_JSONL" <<'PY'
import json,sys
try:
    rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    ev=[float(r["evaluation/mean_episode_return"]) for r in rows if "evaluation/mean_episode_return" in r]
    print(max(ev) if ev else -1e9)
except Exception:
    print(-1e9)
PY
)
  ok=$($PY -c "print(1 if float('$best')>=float('$THRESH') else 0)" 2>/dev/null || echo 0)
  if [ "$ok" = "1" ]; then log "seed2 competent: best eval return = $best"; break; fi
  if ! pgrep -f "configs/5m_vs_6m_seed2.yaml" >/dev/null; then
    log "WARNING: seed2 training process ended before threshold; using current best ($best)."; break
  fi
  if [ $waited -ge $((MAX_WAIT_MIN*60)) ]; then
    log "WARNING: wait cap reached; using current best ($best)."; break
  fi
  log "seed2 best so far = $best; still training ..."
  sleep 120; waited=$((waited+120))
done

# --- 2. snapshot B ------------------------------------------------------------------
if [ ! -f "$SEED2_BEST" ]; then log "FATAL: no seed2 best checkpoint at $SEED2_BEST"; exit 1; fi
cp -f "$SEED2_BEST" "$B_SNAPSHOT"
log "snapshotted continuation B -> $B_SNAPSHOT"

# --- 3. launch parallel shards (1 env seed each) ------------------------------------
pids=(); s=1
while [ $s -le $N_SEEDS ]; do
  $PY "$ROOT/scripts/verify_rank_reliability.py" \
    --n-seeds 1 --seed0 "$s" --waves greedy,epsgreedy --eps 0.3 \
    --k-cand 6 --margin 0.1 --max-probes 4 --max-joints 0 --k-near 64 \
    --cont-ckpt "$B_SNAPSHOT" --device cpu \
    --out "$SHARD_DIR/shard_${s}.json" > "$SHARD_DIR/shard_${s}.log" 2>&1 &
  pids+=($!); log "launched shard seed0=$s pid=$!"
  s=$((s+1)); sleep 3   # stagger vault reads
done
log "launched ${#pids[@]} shards; waiting for all to finish ..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || { fail=1; log "shard pid=$pid exited non-zero"; }; done
log "all shards done (any_fail=$fail)"

# --- 4. merge -----------------------------------------------------------------------
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
log "merged -> $ROOT/logs/rank_reliability_full.json ; DONE"
