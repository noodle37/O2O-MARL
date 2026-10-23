#!/bin/bash
# Sharded driver for the task-2 temporal-evolution measurement.
#
# Usage: run_rank_temporal_auto.sh <manifest> <mode> <tag> [n_seeds] [parallel]
#   manifest: one continuation spec per line:  tag=path[:t_env]
#             (# comments and blank lines ignored)
#   mode    : fixed | own | both   (see scripts/verify_rank_temporal.py)
#   tag     : output tag; shards go to logs/rank_temporal_shards/<tag>/ and the
#             merged result to logs/rank_temporal_<tag>.json
#
# Launch:  nohup bash scripts/run_rank_temporal_auto.sh <manifest> fixed batch1 > logs/rank_temporal_batch1.out 2>&1 &
set -u
ROOT=/home/jxy/multiAgentRL
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
MANIFEST="${1:?usage: $0 <manifest> <mode> <tag> [n_seeds] [parallel]}"
MODE="${2:-fixed}"
TAG="${3:?usage: $0 <manifest> <mode> <tag> [n_seeds] [parallel]}"
N_SEEDS="${4:-12}"
PARALLEL="${5:-12}"
SHARD_DIR="$ROOT/logs/rank_temporal_shards/$TAG"
LOG="$ROOT/logs/rank_temporal_${TAG}.log"
mkdir -p "$SHARD_DIR"
log(){ echo "[temporal $TAG $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

mapfile -t SPECS < <(grep -v '^\s*#' "$MANIFEST" | grep -v '^\s*$')
if [ "${#SPECS[@]}" -eq 0 ]; then log "FATAL: empty manifest $MANIFEST"; exit 1; fi
CONT_ARGS=()
for s in "${SPECS[@]}"; do CONT_ARGS+=(--cont-spec "$s"); done
log "start mode=$MODE n_seeds=$N_SEEDS parallel=$PARALLEL conts=${SPECS[*]}"

run_one(){
  local seed=$1
  for attempt in 1 2 3; do
    if [ -f "$SHARD_DIR/shard_${seed}.json" ]; then log "seed $seed already done; skip"; return 0; fi
    log "seed $seed attempt $attempt"
    "$PY" "$ROOT/scripts/verify_rank_temporal.py" \
      --frozen-ckpt checkpoints/ovmse_paper_5m_vs_6m/ovmse_paper_seed1_best.pt \
      --mode "$MODE" "${CONT_ARGS[@]}" \
      --n-seeds 1 --seed0 "$seed" --waves greedy,epsgreedy --eps 0.3 \
      --k-cand 6 --margin 0.1 --max-probes 4 --max-joints 0 --k-near 64 \
      --device cpu --out "$SHARD_DIR/shard_${seed}.json" >> "$SHARD_DIR/shard_${seed}.log" 2>&1
    if [ $? -eq 0 ] && [ -f "$SHARD_DIR/shard_${seed}.json" ]; then
      log "seed $seed OK (attempt $attempt)"; return 0
    fi
    log "seed $seed FAILED attempt $attempt (rc=$?)"
    sleep 20
  done
  return 1
}

pids=(); s=1
while [ $s -le $N_SEEDS ]; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  run_one "$s" & pids+=($!)
  s=$((s+1)); sleep 3
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || { fail=1; log "worker pid=$pid failed"; }; done
log "shards done (any_fail=$fail)"

"$PY" - "$ROOT" "$SHARD_DIR" "$TAG" <<'PY'
import json,sys,glob,os
root, shard_dir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
from scripts.verify_rank_temporal import _temporal_summary
records=[]; meta=None
for f in sorted(glob.glob(os.path.join(shard_dir,"shard_*.json"))):
    try:
        d=json.load(open(f))
    except Exception as e:
        print("skip unreadable shard", f, e); continue
    records.extend(d.get("records",[])); meta=d
if meta is None:
    print("NO SHARDS MERGED"); sys.exit(1)
payload={k:meta.get(k) for k in ("config","frozen_checkpoint","conts","mode","waves","eps","k_cand","margin","k_near")}
payload["parameter_updates"]=0
payload["records"]=records
payload["summary"]=_temporal_summary(records)
out=os.path.join(root,"logs",f"rank_temporal_{tag}.json")
json.dump(payload, open(out,"w"), indent=2)
print("MERGED", len(records), "records ->", out)
for k,v in payload["summary"]["by_mode_cont"].items():
    print(k, json.dumps(v))
PY
log "merged -> logs/rank_temporal_${TAG}.json ; DONE"
