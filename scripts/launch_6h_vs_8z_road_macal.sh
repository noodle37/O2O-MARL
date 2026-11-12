#!/usr/bin/env bash
# 6h_vs_8z two-algorithm batch on GPU1 / GPU3, three seeds each, all parallel.
#
#   GPU1  MaROAD   full O2O: per-seed OVMSE offline pretraining, then online
#                  from that seed's own offline best checkpoint.
#   GPU3  MaCal-QL continuous O2O: no offline weights, trained from scratch on
#                  every batch mixed 50/50 offline vault + online rollouts.
#
# The 6h_vs_8z CFCQL vault is not present in this workspace yet. The launcher
# downloads it through the HF mirror (curl, retrying on 429/5xx) before any
# training job starts, so the whole thing can be launched under nohup and the
# terminal closed.
#
# Usage:
#   nohup bash scripts/launch_6h_vs_8z_road_macal.sh \
#     > logs/6h_vs_8z_road_macal_master.log 2>&1 &
#
# Overrides: SEEDS ROAD_GPU MACAL_GPU ROOT LOGS OFFLINE_TMAX ONLINE_TMAX
set -uo pipefail

cd "$(dirname "$0")/.."
export SC2PATH="${SC2PATH:-/home/jxy/StarCraftII}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

PY="${PY:-/home/jxy/miniconda3/envs/multiAgentRL/bin/python}"
SEEDS="${SEEDS:-1 2 3}"
ROAD_GPU="${ROAD_GPU:-1}"
MACAL_GPU="${MACAL_GPU:-3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${ROOT:-checkpoints/6h_vs_8z_road_macal_${STAMP}}"
LOGS="${LOGS:-logs/6h_vs_8z_road_macal_${STAMP}}"
OFFLINE_TMAX="${OFFLINE_TMAX:-10000000}"
ONLINE_TMAX="${ONLINE_TMAX:-3000000}"
ONLY="${ONLY:-all}"
VAULT_DIR="vaults/cfcql/smac_v1/6h_vs_8z.vlt"
VAULT_URL="https://hf-mirror.com/datasets/InstaDeepAI/og-marl/resolve/main/prior_work/cfcql/smac_v1/6h_vs_8z.zip"

mkdir -p "$ROOT" "$LOGS" vaults/tmp
STATUS="$ROOT/job_status.jsonl"

log() { echo "=== [$(date '+%F %T')] $*"; }
status() { printf '%s\n' "$1" >> "$STATUS"; }

log "code HEAD: $(git rev-parse --short HEAD)"
log "seeds=[$SEEDS] road_gpu=$ROAD_GPU macal_gpu=$MACAL_GPU"
log "offline_tmax=$OFFLINE_TMAX online_tmax=$ONLINE_TMAX"
log "root=$ROOT logs=$LOGS"

# ---------------------------------------------------------------- vault ----
VAULT_BYTES=364773563

unpack_vault() {
  local zip="$1"
  unzip -q -o "$zip" -d vaults/cfcql/smac_v1 || return 1
  # Normalise nested archive layouts to vaults/cfcql/smac_v1/6h_vs_8z.vlt.
  if [[ ! -d "$VAULT_DIR" ]]; then
    local found
    found="$(find vaults/cfcql/smac_v1 -type d -name '6h_vs_8z.vlt' | head -1)"
    if [[ -n "$found" && "$found" != "$VAULT_DIR" ]]; then
      mv "$found" "$VAULT_DIR"
    fi
  fi
  [[ -d "$VAULT_DIR/Medium" ]]
}

ensure_vault() {
  if [[ -d "$VAULT_DIR" && -n "$(ls -A "$VAULT_DIR" 2>/dev/null)" ]]; then
    log "[vault] present: $VAULT_DIR"
    return 0
  fi
  # The archive carries four splits but only Medium is read, so pull just the
  # Medium byte span instead of the full ~348 MB blob.
  log "[vault] selective Medium-only extract"
  if "$PY" scripts/fetch_cfcql_6h_medium.py && [[ -f "$VAULT_DIR/Medium/metadata.json" ]]; then
    log "[vault] ready: $VAULT_DIR"
    return 0
  fi
  log "[vault] selective extract failed; falling back to full zip download"
  local zip="vaults/tmp/6h_vs_8z.zip"
  local attempt sz
  for attempt in $(seq 1 500); do
    sz="$(stat -c%s "$zip" 2>/dev/null || echo 0)"
    if (( sz >= VAULT_BYTES )); then
      if unzip -tq "$zip" >/dev/null 2>&1 && unpack_vault "$zip"; then
        log "[vault] ready: $VAULT_DIR"
        rm -f "$zip"
        return 0
      fi
      log "[vault] corrupt/incomplete zip (${sz} bytes); restarting"
      rm -f "$zip"
      sz=0
    fi
    log "[vault] attempt $attempt resume_at=${sz}/${VAULT_BYTES} bytes"
    # hf-mirror rate-limits and the CAS bridge drops long streams; resume with
    # Range (-C -) so a reset never discards the bytes already on disk.
    curl -L --fail -C - --http1.1 \
      --retry 5 --retry-delay 15 --retry-all-errors \
      --connect-timeout 30 --max-time 2400 \
      --speed-limit 1024 --speed-time 180 \
      -o "$zip" "$VAULT_URL" || true
    sleep 60
  done
  log "[vault] FATAL: could not obtain $VAULT_DIR"
  return 1
}

if ! ensure_vault; then
  status '{"job":"vault","state":"failed"}'
  exit 2
fi

# MaCal-QL scratch runs need an existing file only to satisfy the checkpoint
# resolver; transfer.mode=scratch never loads its weights.
ANCHOR="$ROOT/scratch_anchor.pt"
if [[ ! -f "$ANCHOR" ]]; then
  "$PY" -c "import sys, torch; torch.save({'scratch': True}, sys.argv[1])" "$ANCHOR"
fi

# ---------------------------------------------------------------- jobs -----
run_road_chain() {  # $1=seed
  local seed="$1"
  local base="$ROOT/road_6h_vs_8z/seed${seed}"
  local off="$base/offline"
  local on="$base/online"
  local best="$off/ovmse_paper_seed${seed}_best.pt"
  local final="$on/ovmse_paper_online_full_td_seed${seed}.pt"
  local tag="road_seed${seed}"
  mkdir -p "$off" "$on"

  if [[ -f "$final" ]]; then
    log "[$tag] skip (finished: $final)"
    return 0
  fi

  if [[ ! -f "$best" ]]; then
    status "{\"job\":\"$tag\",\"phase\":\"offline\",\"gpu\":\"$ROAD_GPU\",\"seed\":$seed,\"start\":\"$(date -Iseconds)\"}"
    log "[$tag] offline start gpu=$ROAD_GPU dir=$off"
    CUDA_VISIBLE_DEVICES="$ROAD_GPU" "$PY" train.py --config configs/6h_vs_8z.yaml \
      phase=offline \
      "seed=$seed" \
      "task.dataset=Medium" \
      "t_max=$OFFLINE_TMAX" \
      "batch_size=128" "batch_size_run=8" \
      "test_interval=10000" "num_eval_episodes=32" \
      save_best_checkpoint=true \
      "system_name=OVMSE 6h_vs_8z offline seed$seed" \
      "checkpoint_dir=$off" \
      "behaviour_cache_dir=$off/behaviour" \
      "log_path=$off/offline.jsonl" \
      > "$LOGS/${tag}_offline.out" 2>&1
    local rc=$?
    log "[$tag] offline done rc=$rc"
    if (( rc != 0 )); then
      status "{\"job\":\"$tag\",\"phase\":\"offline\",\"rc\":$rc,\"end\":\"$(date -Iseconds)\"}"
      return "$rc"
    fi
  else
    log "[$tag] reuse offline best $best"
  fi

  if [[ ! -f "$best" ]]; then
    local found
    found="$(find "$off" -maxdepth 1 -name "ovmse_paper_seed*_best.pt" | head -1)"
    [[ -n "$found" ]] || { log "[$tag] FATAL: no offline best under $off"; return 1; }
    best="$found"
  fi

  status "{\"job\":\"$tag\",\"phase\":\"online\",\"gpu\":\"$ROAD_GPU\",\"seed\":$seed,\"start\":\"$(date -Iseconds)\"}"
  log "[$tag] online start gpu=$ROAD_GPU from=$best dir=$on"
  CUDA_VISIBLE_DEVICES="$ROAD_GPU" "$PY" train.py --config configs/road_6h_vs_8z.yaml \
    phase=online \
    "seed=$seed" \
    "task.dataset=Medium" \
    "online.t_max=$ONLINE_TMAX" "t_max=$ONLINE_TMAX" \
    "batch_size=128" "batch_size_run=8" \
    "test_interval=10000" "num_eval_episodes=32" \
    "checkpoint_dir=$on" \
    "checkpoint_path=$best" \
    "log_path=$on/online.jsonl" \
    "system_name=MaROAD 6h_vs_8z seed$seed" \
    > "$LOGS/${tag}_online.out" 2>&1
  local rc=$?
  log "[$tag] online done rc=$rc"
  status "{\"job\":\"$tag\",\"phase\":\"online\",\"rc\":$rc,\"end\":\"$(date -Iseconds)\"}"
  return "$rc"
}

run_macal() {  # $1=seed
  local seed="$1"
  local dir="$ROOT/macal_6h_vs_8z/seed${seed}"
  local final="$dir/macal_online_scratch_seed${seed}.pt"
  local tag="macal_seed${seed}"
  mkdir -p "$dir"

  if [[ -f "$final" ]]; then
    log "[$tag] skip (finished: $final)"
    return 0
  fi

  status "{\"job\":\"$tag\",\"phase\":\"joint\",\"gpu\":\"$MACAL_GPU\",\"seed\":$seed,\"start\":\"$(date -Iseconds)\"}"
  log "[$tag] joint start gpu=$MACAL_GPU dir=$dir"
  CUDA_VISIBLE_DEVICES="$MACAL_GPU" "$PY" train.py --config configs/macal_6h_vs_8z_joint.yaml \
    phase=online \
    "seed=$seed" \
    "task.dataset=Medium" \
    "online.t_max=$ONLINE_TMAX" "t_max=$ONLINE_TMAX" \
    "batch_size=128" "batch_size_run=8" \
    "test_interval=10000" "num_eval_episodes=32" \
    "checkpoint_dir=$dir" \
    "checkpoint_path=$ANCHOR" \
    "log_path=$dir/online.jsonl" \
    "system_name=MaCal-QL 6h_vs_8z joint seed$seed" \
    > "$LOGS/${tag}.out" 2>&1
  local rc=$?
  log "[$tag] joint done rc=$rc"
  status "{\"job\":\"$tag\",\"phase\":\"joint\",\"rc\":$rc,\"end\":\"$(date -Iseconds)\"}"
  return "$rc"
}

pids=()
tags=()
if [[ "$ONLY" == "all" || "$ONLY" == "road" ]]; then
  for seed in $SEEDS; do
    ( run_road_chain "$seed" ) & pids+=("$!"); tags+=("road_seed${seed}")
  done
fi
if [[ "$ONLY" == "all" || "$ONLY" == "macal" ]]; then
  for seed in $SEEDS; do
    ( run_macal "$seed" ) & pids+=("$!"); tags+=("macal_seed${seed}")
  done
fi

log "launched ${#pids[@]} jobs: ${tags[*]}"
fail=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    log "[done] ${tags[$idx]}"
  else
    rc=$?
    log "[FAIL] ${tags[$idx]} exit=$rc (see $LOGS)"
    fail=1
  fi
done

log "all jobs finished fail=$fail"
status "{\"job\":\"master\",\"state\":\"finished\",\"fail\":$fail,\"end\":\"$(date -Iseconds)\"}"
exit "$fail"
