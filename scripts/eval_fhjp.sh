#!/bin/bash
# 640-episode final-checkpoint evaluation for the FHJP arms, plus the OVMSE
# 2s3z online arm (no 640-episode endpoint exists for it yet), so that FHJP can
# be compared against OVMSE / UOT on the project's primary endpoint.
#
# Endpoint protocol is identical to scripts/eval_matched_protocol.sh:
# 640 episodes, chunk 32, Wilson CI, eval seed = the run's online seed.
#
# Usage:
#   nohup bash scripts/eval_fhjp.sh > logs/fhjp_eval640_master.log 2>&1 &
#
# Optional:
#   GPUS="1 2 3"  SKIP_EXISTING=0
#
# GPU 0 is never used: it hosts another user's long-running job.
set -uo pipefail

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

GPUS="${GPUS:-1 2 3}"
EPISODES="${EPISODES:-640}"
CHUNK="${CHUNK:-32}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# job spec: gpu_index config seed checkpoint out tag
# (gpu_index is 1-based into GPUS)
JOBS=(
  "1 configs/fhjp_2s3z.yaml     1 checkpoints/fhjp_2s3z/seed1/ovmse_paper_online_full_td_seed1.pt checkpoints/fhjp_2s3z/seed1/eval640_final.jsonl      fhjp_2s3z_seed1_final"
  "1 configs/fhjp_2s3z.yaml     2 checkpoints/fhjp_2s3z/seed2/ovmse_paper_online_full_td_seed2.pt checkpoints/fhjp_2s3z/seed2/eval640_final.jsonl      fhjp_2s3z_seed2_final"
  "1 configs/fhjp_2s3z.yaml     3 checkpoints/fhjp_2s3z/seed3/ovmse_paper_online_full_td_seed3.pt checkpoints/fhjp_2s3z/seed3/eval640_final.jsonl      fhjp_2s3z_seed3_final"
  "2 configs/fhjp_5m_vs_6m.yaml 1 checkpoints/fhjp_5m_vs_6m/seed1/ovmse_paper_online_full_td_seed1.pt checkpoints/fhjp_5m_vs_6m/seed1/eval640_final.jsonl fhjp_5m_vs_6m_seed1_final"
  "2 configs/fhjp_5m_vs_6m.yaml 2 checkpoints/fhjp_5m_vs_6m/seed2/ovmse_paper_online_full_td_seed2.pt checkpoints/fhjp_5m_vs_6m/seed2/eval640_final.jsonl fhjp_5m_vs_6m_seed2_final"
  "3 configs/2s3z.yaml          1 checkpoints/ovmse_batch_20261014/2s3z_online/ovmse_paper_online_ovmse_seed1.pt checkpoints/ovmse_batch_20261014/2s3z_online/eval640_final.jsonl ovmse_2s3z_seed1_final"
)

read -r -a GPU_LIST <<< "$GPUS"

log() { echo "=== [$(date '+%F %T')] $*"; }

run_one() {
  local gpu_idx="$1" cfg="$2" seed="$3" ckpt="$4" out="$5" tag="$6"
  local gpu="${GPU_LIST[$((gpu_idx - 1))]}"

  if [[ ! -f "$ckpt" ]]; then
    log "[skip] $tag (missing checkpoint $ckpt)"
    return 0
  fi
  if [[ "$SKIP_EXISTING" == "1" && -s "$out" ]] && grep -q '"checkpoint"' "$out"; then
    log "[skip] $tag (already evaluated: $out)"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  log "[start] gpu=$gpu tag=$tag ckpt=$ckpt"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_640.py --config "$cfg" \
    seed="$seed" phase=online task.dataset=Medium \
    --checkpoint "$ckpt" --episodes "$EPISODES" --chunk "$CHUNK" \
    --out "$out" --tag "$tag" > "${out%.jsonl}.out" 2>&1
  local rc=$?
  log "[done] tag=$tag rc=$rc out=$out"
  return "$rc"
}

fail=0
for spec in "${JOBS[@]}"; do
  read -r gi cfg seed ckpt out tag <<< "$spec"
  run_one "$gi" "$cfg" "$seed" "$ckpt" "$out" "$tag" &
done
wait
log "all evals finished"
