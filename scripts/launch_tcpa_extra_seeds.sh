#!/usr/bin/env bash
# Extra TCPA seeds (2 and 3) for the main arm, for every ablation, and for the
# cross-map 2s3z arm, on GPUs 1-3.  GPU 0 is left alone for the other workloads.
#
#   nohup bash scripts/launch_tcpa_extra_seeds.sh > logs/tcpa/extra_seeds.log 2>&1 &
#
# Optional: SEEDS="2 3"  CONCURRENCY=3  T_MAX=3000000  LOGS=<dir>
#
# Every job is one (map, variant, seed) triple and reuses the protocol of the
# seed-1 runs: same config, same CFCQL-D offline prior, same 3M online budget,
# 8 parallel envs, evaluation every 10k steps with 32 episodes.  Only the online
# seed changes (env seeds, exploration, replay sampling), so the extra seeds are
# directly comparable to the seed-1 arms and to the O2O baselines.
#
# The offline prior is the shared seed-1 checkpoint (only one exists per map),
# so a seed here varies the online phase, not the offline pretraining.
set -uo pipefail

cd "$(dirname "$0")/.."
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

PY="${PYTHON:-/home/jxy/miniconda3/envs/multiAgentRL/bin/python}"
SEEDS="${SEEDS:-2 3}"
CONCURRENCY="${CONCURRENCY:-3}"
T_MAX="${T_MAX:-3000000}"
LOGS="${LOGS:-logs/tcpa/extra_seeds}"
STATUS="$LOGS/job_status.jsonl"
JOBS="$LOGS/jobs.txt"

mkdir -p "$LOGS"
log() { echo "=== [$(date '+%F %T')] $*" | tee -a "$LOGS/launcher.log"; }

# One arm per GPU.  GPU 1 keeps the main arm and the IGM ablation, GPU 2 the
# gate and absolute-Q ablations, GPU 3 the residual and falsification arms.
ARMS=(
  "1|5m_vs_6m|full"
  "1|5m_vs_6m|no_igm"
  "2|5m_vs_6m|fixed_gate"
  "2|5m_vs_6m|abs_q"
  "3|5m_vs_6m|no_residual"
  "3|5m_vs_6m|gate_anti"
)

: > "$JOBS"
for seed in $SEEDS; do
  for arm in "${ARMS[@]}"; do
    echo "${arm}|${seed}" >> "$JOBS"
  done
done
# Cross-map check: the 2s3z arm is split over GPU 1 and GPU 2 so no card gets
# the longer queue.
echo "1|2s3z|full|2" >> "$JOBS"
echo "2|2s3z|full|3" >> "$JOBS"

log "code HEAD: $(git rev-parse --short HEAD)"
log "seeds=[$SEEDS] concurrency=${CONCURRENCY} per GPU, t_max=$T_MAX"
log "jobs: $(wc -l < "$JOBS")"

run_one() {
  local gpu="$1" map="$2" variant="$3" seed="$4"
  local cfg="configs/tcpa_${map}.yaml"
  local out="checkpoints/tcpa_${map}/${variant}/seed${seed}"
  local tag="${map}/${variant}/seed${seed}"

  if [[ ! -f "$cfg" ]]; then
    log "[FAIL] $tag: config missing ($cfg)"
    printf '{"tag":"%s","gpu":"%s","end":"%s","exit":%s}\n' \
      "$tag" "$gpu" "$(date -Iseconds)" 3 >> "$STATUS"
    return
  fi

  mkdir -p "$out"
  log "[start] $tag gpu=$gpu"
  printf '{"tag":"%s","gpu":"%s","start":"%s"}\n' \
    "$tag" "$gpu" "$(date -Iseconds)" >> "$STATUS"

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train.py --config "$cfg" \
    "tcpa.variant=${variant}" \
    "seed=${seed}" \
    "t_max=${T_MAX}" \
    "online.t_max=${T_MAX}" \
    "checkpoint_dir=${out}" \
    "log_path=${out}/online.jsonl" \
    > "${out}/train.out" 2>&1
  local rc=$?

  if [[ -s "${out}/online.jsonl" ]]; then
    log "[done] $tag gpu=$gpu exit=$rc"
  else
    log "[FAIL] $tag gpu=$gpu exit=$rc (no log written); see ${out}/train.out"
  fi
  printf '{"tag":"%s","gpu":"%s","end":"%s","exit":%s}\n' \
    "$tag" "$gpu" "$(date -Iseconds)" "$rc" >> "$STATUS"
}

# One queue per GPU; each queue keeps CONCURRENCY runs alive at a time.
queue_gpu() {
  local gpu="$1"
  local running=0
  while IFS='|' read -r g m v s; do
    [[ "$g" == "$gpu" ]] || continue
    run_one "$g" "$m" "$v" "$s" &
    running=$((running + 1))
    if (( running >= CONCURRENCY )); then
      wait -n
      running=$((running - 1))
    fi
  done < "$JOBS"
  wait
}

for gpu in 1 2 3; do
  queue_gpu "$gpu" &
  log "gpu${gpu} queue pid $!"
done

wait
log "[all done]"
