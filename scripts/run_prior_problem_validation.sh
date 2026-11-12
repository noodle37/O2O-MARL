#!/bin/bash
set -euo pipefail

ROOT=/home/jxy/O2O-MARL
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python
OUT="$ROOT/checkpoints/prior_problem_validation"
OFFLINE_DIR="$OUT/offline"
OFFLINE_CKPT="$OFFLINE_DIR/ovmse_paper_seed1_best.pt"
OFFLINE_FINAL="$OFFLINE_DIR/ovmse_paper_seed1.pt"
SEEDS=(1 2 3)
CHECKPOINT_STEPS=(250000 1000000 3000000)

cd "$ROOT"
mkdir -p "$OFFLINE_DIR" "$OUT"
export SC2PATH="${SC2PATH:-$ROOT/experiments/runtime/StarCraftII}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "$$" > "$OUT/launcher.pid"
echo "=== [$(date '+%F %T')] prior problem validation launcher pid=$$ ==="

if [[ ! -d "$SC2PATH/Maps/SMAC_Maps" && -d "$ROOT/experiments/runtime/MapsStaging/SMAC_Maps" ]]; then
  mkdir -p "$SC2PATH/Maps"
  cp -r "$ROOT/experiments/runtime/MapsStaging/SMAC_Maps" "$SC2PATH/Maps/"
fi
sc2_binaries=("$SC2PATH"/Versions/Base*/SC2_x64)
if [[ ! -x "${sc2_binaries[0]}" ]]; then
  echo "StarCraft II executable missing under $SC2PATH" >&2
  exit 3
fi
if [[ ! -f "$SC2PATH/Maps/SMAC_Maps/5m_vs_6m.SC2Map" ]]; then
  echo "SMAC map missing under $SC2PATH/Maps/SMAC_Maps" >&2
  exit 3
fi

if [[ ! -f "$OFFLINE_FINAL" ]]; then
  "$PY" train.py --config configs/5m_vs_6m.yaml \
    phase=offline seed=1 \
    checkpoint_dir="$OFFLINE_DIR" \
    checkpoint_path="$OFFLINE_CKPT" \
    behaviour_cache_dir="$OFFLINE_DIR/behaviour" \
    log_path="$OFFLINE_DIR/offline.jsonl" \
    system.data_device=cpu \
    eval_env_seed=424242
else
  echo "Using existing offline checkpoint: $OFFLINE_CKPT"
fi
if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "Offline training completed without a best checkpoint" >&2
  exit 4
fi

pids=()
for seed in "${SEEDS[@]}"; do
  RUN_DIR="$OUT/seed$seed"
  mkdir -p "$RUN_DIR"
  (
    echo "=== [$(date '+%F %T')] seed=$seed pid=$BASHPID ==="
    echo "$BASHPID" > "$RUN_DIR/pipeline.pid"
    ready=1
    for steps in "${CHECKPOINT_STEPS[@]}"; do
      if [[ ! -f "$RUN_DIR/periodic/ovm_diag_ckpt_t$steps.pt" ]]; then
        ready=0
      fi
    done
    if [[ "$ready" == "0" ]]; then
      "$PY" train.py --config configs/5m_vs_6m.yaml \
        phase=online seed="$seed" \
        checkpoint_path="$OFFLINE_CKPT" \
        checkpoint_dir="$RUN_DIR" \
        log_path="$RUN_DIR/online.jsonl" \
        behaviour_cache_dir="$OFFLINE_DIR/behaviour" \
        transfer.mode=full_td \
        system.mixing_ratio=0.0 \
        system.online_cql_weight=0.0 \
        system.sequential_explore=true \
        system.ovm_diag_train_stats=false \
        system.data_device=cpu \
        online.t_max=3000000 \
        online.test_interval=50000 \
        eval_interval_early=0 \
        eval_interval_late=0 \
        eval_switch_t=0 \
        eval_env_seed=424242 \
        num_eval_episodes=32 \
        ovm_diag.ckpt_every=250000 \
        pv_probe.enabled=false
    else
      echo "Using existing online checkpoints for seed=$seed"
    fi

    if [[ ! -f "$RUN_DIR/interactions.json" ]]; then
      ckpt_args=()
      for steps in "${CHECKPOINT_STEPS[@]}"; do
        ckpt_args+=(--checkpoint "t$steps=$RUN_DIR/periodic/ovm_diag_ckpt_t$steps.pt")
      done
      focal_args=()
      for ((i = 0; i < 5; i++)); do
        focal_args+=("single_$i=$i")
      done
      for ((i = 0; i < 5; i++)); do
        for ((j = i + 1; j < 5; j++)); do
          focal_args+=("pair_${i}_${j}=$i,$j")
        done
      done
      "$PY" scripts/run_e1_teammate_causal.py \
        --offline-ckpt "$OFFLINE_CKPT" \
        "${ckpt_args[@]}" \
        --focal "${focal_args[@]}" \
        --eps 0 \
        --episodes 64 \
        --eval-seed 424242 \
        --device cuda \
        --out "$RUN_DIR/interactions.json"
    else
      echo "Using existing interaction evaluation for seed=$seed"
    fi
    echo "=== [$(date '+%F %T')] seed=$seed DONE ==="
  ) > "$RUN_DIR/pipeline.out" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "Launched seed=$seed pid=$pid log=$RUN_DIR/pipeline.out"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" != "0" ]]; then
  echo "At least one seed failed; inspect seed*/pipeline.out" >&2
  exit "$status"
fi

input_args=()
for seed in "${SEEDS[@]}"; do
  input_args+=("seed$seed=$OUT/seed$seed/interactions.json")
done
"$PY" scripts/analyze_prior_problem.py \
  --input "${input_args[@]}" \
  --bootstrap 2000 \
  --out "$OUT/summary.json"

echo "=== [$(date '+%F %T')] all seeds and analysis DONE ==="
