#!/usr/bin/env bash
# Cross-arm / seed replications for Insights A-F (2026-10-24).
#
# Every job writes to its own JSON under checkpoints/extra_20261024/ and its own
# log under logs/extra_20261024/.  Re-running the script skips finished jobs.
# GPU0 is never used (it hosts the long-running 2halfcheetah PV exp1 jobs).
set -u

cd "$(dirname "$0")/.."
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
# The D (KMeans on 120k x 75 arrays) jobs abort with "double free or corruption"
# when OpenBLAS inherits all 152 cores; cap the BLAS/OpenMP pools.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

ROOT=checkpoints/extra_20261024
LOGS=logs/extra_20261024
mkdir -p "$ROOT" "$LOGS"
STATUS="$ROOT/job_status.jsonl"

run_job() {
  local gpu="$1" tag="$2" out="$3"
  shift 3
  if [ -e "$out" ]; then
    echo "[skip] $tag ($out exists)" | tee -a "$ROOT/launcher.log"
    return 0
  fi
  local log="$LOGS/${tag}.out"
  echo "[start] gpu=$gpu tag=$tag $(date -Iseconds)" | tee -a "$ROOT/launcher.log"
  printf '{"tag":"%s","gpu":"%s","out":"%s","start":"%s","cmd":%s}\n' \
    "$tag" "$gpu" "$out" "$(date -Iseconds)" \
    "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$log" 2>&1
  local rc=$?
  printf '{"tag":"%s","end":"%s","exit":%d,"log":"%s"}\n' \
    "$tag" "$(date -Iseconds)" "$rc" "$log" >> "$STATUS"
  echo "[done] tag=$tag exit=$rc $(date -Iseconds)" | tee -a "$ROOT/launcher.log"
}

A() {  # A <gpu> <tag> <arm>
  local gpu="$1" tag="$2" arm="$3"
  run_job "$gpu" "$tag" "$ROOT/${tag}.json" \
    "$PY" scripts/run_pv_exp2a.py --map 5m_vs_6m --seed 1 --arm "$arm" \
    --t-steps 1000000 --episodes 200 --skip-random --save-episodes \
    --out "$ROOT/${tag}.json"
}

B() {  # B <gpu> <tag> <arm> <t>
  local gpu="$1" tag="$2" arm="$3" t="$4"
  local ckpt="checkpoints/pv_exp1/5m_vs_6m/${arm}_seed1/periodic/ovm_diag_ckpt_t${t}.pt"
  run_job "$gpu" "$tag" "$ROOT/${tag}.json" \
    "$PY" scripts/run_verified_prefs.py --seed 1 --t "$t" --policy-ckpt "$ckpt" \
    --trials 1500 --max-skip 8 --out "$ROOT/${tag}.json"
}

E() {  # E <gpu> <tag> <arm> <t> <updates> <eval_eps>
  local gpu="$1" tag="$2" arm="$3" t="$4" updates="$5" eval_eps="$6"
  run_job "$gpu" "$tag" "$ROOT/${tag}.json" \
    "$PY" scripts/run_replay_gain.py --seed 1 --arm "$arm" --t-steps "$t" \
    --updates "$updates" --batch-eps 32 --eval-episodes "$eval_eps" --lr 1e-4 \
    --save-episodes --out "$ROOT/${tag}.json"
}

queue_gpu1() {
  A 1 A_armB_t1M B
  B 1 B_armB_t1M B 1000000
  E 1 E_A_t1M_lr1e4_u500_e200 A 1000000 500 200
  E 1 E_A_t250k_lr1e4_u500_e200 A 250000 500 200
}

queue_gpu2() {
  A 2 A_armC_t1M C
  B 2 B_armC_t1M C 1000000
  run_job 2 F_A_c96_students3_lr1e4 "$ROOT/F_A_c96_students3_lr1e4.json" \
    "$PY" scripts/run_collection_value.py --seed 1 --arm A \
    --students offline online_250k online_1m --collect-episodes 96 \
    --updates 30 100 --batch-eps 16 --eval-episodes 128 --lr 1e-4 \
    --out "$ROOT/F_A_c96_students3_lr1e4.json"
  E 2 E_B_t1M_lr1e4_u200_e128 B 1000000 200 128
}

queue_gpu3() {
  A 3 A_armD_t1M D
  B 3 B_armD_t1M D 1000000
  E 3 E_C_t1M_lr1e4_u200_e128 C 1000000 200 128
  B 3 B_armD_t250k D 250000
}

queue_cpu() {
  local tag split model epochs out
  for spec in "2 2 12" "3 3 12" "0 0 25"; do
    set -- $spec
    tag="C_split$1_model$2_e$3"
    out="$ROOT/$tag.json"
    run_job cpu "$tag" "$out" \
      "$PY" scripts/run_relational_premise.py --config configs/pv_base_5m_vs_6m.yaml \
      --epochs "$3" --split-seed "$1" --model-seed "$2" --device cpu --out "$out"
  done
  for seed in 1 2; do
    tag="D_seed${seed}_H3-5-10_K4-8-16"
    out="$ROOT/$tag.json"
    run_job cpu "$tag" "$out" \
      "$PY" scripts/run_goal_modality.py --config configs/pv_base_5m_vs_6m.yaml \
      --horizons 3 5 10 --ks 4 8 16 --seed "$seed" --out "$out"
  done
}

queue_gpu1 &
queue_gpu2 &
queue_gpu3 &
queue_cpu &
wait
echo "[all done] $(date -Iseconds)" | tee -a "$ROOT/launcher.log"
