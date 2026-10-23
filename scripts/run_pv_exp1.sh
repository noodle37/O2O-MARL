#!/bin/bash
# Problem-validation experiment 1: single-arm online run.
#
# Usage: run_pv_exp1.sh <map> <arm> <seed> <gpu> [extra overrides...]
#
# Every arm starts from the same offline checkpoint, uses the same online recipe,
# the same sequential exploration, the same fixed evaluation seed and the same
# 1M-step budget.  The only difference is what is retained from the offline model:
#   A  direct fine-tune        plain TD, nothing retained
#   B  continual conservative  plain TD + online CQL term (constraint retained)
#   C  OVMSE                   offline value memory (whole net retained)
#   D  scale correction        arm A recipe from an affine-rescaled checkpoint
#   Z  from scratch            no offline weights at all (protocol check)
#
# Maps: 5m_vs_6m (cfcql SMAC), 2halfcheetah (og-marl MuJoCo, discrete torque bins).
# The 2halfcheetah map has no win rate; the exp1 metrics fall back to the mean
# episode return there.
#
# PV_SMOKE=1 redirects the run directory to checkpoints/pv_smoke/... so short
# validation runs never touch the real experiment output.
set -euo pipefail

MAP="${1:?usage: run_pv_exp1.sh <map> <arm> <seed> <gpu> [overrides...]}"
ARM="${2:?missing arm}"
SEED="${3:?missing seed}"
GPU="${4:?missing gpu}"
shift 4

cd /home/jxy/multiAgentRL
export SC2PATH=/home/jxy/StarCraftII
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
PY=/home/jxy/miniconda3/envs/multiAgentRL/bin/python

case "$MAP" in
  5m_vs_6m)     CFG=configs/pv_base_5m_vs_6m.yaml ;;
  2halfcheetah) CFG=configs/pv_base_2halfcheetah.yaml ;;
  *) echo "unknown map '$MAP' (expected 5m_vs_6m or 2halfcheetah)" >&2; exit 2 ;;
esac

case "$MAP" in
  5m_vs_6m)
    OFFLINE_CKPT=checkpoints/ovmse_batch_20261014/5m_vs_6m_offline/ovmse_paper_seed1_best.pt
    ;;
  2halfcheetah)
    # The offline_final checkpoint: on this map "best-so-far" degenerated to the
    # untrained t=0 snapshot (every later eval was worse), which would make arm Z
    # indistinguishable from the other arms.
    OFFLINE_CKPT=checkpoints/pv_exp1_offline/2halfcheetah_medium/ovmse_paper_seed1.pt
    ;;
esac

if [[ ! -f "$OFFLINE_CKPT" ]]; then
  echo "FATAL: shared offline checkpoint missing: $OFFLINE_CKPT" >&2
  exit 3
fi

if [[ "${PV_SMOKE:-0}" == "1" ]]; then
  RUN_DIR="checkpoints/pv_smoke/${MAP}/${ARM}_seed${SEED}"
else
  RUN_DIR="checkpoints/pv_exp1/${MAP}/${ARM}_seed${SEED}"
fi
SCALED_CKPT="checkpoints/pv_exp1_D_ckpt/${MAP}/D_seed${SEED}_scaled.pt"
mkdir -p "$RUN_DIR"

# Arm-specific overrides.  Everything not listed here comes from the shared base
# config, so the arms differ only in what is retained.
OVERRIDES=()
case "$ARM" in
  A)
    : # base defaults: plain TD, nothing retained
    ;;
  B)
    # Keep the conservative term online, with the same weight the offline CFCQL
    # phase used (50.0).  The behaviour prior comes from the offline cache.
    OVERRIDES+=(system.online_cql_weight=50.0)
    ;;
  C)
    OVERRIDES+=(transfer.mode=ovmse system.use_ovm=true online.use_ovm=true)
    ;;
  D)
    # Same recipe as arm A, started from the affine-rescaled checkpoint.  The
    # frozen offline reference stays the unscaled file so every arm's "offline"
    # metrics refer to the same network (a > 0 keeps the greedy policy identical).
    if [[ ! -f "$SCALED_CKPT" ]]; then
      echo "FATAL: arm D start checkpoint missing: $SCALED_CKPT" >&2
      echo "       build it with scripts/make_scale_corrected_ckpt.py" >&2
      exit 4
    fi
    OVERRIDES+=(checkpoint_path="$SCALED_CKPT" transfer.offline_reference_ckpt="$OFFLINE_CKPT")
    ;;
  Z)
    # No offline weights at all: skip the load, drop the frozen offline nets and
    # the diagnostics that need them.
    OVERRIDES+=(transfer.mode=scratch transfer.load_offline_nets=false system.ovm_diag_train_stats=false)
    ;;
  *)
    echo "unknown arm '$ARM' (expected A|B|C|D|Z)" >&2
    exit 2
    ;;
esac

echo "=== [$(date '+%F %T')] pv exp1 map=$MAP arm=$ARM seed=$SEED gpu=$GPU ==="
echo "config:   $CFG"
echo "offline:  $OFFLINE_CKPT"
echo "run dir:  $RUN_DIR"
echo "overrides: ${OVERRIDES[*]-} $*"

"$PY" train.py --config "$CFG" \
  seed="$SEED" \
  checkpoint_dir="$RUN_DIR" \
  log_path="$RUN_DIR/online.jsonl" \
  pv_probe.out="$RUN_DIR/probe.jsonl" \
  "${OVERRIDES[@]}" \
  "$@"

echo "=== [$(date '+%F %T')] pv exp1 map=$MAP arm=$ARM seed=$SEED DONE ==="
