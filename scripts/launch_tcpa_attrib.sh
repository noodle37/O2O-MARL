#!/usr/bin/env bash
# Launch the TCPA constraint-attribution matrix on one GPU.
#
#   bash scripts/launch_tcpa_attrib.sh            # full matrix, GPU 3
#   GPU=1 MAXPAR=2 bash scripts/launch_tcpa_attrib.sh
#   T_MAX=20000 ROOT=checkpoints/_smoke/tcpa_attrib bash scripts/launch_tcpa_attrib.sh
#
# Design: 4 constraint levels x 2 teammate-cooperation conditions = 8 arms,
# plus 2 exploratory arms (see below) = 10 in total.
#
#   arm              tcpa.variant   coop.mode         what it isolates
#   none_matched     no_igm         default           no constraint, cooperative team
#   none_broken      no_igm         frozen_offline    no constraint, team does not match
#   anchor_matched   value_anchor   default           generic trust region (H1)
#   anchor_broken    value_anchor   frozen_offline    generic trust region, team does not match
#   placebo_matched  placebo        default           same hinge, meaningless target
#   placebo_broken   placebo        frozen_offline    same hinge, meaningless target
#   prior_matched    full           default           real TCPA, cooperative team
#   prior_broken     full           frozen_offline    real TCPA, team does not match
#   fixed_gate_matched  fixed_gate  default           unconditional old-action retention
#   fixed_gate_broken   fixed_gate  frozen_offline    ... with the team not matching
#
# The decisive readouts are the two interaction terms
#   (prior - none)_matched - (prior - none)_broken     -> is the old action team-specific?
#   (anchor - none)_matched - (anchor - none)_broken   -> is a generic constraint universal?
# and the time profile of each contrast (early vs late).
#
# The two fixed_gate arms are EXPLORATORY and are not part of the pre-registered
# decision rule: the finished TCPA matrix shows the dynamic gate of `full`
# suppresses its own penalty exactly where it would bite (effective prior loss
# 0.0041 against 0.0126 for an unconditional gate), so Δ_prior in the early
# window is ~0 and the pre-registered interaction has little to measure.
# `fixed_gate` is the arm with a demonstrated early effect (+0.59 over three
# seeds), so its team-dependence is the informative version of the question.
#
# Robustness: card 3 is shared with another user's job that already holds
# ~45 GB, so an arm can hit a CUDA OOM at any time.  The launcher therefore
# (a) admits a new arm only when the card has MIN_FREE_MIB of headroom,
# (b) retries a failed arm up to RETRIES times with a backoff,
# (c) keeps the arms in priority order, so the decisive
# prior/none x matched/broken factorial finishes first, and
# (d) skips arms that are already running or already finished, so the
# launcher can be restarted without starting a duplicate arm.
set -uo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-3}"
# Card 3 is shared: another user's job holds ~45 GB, which leaves ~2.6 GB.
# One arm costs ~774 MiB, so three fit with ~290 MiB of headroom; a fourth
# does not fit and is what OOM'd the first smoke.  MAXPAR is an upper bound,
# not a target: an arm is admitted only when MIN_FREE_MIB is available.
MAXPAR="${MAXPAR:-4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-1000}"
SEED="${SEED:-1}"
T_MAX="${T_MAX:-1000000}"
RETRIES="${RETRIES:-4}"
RETRY_SLEEP="${RETRY_SLEEP:-600}"
PYTHON="${PYTHON:-/home/jxy/miniconda3/envs/multiAgentRL/bin/python}"
CONFIG="${CONFIG:-configs/tcpa_attrib_5m_vs_6m.yaml}"
ROOT="${ROOT:-checkpoints/tcpa_attrib_5m_vs_6m}"
LOGDIR="${LOGDIR:-logs/tcpa_attrib}"
STATUS="${STATUS:-${LOGDIR}/status.tsv}"

# arm name | tcpa.variant | coop.mode  (priority order: the decisive factorial first)
ARMS=(
  "none_matched|no_igm|default"
  "prior_matched|full|default"
  "none_broken|no_igm|frozen_offline"
  "prior_broken|full|frozen_offline"
  "anchor_matched|value_anchor|default"
  "fixed_gate_matched|fixed_gate|default"
  "fixed_gate_broken|fixed_gate|frozen_offline"
  "placebo_matched|placebo|default"
  "anchor_broken|value_anchor|frozen_offline"
  "placebo_broken|placebo|frozen_offline"
)

# Frozen teammates for the broken condition: every agent except the focal one
# (agent 0).  Override with FROZEN="1 2 3 4".
FROZEN="${FROZEN:-1 2 3 4}"
FROZEN_CSV="$(echo "$FROZEN" | tr ' ' ',')"

# Extra hydra-style overrides, e.g. EXTRA="system.device=cpu" for a smoke run
# that must not touch the (nearly full) shared card.
EXTRA="${EXTRA:-}"
# Optional arm filter, e.g. ONLY="placebo_matched,prior_broken".
ONLY="${ONLY:-}"

mkdir -p "$LOGDIR"

# Only one launcher may manage the queue at a time.  The watcher can restart a
# dead launcher, and a manual handover is possible too; without this lock both
# could admit the same arm at the same moment and two processes would write
# into one log.  flock is released automatically when the launcher exits.
exec 9>"${LOGDIR}/launcher.lock"
if ! flock -n 9; then
  echo "[attrib] another launcher already holds ${LOGDIR}/launcher.lock; exiting"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU"
# The card is shared and nearly full; expandable segments keeps the allocator
# from fragmenting a ~800 MiB footprint into several unusable blocks.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The vault is read through flashbax/jax.  On the GPU, jax's XLA backend
# pre-allocates a large slice of the card for a read that needs none of it,
# which is what made concurrent arms die with a CUDA OOM here.  Keep jax on
# the CPU; the dataset is small and the learner itself still uses the GPU.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" \
    2>/dev/null | head -1 | tr -d ' '
}

arm_out() {
  echo "$ROOT/$1/seed${SEED}"
}

# Is this arm's training process alive?  The pattern is the arm's checkpoint
# directory, which appears verbatim in the training command line.
arm_running() {
  local out
  out="$(arm_out "$1")"
  pgrep -f "checkpoint_dir=${out}" >/dev/null 2>&1
}

# Has this arm already reached the budget?  The log is append-only, so a
# restarted arm leaves an earlier run in front of it; only the last run counts.
arm_finished() {
  local out
  out="$(arm_out "$1")"
  "$PYTHON" scripts/tcpa_arm_state.py \
    --log "${out}/online.jsonl" --t-max "$T_MAX" 2>/dev/null | grep -q "^done$"
}

count_running_arms() {
  local n=0 spec arm
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r arm _variant _coop <<<"$spec"
    if arm_running "$arm"; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

run_arm() {
  local arm="$1" variant="$2" coop="$3"
  local out="$ROOT/$arm/seed${SEED}"
  local attempt=1
  while [ "$attempt" -le "$RETRIES" ]; do
    mkdir -p "$out"
    echo -e "$(date '+%F %T')\t${arm}\tattempt=${attempt}\tstart" >> "$STATUS"
    echo "[attrib] start arm=$arm variant=$variant coop=$coop gpu=${GPU} attempt=${attempt} -> $out"
    if "$PYTHON" train.py --config "$CONFIG" \
        "tcpa.variant=${variant}" \
        "coop.mode=${coop}" \
        "coop.frozen_agents=[${FROZEN_CSV}]" \
        "seed=${SEED}" \
        "t_max=${T_MAX}" \
        "online.t_max=${T_MAX}" \
        "checkpoint_dir=${out}" \
        "log_path=${out}/online.jsonl" \
        ${EXTRA} \
        > "${out}/train.out" 2>&1; then
      echo -e "$(date '+%F %T')\t${arm}\tattempt=${attempt}\tdone" >> "$STATUS"
      echo "[attrib] done  arm=$arm"
      return 0
    fi
    status=$?
    echo -e "$(date '+%F %T')\t${arm}\tattempt=${attempt}\tfailed(${status})" >> "$STATUS"
    echo "[attrib] FAILED arm=$arm exit=${status} (attempt ${attempt}/${RETRIES})" >&2
    if [ "$attempt" -lt "$RETRIES" ]; then
      echo "[attrib] retrying arm=$arm in ${RETRY_SLEEP}s" >&2
      sleep "$RETRY_SLEEP"
    fi
    attempt=$((attempt + 1))
  done
  echo -e "$(date '+%F %T')\t${arm}\tattempt=${attempt}\tabandoned" >> "$STATUS"
  return 1
}

echo -e "$(date '+%F %T')\tlauncher\tstart gpu=${GPU} maxpar=${MAXPAR} t_max=${T_MAX}" >> "$STATUS"

for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm variant coop <<<"$spec"
  # Optional arm filter, e.g. ONLY="placebo_matched,prior_broken".
  if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",${arm},"; then
    continue
  fi
  # Skip arms that already ran to the budget, and arms that are already
  # running (for instance after a launcher restart): starting a second copy
  # of an arm would corrupt its log.
  if arm_finished "$arm"; then
    echo "[attrib] skip arm=$arm (already finished)"
    echo -e "$(date '+%F %T')\t${arm}\tskipped\tfinished" >> "$STATUS"
    continue
  fi
  if arm_running "$arm"; then
    echo "[attrib] skip arm=$arm (already running)"
    echo -e "$(date '+%F %T')\t${arm}\tskipped\trunning" >> "$STATUS"
    continue
  fi
  # Wait for a free slot *and* for the card to have enough headroom.
  while true; do
    running="$(count_running_arms)"
    free="$(gpu_free_mib)"
    if [ "$running" -lt "$MAXPAR" ] && [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MIB" ]; then
      break
    fi
    sleep 30
  done
  run_arm "$arm" "$variant" "$coop" &
  sleep 20  # let the new process claim its memory before admitting the next one
done
wait

echo "[attrib] all arms finished: $ROOT"
