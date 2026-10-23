#!/usr/bin/env python3
"""Periodic-checkpoint trigger: fire on crossing a ckpt_every boundary.

The training loop advances t_env by whole episodes. On variable-length maps
(SMAC) t_env almost never lands exactly on a multiple of ckpt_every, so the old
condition (comparing consecutive integer divisions) silently never fired and no
periodic checkpoint was ever written — which starved the task-2 temporal
measurement of its intermediate online checkpoints. These tests pin the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.online import _periodic_boundary

CKPT_EVERY = 500_000


def _smac_increments(total: int):
    """Episode lengths in the SMAC 5m_vs_6m range (deterministic, varied)."""
    lengths = [12, 27, 18, 33, 21, 15, 29, 24, 19, 31, 16, 26]
    i = 0
    t = 0
    while t < total:
        t += lengths[i % len(lengths)]
        i += 1
        yield t


def _old_condition_fires(t_env: int) -> bool:
    """The pre-fix trigger, kept here to document the bug it caused."""
    return t_env // CKPT_EVERY != (t_env - 1) // CKPT_EVERY


def _simulate(increments, t_start: int = 0):
    """Mirror the training loop's checkpoint bookkeeping; return (t_env, name) fires."""
    last = t_start // CKPT_EVERY if CKPT_EVERY > 0 else -1
    fires = []
    for t in increments:
        boundary = _periodic_boundary(t, CKPT_EVERY, last)
        if boundary is not None:
            last = boundary
            fires.append((t, f"ovm_diag_ckpt_t{boundary * CKPT_EVERY}.pt"))
    return fires


def test_smac_variable_lengths_fire_once_per_boundary():
    total = 3_200_000
    # The old condition misses every boundary t_env jumps over: over a full
    # 3M-step SMAC run it fires zero times.
    assert not any(_old_condition_fires(t) for t in _smac_increments(total))

    fires = _simulate(_smac_increments(total))
    assert [name for _, name in fires] == [
        "ovm_diag_ckpt_t500000.pt",
        "ovm_diag_ckpt_t1000000.pt",
        "ovm_diag_ckpt_t1500000.pt",
        "ovm_diag_ckpt_t2000000.pt",
        "ovm_diag_ckpt_t2500000.pt",
        "ovm_diag_ckpt_t3000000.pt",
    ]
    # Each fire lands just past its boundary; the true t_env stays available
    # to the caller (it goes into the checkpoint's `extra`).
    for (t_env, name), multiple in zip(fires, range(1, 7)):
        assert multiple * CKPT_EVERY <= t_env < (multiple + 1) * CKPT_EVERY


def test_exact_multiple_still_fires():
    # Mamujoco has fixed 1000-step episodes, so t_env used to land exactly on
    # the multiple; the fix must not change that behaviour or its filename.
    fires = _simulate(range(0, 1_100_000, 1000))
    assert [name for _, name in fires] == [
        "ovm_diag_ckpt_t500000.pt",
        "ovm_diag_ckpt_t1000000.pt",
    ]
    assert [t for t, _ in fires] == [500_000, 1_000_000]


def test_resume_mid_interval_does_not_refire():
    # A branch resuming at t=1,009,732 is already inside interval 2: it must
    # not immediately re-save that interval, only the next boundary.
    fires = _simulate(_smac_increments(1_600_000), t_start=1_009_732)
    assert [name for _, name in fires] == ["ovm_diag_ckpt_t1500000.pt"]


def test_disabled_never_fires():
    assert _periodic_boundary(500_000, 0, -1) is None
    assert _periodic_boundary(500_000, -1, -1) is None
