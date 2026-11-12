#!/usr/bin/env python3
"""Aggregate the TCPA constraint-attribution matrix into the attribution table.

The experiment is a 2x2 factorial:

    constraint   x   teammate cooperation
    ------------      --------------------
    none             matched
    anchor           broken
    placebo
    prior

* constraint level
    none      no retention term at all (TD only)
    anchor    generic trust region towards the frozen offline value function
              (a constraint that never selects an action)
    placebo   the same gated IGM hinge as ``prior``, but the retained action is a
              random legal action (matched negative control)
    prior     the real TCPA arm: the gated IGM hinge keeps the offline greedy
              joint action locally greedy
* teammate cooperation
    matched   every agent plays the online policy
    broken    the teammates stay on the frozen offline prior while the focal
              agent is trained online, so the offline joint action no longer
              describes the team that is being trained

Pairing口径
-----------
All arms share the same fixed-seed evaluation environment (``eval_env_seed``),
so at the same *evaluation index* they play the same episodes.  They do **not**
share the same ``t_env``: each arm accumulates env steps by its own episode
lengths, so the n-th evaluation of two arms can sit ~100 steps apart (observed:
10107 / 10065 / 10126 for the first post-training point).  Pairing on the exact
``t_env`` would therefore leave only the two t=0 rows in common.

Every contrast in this script is therefore a **paired difference by evaluation
index**, and the bootstrap resamples the evaluation points.

Usage:
    python scripts/analyze_tcpa_attrib.py
    python scripts/analyze_tcpa_attrib.py --root checkpoints/tcpa_attrib_5m_vs_6m \
        --json checkpoints/tcpa_attrib_5m_vs_6m/summary.json \
        --markdown documentation/experiment_reports/tcpa_attrib_5m_vs_6m.md
"""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ARMS: Tuple[Tuple[str, str, str], ...] = (
    # arm,             constraint,  cooperation
    ("none_matched", "none", "matched"),
    ("none_broken", "none", "broken"),
    ("anchor_matched", "anchor", "matched"),
    ("anchor_broken", "anchor", "broken"),
    ("placebo_matched", "placebo", "matched"),
    ("placebo_broken", "placebo", "broken"),
    ("prior_matched", "prior", "matched"),
    ("prior_broken", "prior", "broken"),
    # Exploratory arms (not part of the pre-registered rule): unconditional
    # old-action retention, which the finished TCPA matrix shows is the arm
    # with a real early effect.
    ("fixed_gate_matched", "fixed_gate", "matched"),
    ("fixed_gate_broken", "fixed_gate", "broken"),
)

CONSTRAINTS = ("none", "anchor", "placebo", "prior", "fixed_gate")
COOPERATIONS = ("matched", "broken")

# Windows of the online phase.  "early" is the window the first hypothesis is
# about; "mid" is the recovery; "late" is the end of the budget.
WINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("early", 0, 300_000),
    ("mid", 300_000, 600_000),
    ("late", 600_000, 10**9),
    ("all", 0, 10**9),
)

DIAG_KEYS = (
    "tcpa/regret_mean_interval_mean",
    "tcpa/gate_mean_interval_mean",
    "tcpa/frac_gate_gt_half_interval_mean",
    "tcpa/offline_action_agreement_interval_mean",
    "tcpa/hinge_mean_interval_mean",
    "tcpa/dq_absmean_interval_mean",
    "tcpa/anchor_dev_mean_interval_mean",
    "tcpa/prior_loss_interval_mean",
)

EVAL_KEY = "evaluation/mean_episode_return"

# How far apart the paired curves may be in training time before the pairing
# is treated as broken.  The arms drift apart by their own episode lengths:
# measured 61..282 steps over the first 150k of the attribution matrix.  An
# arm that crashes and restarts starts again at 0 while the others keep
# going, which is tens of thousands of steps -- so 20k is a wide margin over
# the normal drift and still catches a restart.
MISALIGN_LIMIT_STEPS = 20_000

Point = Tuple[float, Dict]


def read_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def eval_curve(rows: Sequence[Dict]) -> List[Point]:
    """Ordered evaluation points ``(t_env, row)``.

    The logger appends, so an arm that was restarted after a crash would leave
    its earlier rows in front of the new ones.  A restart shows up as a
    decrease of ``t_env``; in that case only the last run is kept, so the
    evaluation indices stay aligned with the other arms.
    """
    points: List[Point] = []
    for row in rows:
        if EVAL_KEY not in row:
            continue
        t = float(row.get("t_env", 0.0))
        if points and t < points[-1][0] - 1e-9:
            points = []
        points.append((t, row))
    return points


def _window_index(
    curves: Sequence[List[Point]], lo: int, hi: int
) -> List[int]:
    """Indices whose (paired) t_env falls into ``[lo, hi)``.

    The paired time is the *mean* ``t_env`` of the curves at that evaluation
    index.  It must not be halved: an earlier version wrote
    ``0.5 * sum(...) / len(curves)``, which is half the mean and therefore
    doubled every window boundary -- the "early" window (0-300k) then also
    swallowed everything up to 600k, so the pre-registered readout would have
    been computed on the wrong window.
    """
    n = min(len(c) for c in curves) if curves else 0
    keep: List[int] = []
    for i in range(n):
        t = sum(float(c[i][0]) for c in curves) / len(curves)
        if lo <= t < hi:
            keep.append(i)
    return keep


def _bootstrap(diffs: np.ndarray, seed: int = 20261212) -> Dict[str, Optional[float]]:
    mean = float(diffs.mean())
    if diffs.size < 2:
        return {"n": int(diffs.size), "mean": mean, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(4000, diffs.size))
    boots = diffs[idx].mean(axis=1)
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(diffs.size),
        "mean": mean,
        "lo": float(lo_ci),
        "hi": float(hi_ci),
        "p_positive": float((boots > 0).mean()),
    }


def _bootstrap_block(
    diffs: np.ndarray, seed: int = 20261212, block_len: Optional[int] = None
) -> Dict[str, Optional[float]]:
    """Moving-block bootstrap for autocorrelated paired differences.

    The evaluation points of a training trajectory are autocorrelated: measured
    on the finished TCPA matrix (six paired contrasts x three seeds, early
    window), the lag-1 autocorrelation of the paired differences is
    +0.16..+0.47 (about +0.29 on average), which puts the effective sample
    size of a 31-point window at 16-23 rather than 31.  Resampling single
    points therefore treats correlated observations as independent and yields
    an interval that is too narrow by roughly ``sqrt(n / n_eff)``: the
    measured block/point width ratio is 1.11-1.28 (1.18 on average).

    The moving-block bootstrap resamples contiguous blocks instead, with a
    block length of ``n^(1/3)`` (Carlstein's rule), which is 3 points for the
    31-point early window -- enough to carry the measured dependence.
    """
    arr = np.asarray(diffs, dtype=np.float64)
    n = arr.size
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "block_len": None}
    mean = float(arr.mean())
    if n < 2:
        return {
            "n": int(n), "mean": mean, "lo": None, "hi": None, "block_len": None,
        }
    if block_len is None:
        block_len = max(2, int(round(n ** (1.0 / 3.0))))
    block_len = int(min(block_len, n))
    if n - block_len + 1 < 2:
        # Only one distinct block exists, so the resampling has no variability
        # at all and the interval would collapse onto the mean.  That would
        # read as a very confident interval, which is exactly the failure mode
        # this bootstrap is meant to avoid, so no interval is reported.
        return {
            "n": int(n), "mean": mean, "lo": None, "hi": None,
            "block_len": block_len,
        }
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block_len + 1, size=(4000, n_blocks))
    offsets = np.arange(block_len)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(4000, -1)[:, :n]
    boots = arr[idx].mean(axis=1)
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(n),
        "mean": mean,
        "lo": float(lo_ci),
        "hi": float(hi_ci),
        "block_len": block_len,
        "p_positive": float((boots > 0).mean()),
    }


def _paired_diffs(
    curve_a: List[Point],
    curve_b: List[Point],
    lo: int,
    hi: int,
    key: str = EVAL_KEY,
) -> List[float]:
    """The per-evaluation-point paired differences ``a - b`` inside the window."""
    keep = _window_index([curve_a, curve_b], lo, hi)
    diffs: List[float] = []
    for i in keep:
        va = curve_a[i][1].get(key)
        vb = curve_b[i][1].get(key)
        if va is None or vb is None:
            continue
        diffs.append(float(va) - float(vb))
    return diffs


def paired_contrast(
    curve_a: List[Point],
    curve_b: List[Point],
    lo: int,
    hi: int,
    key: str = EVAL_KEY,
    with_block: bool = False,
) -> Dict[str, Optional[float]]:
    """Paired mean difference ``a - b`` over the shared evaluation indices.

    ``with_block`` adds the moving-block bootstrap of the *same* difference
    under the ``"block"`` key.  The point bootstrap stays the registered one;
    the block interval is the autocorrelation-aware robustness reading.
    """
    diffs = _paired_diffs(curve_a, curve_b, lo, hi, key)
    if not diffs:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    arr = np.asarray(diffs, dtype=np.float64)
    stat = _bootstrap(arr)
    if with_block:
        stat["block"] = _bootstrap_block(arr)
    return stat


def interaction(
    curve_treat_m: List[Point],
    curve_base_m: List[Point],
    curve_treat_b: List[Point],
    curve_base_b: List[Point],
    lo: int,
    hi: int,
    key: str = EVAL_KEY,
    with_block: bool = False,
) -> Dict[str, Optional[float]]:
    """Interaction term: (treat-base)_matched - (treat-base)_broken.

    All four curves are evaluated on the same fixed-seed evaluation environment,
    so the four differences can be formed point by point at the same evaluation
    index.  The interaction is a paired difference-of-differences over those
    points, and the CI is a bootstrap over them.

    ``with_block`` adds the moving-block bootstrap under the ``"block"`` key.
    """
    diffs = interaction_points(
        curve_treat_m, curve_base_m, curve_treat_b, curve_base_b, lo, hi, key
    )
    if not diffs:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    arr = np.asarray(diffs, dtype=np.float64)
    stat = _bootstrap(arr)
    if with_block:
        stat["block"] = _bootstrap_block(arr)
    return stat


def difference_points(
    curve_a: List[Point], curve_b: List[Point], lo: int, hi: int, key: str = EVAL_KEY
) -> List[float]:
    """The per-evaluation-point paired differences ``a - b``."""
    keep = _window_index([curve_a, curve_b], lo, hi)
    diffs = []
    for i in keep:
        va = curve_a[i][1].get(key)
        vb = curve_b[i][1].get(key)
        if va is None or vb is None:
            continue
        diffs.append(float(va) - float(vb))
    return diffs


def interaction_points(
    curve_treat_m: List[Point],
    curve_base_m: List[Point],
    curve_treat_b: List[Point],
    curve_base_b: List[Point],
    lo: int,
    hi: int,
    key: str = EVAL_KEY,
) -> List[float]:
    """The per-evaluation-point difference-of-differences behind P1."""
    curves = [curve_treat_m, curve_base_m, curve_treat_b, curve_base_b]
    keep = _window_index(curves, lo, hi)
    diffs = []
    for i in keep:
        vals = [c[i][1].get(key) for c in curves]
        if any(v is None for v in vals):
            continue
        diffs.append(
            (float(vals[0]) - float(vals[1])) - (float(vals[2]) - float(vals[3]))
        )
    return diffs


def stability(diffs: Sequence[float]) -> Dict[str, Optional[float]]:
    """How fragile a paired difference is: sign consistency and split halves.

    The bootstrap interval over the evaluation points describes the point-to-
    point fluctuation of one trajectory.  It says nothing about how much of the
    effect is carried by a handful of points, so these two extra readings are
    reported next to it: the fraction of points that share the sign, and
    whether the first and second half of the window agree.
    """
    if not diffs:
        return {
            "n": 0,
            "mean": None,
            "frac_positive": None,
            "first_half_mean": None,
            "second_half_mean": None,
            "sign_consistent": None,
            "halves_agree": None,
        }
    arr = np.asarray(diffs, dtype=np.float64)
    mid = arr.size // 2
    first, second = arr[:mid], arr[mid:]
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "frac_positive": float((arr > 0).mean()),
        "first_half_mean": float(first.mean()) if first.size else None,
        "second_half_mean": float(second.mean()) if second.size else None,
        "sign_consistent": bool(np.all(arr >= 0) or np.all(arr <= 0)),
        "halves_agree": bool(
            first.size
            and second.size
            and np.sign(first.mean()) == np.sign(second.mean())
        ),
    }


def mean_window(
    curve: List[Point], lo: int, hi: int, key: str = EVAL_KEY
) -> Optional[float]:
    values = []
    for t, row in curve:
        if not (lo <= t < hi):
            continue
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def fmt_ci(stat: Dict[str, Optional[float]], digits: int = 2) -> str:
    if stat.get("mean") is None:
        return "-"
    if stat.get("lo") is None:
        return f"{stat['mean']:+.{digits}f} n={stat.get('n', 0)}"
    return (
        f"{stat['mean']:+.{digits}f} "
        f"[{stat['lo']:+.{digits}f}, {stat['hi']:+.{digits}f}] n={stat.get('n', 0)}"
    )


def classify(stat: Dict[str, Optional[float]]) -> str:
    """``显著为正`` / ``含 0`` / ``显著为负`` / ``无数据`` for one statistic."""
    if stat.get("mean") is None or stat.get("lo") is None:
        return "无数据"
    if float(stat["lo"]) > 0.0:
        return "显著为正"
    if float(stat["hi"]) < 0.0:
        return "显著为负"
    return "含 0"


def verdict_from_classes(p1: str, p2: str) -> Tuple[str, str]:
    """Apply the pre-registered decision rule to the two primary readouts.

    ``p1`` is the interaction of the prior retention (does the value of
    keeping the old action depend on the teammate-cooperation condition?) and
    ``p2`` is prior minus anchor (does the *content* of the constraint matter,
    rather than having a constraint at all?).

    The rule was written down on 2026-11-12 02:25, before any of the decisive
    arms had finished; see the design document, section 4.
    """
    if "无数据" in (p1, p2):
        return (
            "**证据不足**：主窗口还没有足够数据。",
            "",
        )
    if p1 == "显著为正" and p2 == "显著为正":
        return (
            "**支持 H2**（旧动作的收益来自协作内容，且依赖队友配合）。",
            "P1 显著为正说明保留旧动作的收益随队友配合条件变化；"
            "P2 显著为正说明这种收益不能由一个通用的价值约束替代。",
        )
    if p1 == "含 0" and p2 == "含 0":
        return (
            "**支持 H1**（收益由与内容无关、且与队友配合条件无关的通用早期约束解释）。",
            "P1 与 P2 的 CI 都含 0，两个量都没给出 H2 需要的证据。",
        )
    failed = []
    if p1 != "显著为正":
        failed.append(f"P1={p1}")
    if p2 != "显著为正":
        failed.append(f"P2={p2}")
    return (
        "**证据不足**（混合结果，不下结论）。",
        "没通过的项：" + "、".join(failed) + "。",
    )


def lag1_autocorr(diffs: Sequence[float]) -> Optional[float]:
    """Lag-1 autocorrelation of the paired differences, if it is defined.

    The evaluation points of one training trajectory are not independent: the
    policy at evaluation point *k+1* is barely different from the policy at
    point *k*.  The autocorrelation is reported so the reader can see how much
    the effective sample size is below the number of points.
    """
    if len(diffs) < 3:
        return None
    arr = np.asarray(diffs, dtype=np.float64)
    centered = arr - arr.mean()
    denom = float((centered * centered).sum())
    if denom <= 0.0:
        return None
    return float((centered[:-1] * centered[1:]).sum() / denom)


def effective_sample_size(diffs: Sequence[float]) -> Optional[float]:
    """``n / (1 + 2 * rho1)`` -- the usual first-order correction."""
    rho = lag1_autocorr(diffs)
    if rho is None:
        return None
    n = len(diffs)
    factor = 1.0 + 2.0 * rho
    if factor <= 0.0:
        return float(n)
    return float(n) / factor


def block_veto(
    point_classes: Dict[str, str], block_classes: Dict[str, str]
) -> Tuple[bool, List[str]]:
    """The withdraw-only amendment for autocorrelated evaluation points.

    The pre-registered rule was written with a point bootstrap over the
    evaluation points.  On the finished TCPA matrix the paired differences
    have a lag-1 autocorrelation of +0.16..+0.47, so the effective sample
    size of a 31-point window is 11-22 instead of 31 and the point interval
    is roughly 35% too narrow.  The moving-block bootstrap is the
    autocorrelation-aware version of the same reading.

    The amendment is a **veto, not a second rule**: when the two intervals
    disagree on a primary readout, the registered branch can no longer be
    maintained and the reading falls back to "evidence insufficient".  It
    never re-runs the decision rule to look for a branch that the
    disagreement would open up -- that could *grant* a conclusion, and this
    amendment may only withdraw one.
    """
    reasons: List[str] = []
    for name, point_class in point_classes.items():
        block_class = block_classes.get(name, "无数据")
        if point_class == block_class:
            continue
        reasons.append(
            f"{name}：登记用点自助判「{point_class}」，块自助判「{block_class}」"
        )
    return bool(reasons), reasons


def eval_cadence(curve: List[Point]) -> Dict[str, Optional[float]]:
    """How many env steps the arm advances between two evaluations.

    Every arm evaluates on the same ~10k-step grid, and that is what makes
    "paired by evaluation index" equivalent to "paired by training step":
    the n-th evaluation of an arm is at roughly ``10k * n`` steps of its own
    run, whichever wall-clock time that arm happened to start at.  The
    cadence is reported so the pairing can be checked rather than assumed.
    """
    if len(curve) < 3:
        return {"n": len(curve), "mean_step": None, "min_step": None, "max_step": None}
    times = [float(t) for t, _row in curve]
    increments = [times[i] - times[i - 1] for i in range(2, len(times))]
    if not increments:
        return {"n": len(curve), "mean_step": None, "min_step": None, "max_step": None}
    return {
        "n": len(curve),
        "mean_step": float(np.mean(increments)),
        "min_step": float(np.min(increments)),
        "max_step": float(np.max(increments)),
    }


def pairing_spread(
    curves: Sequence[List[Point]], lo: int, hi: int
) -> Dict[str, Optional[float]]:
    """How far apart the paired curves are in training time.

    The arms accumulate env steps at their own pace, so at the same evaluation
    index their ``t_env`` differ by a little (measured: 61..282 steps over
    the first 150k of the attribution matrix, i.e. under 3% of one 10k-step
    evaluation block).  A *large* spread would mean the curves no longer
    advance on the same grid -- for instance if one arm had been restarted
    with a different evaluation cadence, or if two runs had been interleaved
    in one log.  Then the index pairing would no longer mean "the same
    training step", and the reading would have to be withdrawn.
    """
    keep = _window_index(curves, lo, hi)
    spreads: List[float] = []
    for i in keep:
        times = [float(c[i][0]) for c in curves]
        spreads.append(max(times) - min(times))
    if not spreads:
        return {"n": 0, "max_spread": None, "mean_spread": None, "at_index": None}
    arr = np.asarray(spreads, dtype=np.float64)
    worst = int(np.argmax(arr))
    return {
        "n": len(spreads),
        "max_spread": float(arr.max()),
        "mean_spread": float(arr.mean()),
        "at_index": int(keep[worst]),
    }


def minimum_detectable_effect(
    stat: Dict[str, Optional[float]], z_sum: float = 2.80
) -> Optional[float]:
    """The smallest effect the registered rule could detect with ~80% power.

    A 95% CI half-width is ``1.96 * SE``, so the standard error is
    ``(hi - lo) / 3.92``; detecting a two-sided effect ``d`` with 80% power
    needs ``d >= (1.96 + 0.84) * SE``.  This turns "the CI contains zero"
    into a statement about what the experiment could and could not have
    seen, instead of a statement about the truth.
    """
    lo = stat.get("lo")
    hi = stat.get("hi")
    if lo is None or hi is None:
        return None
    se = (float(hi) - float(lo)) / 3.92
    return z_sum * se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/tcpa_attrib_5m_vs_6m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default="")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    curves: Dict[str, List[Point]] = {}
    for arm, _constraint, _coop in ARMS:
        path = os.path.join(args.root, arm, f"seed{args.seed}", "online.jsonl")
        curves[arm] = eval_curve(read_rows(path))

    missing = [arm for arm in curves if not curves[arm]]
    lines: List[str] = []
    summary: Dict = {"arms": {}, "contrasts": {}}

    lines.append("# TCPA 约束归因实验：早期约束 vs 队友配合条件下的旧动作保留")
    lines.append("")
    lines.append(
        "**配对口径：按评估序号配对，不按精确 `t_env`。** 各臂共用同一个固定种子的"
        "评估环境（`eval_env_seed`），所以同一个评估序号上跑的是同一批对局；"
        "但各臂按自己的回合长度累计 env 步，同一序号的 `t_env` 会差几十到上百步"
        "（实测首个训练点为 10107 / 10065 / 10126），按 `t_env` 取交集只剩两行 "
        "`t=0`。因此所有对比都是**逐评估序号的配对差**，bootstrap 在评估点之间重采样。"
    )
    lines.append("")

    # ------------------------------------------------------------------ table
    lines.append("## 1. 各臂表现（窗口内评估点的平均回报）")
    lines.append("")
    lines.append(
        f"| {'臂':<16}{'n_eval':>7}{'last_t':>10}"
        + "".join(f"{w:>12}" for w, _lo, _hi in WINDOWS)
        + f"{'win_rate':>10}"
    )
    lines.append("|" + "|".join("-" * 9 for _ in range(3 + len(WINDOWS) + 1)) + "|")
    for arm, constraint, coop in ARMS:
        curve = curves[arm]
        if not curve:
            lines.append(
                f"| {arm:<16}{'-':>7}{'-':>10}"
                + "".join(f"{'-':>12}" for _ in WINDOWS)
                + f"{'-':>10}|"
            )
            continue
        last_t = int(curve[-1][0])
        cells = ""
        for _w, lo, hi in WINDOWS:
            cells += f"{fmt(mean_window(curve, lo, hi)):>12}"
        win = mean_window(curve, 0, 10**9, "evaluation/win_rate")
        lines.append(f"| {arm:<16}{len(curve):>7}{last_t:>10}{cells}{fmt(win):>10}")
        summary["arms"][arm] = {
            "constraint": constraint,
            "cooperation": coop,
            "n_eval": len(curve),
            "last_t_env": last_t,
            "windows": {w: mean_window(curve, lo, hi) for w, lo, hi in WINDOWS},
            "win_rate": win,
            "mechanism_early": {
                key: mean_window(curve, 0, 300_000, key) for key in DIAG_KEYS
            },
        }
    lines.append("")

    # --------------------------------------------------------------- contrasts
    lines.append("## 2. 析因对比（配对差，bootstrap 95% CI）")
    lines.append("")
    lines.append(
        "每个约束档位相对 `none` 的差，以及**交互项**"
        "（配合条件的差 = 该约束在 matched 下的效应 − 在 broken 下的效应，"
        "逐点配对的差之差）。"
    )
    lines.append("")
    lines.append(
        f"| {'对比':<34}" + "".join(f"{w:>26}" for w, _lo, _hi in WINDOWS) + " |"
    )
    lines.append("|" + "|".join("-" * 26 for _ in range(1 + len(WINDOWS))) + "|")

    contrast_rows = []
    for constraint in CONSTRAINTS:
        if constraint == "none":
            continue
        for coop in COOPERATIONS:
            arm = f"{constraint}_{coop}"
            base = f"none_{coop}"
            cells = ""
            stats = {}
            for w, lo, hi in WINDOWS:
                stat = paired_contrast(curves[arm], curves[base], lo, hi)
                stats[w] = stat
                cells += f"{fmt_ci(stat):>26}"
            lines.append(f"| {constraint} − none ({coop})" + cells + " |")
            contrast_rows.append((f"{constraint} − none ({coop})", stats))
        # interaction term
        cells = ""
        stats = {}
        for w, lo, hi in WINDOWS:
            stat = interaction(
                curves[f"{constraint}_matched"],
                curves[f"none_matched"],
                curves[f"{constraint}_broken"],
                curves[f"none_broken"],
                lo,
                hi,
            )
            stats[w] = stat
            cells += f"{fmt_ci(stat):>26}"
        lines.append(f"| **交互** {constraint} (matched−broken)" + cells + " |")
        contrast_rows.append((f"interaction {constraint} (matched-broken)", stats))

    # prior vs anchor: the content-specific contrast
    for coop in COOPERATIONS:
        cells = ""
        stats = {}
        for w, lo, hi in WINDOWS:
            stats[w] = paired_contrast(
                curves[f"prior_{coop}"], curves[f"anchor_{coop}"], lo, hi
            )
            cells += f"{fmt_ci(stats[w]):>26}"
        lines.append(f"| prior − anchor ({coop})" + cells + " |")
        contrast_rows.append((f"prior − anchor ({coop})", stats))

    for name, stats in contrast_rows:
        summary["contrasts"][name] = stats
    lines.append("")

    # ------------------------------------------------------------- trajectories
    lines.append("## 3. 逐评估点的回报曲线（配对序号）")
    lines.append("")
    lines.append(
        "窗口平均会把「早期下滑 → 恢复」的形状抹平，所以这里按评估序号列出各臂的回报。"
        "各臂在同一个序号上跑的是同一批对局，但 `t_env` 略有差异（表中列的是 "
        "`none_matched` 的 `t_env`）。"
    )
    lines.append("")
    lines.append(
        f"| {'idx':>4}{'t_env':>9}"
        + "".join(f"{arm.replace('_', ' ')[:12]:>14}" for arm, _c, _k in ARMS)
        + " |"
    )
    lines.append("|" + "|".join("-" * 9 for _ in range(2 + len(ARMS))) + "|")
    n_rows = max((len(curves[arm]) for arm, _c, _k in ARMS), default=0)
    for idx in range(n_rows):
        cells = ""
        for arm, _c, _k in ARMS:
            curve = curves[arm]
            if idx < len(curve):
                cells += f"{fmt(curve[idx][1].get(EVAL_KEY)):>14}"
            else:
                cells += f"{'-':>14}"
        ref = next(
            (
                curves[arm][idx][0]
                for arm, _c, _k in ARMS
                if idx < len(curves[arm])
            ),
            None,
        )
        lines.append(
            f"{idx:>4}{('-' if ref is None else int(ref)):>9}{cells} |"
        )
    lines.append("")

    # ------------------------------------------------------------ diagnostics
    lines.append("## 4. 机制诊断（窗口内平均）")
    lines.append("")
    lines.append(
        f"| {'臂':<16}{'window':>8}"
        + "".join(
            f"{k.replace('tcpa/', '').replace('_interval_mean', ''):>26}"
            for k in DIAG_KEYS
        )
        + " |"
    )
    lines.append("|" + "|".join("-" * 9 for _ in range(2 + len(DIAG_KEYS))) + "|")
    for arm, _constraint, _coop in ARMS:
        curve = curves[arm]
        if not curve:
            continue
        for w, lo, hi in WINDOWS:
            if w == "all":
                continue
            cells = ""
            for key in DIAG_KEYS:
                value = mean_window(curve, lo, hi, key)
                cells += f"{fmt(value, 4):>26}"
            lines.append(f"| {arm:<16}{w:>8}{cells} |")
    lines.append("")

    # Paired broken-minus-matched mechanism deltas: the informative view of
    # whether the teammate-cooperation condition changed the *mechanism* and
    # not just the outcome.
    lines.append("### 4.1 配合条件的机制差（broken − matched，同一评估序号配对）")
    lines.append("")
    mech_keys = (
        "tcpa/regret_mean_interval_mean",
        "tcpa/gate_mean_interval_mean",
        "tcpa/offline_action_agreement_interval_mean",
        "tcpa/dq_absmean_interval_mean",
        "evaluation/mean_episode_return",
    )
    lines.append(
        f"| {'约束档位':<10}"
        + "".join(f"{k.replace('tcpa/', '').replace('_interval_mean', '')[:20]:>24}" for k in mech_keys)
        + " |"
    )
    lines.append("|" + "|".join("-" * 10 for _ in range(1 + len(mech_keys))) + "|")
    for constraint in ("none", "anchor", "placebo", "prior", "fixed_gate"):
        cells = ""
        any_data = False
        for key in mech_keys:
            stat = paired_contrast(
                curves[f"{constraint}_broken"], curves[f"{constraint}_matched"], 0, 10**9, key
            )
            if stat.get("mean") is None:
                cells += f"{'-':>24}"
            else:
                any_data = True
                cells += f"{stat['mean']:>+24.4f}"
        if any_data:
            lines.append(f"| {constraint:<10}{cells} |")
    lines.append("")
    lines.append(
        "读法：`none` 行的机制差说明**配合条件本身把机制改成了什么样**——"
        "若 `regret` 在 broken 下反而更小、`gate` 更松，"
        "说明 gate 的信号是「离线联合动作在价值上还 competitive 吗」，"
        "而不是「旧联合动作还适不适合当前这支在跑的队伍」，"
        "两者在本实验的 `broken` 条件下恰好分开（见设计文档第 4 节的运行后补记）。"
    )
    lines.append("")

    # --------------------------------------------------- constraint activity
    # A constraint arm whose loss stays at exactly zero never engaged.  A
    # read-out built on such an arm says nothing about the constraint at all,
    # so this has to be checked rather than assumed.
    lines.append("### 4.2 约束活性检查")
    lines.append("")
    lines.append(
        "如果某个**带约束**的臂，它的 hinge 损失（`tcpa/prior_loss`）在整个 early 窗口"
        "恒为 0，说明这个约束**一次都没有生效**——那么基于它的读数（Δ_prior、"
        "P1、P2）说的就不是「约束有没有用」，而是「约束根本没在跑」。"
        "下表逐臂给出 early 窗口的有效 hinge 损失与活性判定。"
    )
    lines.append("")
    lines.append(
        f"| {'臂':<16}{'variant':<12}{'early prior_loss':>20}{'hinge_mean':>14}{'活性':>14} |"
    )
    lines.append("|" + "|".join("-" * 12 for _ in range(5)) + "|")
    hinge_arms = {
        "prior_matched",
        "prior_broken",
        "placebo_matched",
        "placebo_broken",
        "fixed_gate_matched",
        "fixed_gate_broken",
    }
    activity: Dict[str, Dict[str, Optional[float]]] = {}
    for arm, constraint, _coop in ARMS:
        curve = curves[arm]
        if not curve:
            continue
        loss = mean_window(curve, 0, 300_000, "tcpa/prior_loss_interval_mean")
        hinge = mean_window(curve, 0, 300_000, "tcpa/hinge_mean_interval_mean")
        activity[arm] = {"prior_loss": loss, "hinge_mean": hinge}
        if arm in hinge_arms:
            state = "无数据" if loss is None else ("生效" if loss > 0.0 else "**未生效**")
        elif constraint == "none":
            state = "无约束项(设计如此)"
        else:
            state = "非 hinge 约束"
        loss_text = "-" if loss is None else f"{loss:.4f}"
        hinge_text = "-" if hinge is None else f"{hinge:.4f}"
        lines.append(
            f"| {arm:<16}{constraint:<12}{loss_text:>20}{hinge_text:>14}{state:>14} |"
        )
    lines.append("")
    inert_decisive = [
        arm
        for arm in ("prior_matched", "prior_broken")
        if activity.get(arm, {}).get("prior_loss") == 0.0
    ]
    activity_ok = not inert_decisive
    activity_qualifier = ""
    if not activity_ok:
        activity_qualifier = (
            "**约束未生效：登记结论已撤回。** "
            f"决定性臂 {'、'.join(inert_decisive)} 在 early 窗口的 "
            "`tcpa/prior_loss` 恒为 0，说明 hinge 一次都没有触发。"
            "在这种情况下 P1/P2 测的不是「保留旧动作有没有用」，"
            "而是「一个没在跑的约束有没有用」——读数不可解释。"
            "按预登记原则，此处只能记为**证据不足**。"
        )
        lines.append(activity_qualifier)
        lines.append("")

    # ------------------------------------------------------- pre-registered rule
    lines.append("## 5. 预先登记的判读")
    lines.append("")
    lines.append(
        "判读规则写于 2026-11-12 02:25（决定性的 4 个臂一个都还没跑完），"
        "见设计文档 `tcpa_attrib_5m_vs_6m_design.md` 第 4 节。主窗口 = early（0–300k），"
        "CI 取 95%，「显著」= CI 不含 0。"
    )
    lines.append("")
    lines.append(
        f"| {'读数':<30}{'点自助 值 [95% CI]':>30}{'n':>6}{'判定':>12}"
        f"{'块自助 值 [95% CI]':>30}{'块判定':>12} |"
    )
    lines.append("|" + "|".join("-" * 10 for _ in range(6)) + "|")

    primary = "early"
    primary_lo, primary_hi = next(
        (lo, hi) for w, lo, hi in WINDOWS if w == primary
    )
    readings = {
        "P1 交互 prior (matched−broken)": summary["contrasts"].get(
            f"interaction prior (matched-broken)", {}
        ).get(primary, {}),
        "P2 prior − anchor (matched)": summary["contrasts"].get(
            "prior − anchor (matched)", {}
        ).get(primary, {}),
    }
    # The autocorrelation-aware robustness reading of the same two numbers:
    # a moving-block bootstrap instead of a point bootstrap (2026-11-12 03:40
    # amendment).  It never replaces the registered reading; it can only veto
    # it, see ``block_veto``.
    block_stats = {
        "P1 交互 prior (matched−broken)": interaction(
            curves["prior_matched"],
            curves["none_matched"],
            curves["prior_broken"],
            curves["none_broken"],
            primary_lo,
            primary_hi,
            with_block=True,
        ).get("block", {}),
        "P2 prior − anchor (matched)": paired_contrast(
            curves["prior_matched"],
            curves["anchor_matched"],
            primary_lo,
            primary_hi,
            with_block=True,
        ).get("block", {}),
    }

    classes = {}
    block_classes = {}
    for name, stat in readings.items():
        verdict_txt = classify(stat)
        classes[name] = verdict_txt
        block_stat = block_stats.get(name, {})
        block_classes[name] = classify(block_stat)
        lines.append(
            f"| {name:<30}{fmt_ci(stat):>30}{stat.get('n', 0):>6}{verdict_txt:>12}"
            f"{fmt_ci(block_stat):>30}{block_classes[name]:>12} |"
        )
    lines.append("")

    p1 = classes.get("P1 交互 prior (matched−broken)")
    p2 = classes.get("P2 prior − anchor (matched)")
    conclusion, detail = verdict_from_classes(p1, p2)
    lines.append(conclusion)
    if detail:
        lines.append("")
        lines.append(detail)
    lines.append("")

    # The withdraw-only amendment: when the point bootstrap and the block
    # bootstrap disagree on a primary readout, the registered branch cannot
    # be maintained.  The amendment withdraws the conclusion, it never grants
    # one -- it does not re-run the rule to look for the other branch.
    vetoed, veto_reasons = block_veto(classes, block_classes)
    block_qualifier = ""
    if vetoed:
        block_qualifier = (
            "**块自助修正：登记结论已撤回。** "
            + "；".join(veto_reasons)
            + "。登记用的点自助把有自相关的评估点当成独立样本，区间偏窄约 11%–28%"
            "（既有矩阵实测）；按 2026-11-12 03:40 的补记，两个区间不一致时只能"
            "**撤回**登记结论、改记为证据不足，不能反过来改判到另一支（H1 或 H2）。"
        )
        lines.append(block_qualifier)
        lines.append("")

    # The pairing itself has to hold: the arms drift apart by their own
    # episode lengths (a few hundred steps), but an arm that crashed and
    # restarted starts again at 0 while the others keep going, and the window
    # assignment would then be driven by a mean that describes none of the
    # curves.  A broken pairing withdraws the conclusion; it cannot repair it.
    align_stats = {
        "P1": pairing_spread(
            [
                curves["prior_matched"],
                curves["none_matched"],
                curves["prior_broken"],
                curves["none_broken"],
            ],
            primary_lo,
            primary_hi,
        ),
        "P2": pairing_spread(
            [curves["prior_matched"], curves["anchor_matched"]],
            primary_lo,
            primary_hi,
        ),
    }
    misaligned = [
        key
        for key, stat in align_stats.items()
        if stat["max_spread"] is not None
        and float(stat["max_spread"]) > MISALIGN_LIMIT_STEPS
    ]
    align_ok = not misaligned
    align_qualifier = ""
    if not align_ok:
        details = "；".join(
            f"{key} 的同一评估序号上各臂 t_env 最大相差 "
            f"{int(float(align_stats[key]['max_spread']))} 步"
            for key in misaligned
        )
        align_qualifier = (
            "**配对失效：登记结论已撤回。** "
            + details
            + f"，超过 {MISALIGN_LIMIT_STEPS} 步的告警线"
            "（正常情况下各臂同索引的 `t_env` 只差几十到几百步，"
            "因为它们在同一个 ~10k 步的评估网格上）。"
            "这说明各臂的评估网格不再一致，同一评估序号不再对应同一个训练阶段，"
            "窗口归属与配对差都不再可信。按预登记原则，此处只能记为**证据不足**，"
            "不能凭错位的数据下任何结论。"
        )
        lines.append(align_qualifier)
        lines.append("")

    # --------------------------------------------------- manipulation check
    # The registered rule reads "P1 contains 0 and P2 contains 0" as H1.  That
    # reading needs the broken-teammate manipulation to have bitten: if
    # freezing the teammates changed nothing at all, a P1 of zero says the
    # experiment never produced the contrast, not that H1 is true.  The
    # amendment below can only *withdraw* the H1 reading, never grant a
    # conclusion, so it never makes a conclusion easier to reach.
    manip_stats = {}
    for constraint in ("none", "anchor", "placebo", "prior", "fixed_gate"):
        stat = paired_contrast(
            curves[f"{constraint}_broken"], curves[f"{constraint}_matched"], 0, 300_000
        )
        manip_stats[constraint] = stat
    manip_classes = {
        key: classify(stat) for key, stat in manip_stats.items()
    }
    manip_with_data = [
        key for key, value in manip_classes.items() if value != "无数据"
    ]
    manip_effective = any(
        manip_classes[key] in ("显著为正", "显著为负") for key in manip_with_data
    )
    qualifier = ""
    if p1 == "含 0" and p2 == "含 0" and not manip_effective:
        qualifier = (
            "**操纵检验未通过——本结论不能读作「支持 H1」。** "
            "预登记规则里「P1 与 P2 都含 0 → 支持 H1」这一支隐含一个前提："
            "「队友冻结」这个干预确实在结果层面产生了变化。"
            "本次有数据的 "
            f"{len(manip_with_data)} 个约束档位的 `broken − matched` 差全部含 0，"
            "说明干预没有造出可测量的对比——此时 P1 含 0 只说明实验缺少对比，"
            "而不是 H1 成立。按预登记原则，此处应记为**证据不足**。"
        )
    if qualifier:
        lines.append(qualifier)
        lines.append("")

    lines.append(
        "探索性读数（不参与判读）：P1 与 `Δ_prior` 从 early 到 late 的变化（时间剖面）；"
        "机制诊断里 `tcpa/regret_mean` / `gate_mean` 在两个条件下是否分化"
        "（若 gate 在 `broken` 下并没有收紧，说明 gate 没有识别出队友不配合，"
        "属于实现层结论）。"
    )
    lines.append("")

    # ------------------------------------------------------- robustness (exploratory)
    lines.append("### 5.1 稳健性（探索性，不参与判读）")
    lines.append("")
    lines.append(
        "bootstrap 区间描述的是**一条轨迹上评估点之间的波动**，不说明效应是不是由"
        "少数几个点扛起来的。下面两项补充读数：主窗口内**符号一致率**（有多大比例的"
        "评估点与均值同号）与**前后半段是否同号**。"
    )
    lines.append("")

    def _pad(text: str, width: int) -> str:
        """Pad to a *display* width, counting CJK characters as two columns."""
        shown = sum(
            2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text
        )
        return text + " " * max(0, width - shown)

    def _stable_row(label: str, diffs: List[float]) -> str:
        stat = stability(diffs)
        if stat["n"] == 0:
            return (
                f"| {_pad(label, 34)}{'-':>7}{'-':>10}{'-':>12}{'-':>12}"
                f"{'-':>10}{'-':>10} |"
            )
        return (
            f"| {_pad(label, 34)}{stat['n']:>7}{stat['mean']:>+10.3f}"
            f"{stat['frac_positive'] * 100:>11.0f}%"
            f"{stat['first_half_mean']:>+12.3f}{stat['second_half_mean']:>+12.3f}"
            f"{('是' if stat['sign_consistent'] else '否'):>10}"
            f"{('是' if stat['halves_agree'] else '否'):>10} |"
        )

    lines.append(
        f"| {'读数':<34}{'n':>7}{'均值':>10}{'同号点占比':>12}"
        f"{'前半段均值':>12}{'后半段均值':>12}{'全同号':>10}{'两半同号':>10} |"
    )
    lines.append("|" + "|".join("-" * 11 for _ in range(8)) + "|")

    p1_points = interaction_points(
        curves["prior_matched"],
        curves["none_matched"],
        curves["prior_broken"],
        curves["none_broken"],
        0,
        300_000,
    )
    p2_points = difference_points(
        curves["prior_matched"], curves["anchor_matched"], 0, 300_000
    )
    rows = [
        ("P1 交互 prior", p1_points),
        ("  └ prior − none (matched)", difference_points(
            curves["prior_matched"], curves["none_matched"], 0, 300_000)),
        ("  └ prior − none (broken)", difference_points(
            curves["prior_broken"], curves["none_broken"], 0, 300_000)),
        ("P2 prior − anchor (matched)", p2_points),
        ("噪声地板 placebo − none (matched)", difference_points(
            curves["placebo_matched"], curves["none_matched"], 0, 300_000)),
        ("噪声地板 placebo − none (broken)", difference_points(
            curves["placebo_broken"], curves["none_broken"], 0, 300_000)),
        ("探索 fixed_gate − none (matched)", difference_points(
            curves["fixed_gate_matched"], curves["none_matched"], 0, 300_000)),
        ("探索 fixed_gate − none (broken)", difference_points(
            curves["fixed_gate_broken"], curves["none_broken"], 0, 300_000)),
    ]
    for label, diffs in rows:
        lines.append(_stable_row(label, diffs))
    lines.append("")
    lines.append(
        "**噪声地板**两行是安慰剂臂（有同形 hinge、但被保留的「旧动作」是随机合法动作）"
        "相对无约束臂的差。安慰剂与真 TCPA 在 hinge 形式、gate、权重上完全一致，"
        "所以它的效应就是「只要有这个约束形式就有用」的那部分；"
        "`prior` 的效应若明显大于安慰剂，才说明起作用的是**旧动作的内容**而不是约束形式。"
        "这一行同时也是单种子单臂差的经验噪声尺度。"
    )
    lines.append("")
    summary["robustness"] = {
        "P1": stability(p1_points),
        "P2": stability(p2_points),
    }

    # ------------------------------------------------------ manipulation check
    lines.append("### 5.2 操纵检验（broken 干预是否在结果层面生效）")
    lines.append("")
    lines.append(
        "预登记规则里「P1 与 P2 都含 0 → 支持 H1」这一支隐含前提：**队友冻结**这个干预"
        "确实在结果层面造出了变化。如果四个约束档位的 `broken − matched` 差全部含 0，"
        "说明干预没生效，此时 P1 含 0 只能读作「实验缺少对比」，不能读作 H1 成立。"
        "下表就是这项检验。"
    )
    lines.append("")
    lines.append(
        f"| {'对比（broken − matched）':<34}" + "".join(f"{w:>30}" for w, _lo, _hi in WINDOWS) + " |"
    )
    lines.append("|" + "|".join("-" * 26 for _ in range(1 + len(WINDOWS))) + "|")
    for constraint in ("none", "anchor", "placebo", "prior", "fixed_gate"):
        cells = ""
        for _w, lo, hi in WINDOWS:
            cells += f"{fmt_ci(manip_stats[constraint] if _w == 'early' else paired_contrast(curves[f'{constraint}_broken'], curves[f'{constraint}_matched'], lo, hi)):>30}"
        lines.append(f"| {constraint:<34}{cells} |")
    lines.append("")
    lines.append(
        f"**检验结果：{'通过' if manip_effective else '未通过'}**"
        + (
            f"（{len(manip_with_data)} 个有数据的约束档位里，至少一个的 "
            "`broken − matched` 差显著不为 0，"
            "干预在结果层面产生了可测量的变化，P1 的解读成立）。"
            if manip_effective
            else (
                f"（{len(manip_with_data)} 个有数据的约束档位的 `broken − matched` 差全部含 0，"
                "干预没有造出可测量的结果变化，P1 的解读不成立"
                + (
                    "；其余档位尚无数据）。"
                    if len(manip_with_data) < 4
                    else "）。"
                )
            )
        )
    )
    lines.append("")
    lines.append(
        "注意与 4.1 节的机制读数区分：**机制上** broken 条件确实改变了 gate 与 regret"
        "（见 4.1），但**结果上**它是否改变了回报是另一回事。"
        "预登记规则的 H1 分支要求的是后者。"
    )
    lines.append("")

    # ------------------------------------------------- autocorrelation robustness
    lines.append("### 5.3 自相关稳健性（2026-11-12 03:40 补记）")
    lines.append("")
    lines.append(
        "同一批评估点上的配对差不是独立样本：第 k 个评估点的策略与第 k+1 个评估点的策略"
        "只差几万个 env 步，差值序列带正自相关。登记用的点自助把相关样本当成独立样本，"
        "区间因此偏窄；块自助（moving-block bootstrap，块长 n^(1/3)）按连续块重采样，"
        "是同一读数的自相关稳健版本。"
    )
    lines.append("")
    lines.append(
        f"| {'读数':<30}{'n':>5}{'lag-1 自相关':>14}{'有效样本量 n_eff':>18}"
        f"{'点自助判定':>14}{'块自助判定':>14} |"
    )
    lines.append("|" + "|".join("-" * 12 for _ in range(6)) + "|")
    for label, points, name in (
        ("P1 交互 prior", p1_points, "P1 交互 prior (matched−broken)"),
        ("P2 prior − anchor", p2_points, "P2 prior − anchor (matched)"),
    ):
        rho = lag1_autocorr(points)
        neff = effective_sample_size(points)
        rho_text = "-" if rho is None else f"{rho:+.3f}"
        neff_text = "-" if neff is None else f"{neff:.1f}"
        lines.append(
            f"| {label:<30}{len(points):>5}{rho_text:>14}{neff_text:>18}"
            f"{classes.get(name, '无数据'):>14}{block_classes.get(name, '无数据'):>14} |"
        )
    lines.append("")
    lines.append(
        "**修正方向（只保守、不放宽）**：两个区间一致 → 登记结论不变；"
        "不一致 → 登记结论被**撤回**，改记为证据不足。"
        "补记只能让判定更难成立，不能把结论改到另一支（H1 或 H2）。"
    )
    lines.append("")

    # ------------------------------------------------ pairing alignment check
    lines.append("### 5.4 评估节奏与配对一致性检查")
    lines.append("")
    lines.append(
        "**按评估序号配对是否等价于按训练步数配对**，取决于各臂的评估节奏是否一致。"
        "下表给出每个臂相邻两次评估之间的 env 步数："
    )
    lines.append("")
    lines.append(
        f"| {'臂':<16}{'n_eval':>8}{'首点 t':>12}{'末点 t':>12}"
        f"{'平均步长':>14}{'最短':>12}{'最长':>12} |"
    )
    lines.append("|" + "|".join("-" * 12 for _ in range(7)) + "|")
    for arm, _constraint, _coop in ARMS:
        curve = curves[arm]
        if not curve:
            continue
        cad = eval_cadence(curve)
        mean_text = "-" if cad["mean_step"] is None else f"{cad['mean_step']:.0f}"
        min_text = "-" if cad["min_step"] is None else f"{cad['min_step']:.0f}"
        max_text = "-" if cad["max_step"] is None else f"{cad['max_step']:.0f}"
        lines.append(
            f"| {arm:<16}{len(curve):>8}{int(curve[0][0]):>12}{int(curve[-1][0]):>12}"
            f"{mean_text:>14}{min_text:>12}{max_text:>12} |"
        )
    lines.append("")
    lines.append(
        "各臂都在**同一个 ~10k 步的评估网格**上，所以第 n 个评估点对应各臂自己"
        "训练到 ~10k·n 步的位置——**索引配对 ≈ 步数配对**，与这个臂是几点钟开始跑的无关。"
        "同索引上各臂 `t_env` 的实际差别只有几十到几百步（实测 61–282 步，"
        "不到一个评估块的 3%），来自各臂回合长度的抖动。"
    )
    lines.append("")
    lines.append(
        "因此「某臂中途崩溃重启」**不会**让索引配对失真：重启后的臂从同一个离线"
        "初始化重新开始，第 n 个评估点仍然对应它自己的 ~10k·n 步。真正的后果是"
        "（a）这个臂要重跑一遍、队列被拖后；（b）early 窗口测的是重启后那条轨迹，"
        "数值会略有不同，但估计的目标量不变；（c）若最终放弃，该臂的 early 窗口"
        "点数会变少、统计功效下降——这需要在报告里说明，而不是当成有效读数。"
    )
    lines.append("")
    lines.append(
        "下表是配对一致性的**兜底检查**：若各臂的评估网格不再一致"
        "（例如某臂带着不同的评估节奏被重启，或一个日志里混进了两次运行），"
        f"同索引的 `t_env` 最大差会超过 {MISALIGN_LIMIT_STEPS} 步，此时读数不可信，"
        "**撤回**登记结论、改记证据不足。"
    )
    lines.append("")
    lines.append(
        f"| {'读数':<30}{'n':>6}{'t_env 最大差':>16}{'平均差':>14}{'最差序号':>12} |"
    )
    lines.append("|" + "|".join("-" * 12 for _ in range(5)) + "|")
    for key, stat in align_stats.items():
        spread_text = "-" if stat["max_spread"] is None else str(int(stat["max_spread"]))
        mean_text = "-" if stat["mean_spread"] is None else f"{stat['mean_spread']:.0f}"
        index_text = "-" if stat["at_index"] is None else str(stat["at_index"])
        lines.append(
            f"| {key:<30}{stat['n']:>6}{spread_text:>16}{mean_text:>14}{index_text:>12} |"
        )
    lines.append("")
    lines.append(
        f"**检查结果：{'通过（配对成立）' if align_ok else '未通过（配对失效，登记结论已撤回）'}**。"
    )
    lines.append("")

    # ------------------------------------------------------ detectability
    lines.append("### 5.5 可检测效应量（探索性，不参与判读）")
    lines.append("")
    lines.append(
        "「CI 含 0」只有在知道**这个实验能看见多小的效应**时才有意义。"
        "由 95% CI 的半宽反推标准误（SE = (hi − lo) / 3.92），"
        "再按 80% 功效所需的 (1.96 + 0.84)·SE，给出主窗口上各读数"
        "**可检测的最小效应**：真实效应若小于这个量级，登记规则读到的就是「含 0」。"
    )
    lines.append("")
    lines.append(
        f"| {'读数':<30}{'n':>6}{'点自助 SE':>14}{'块自助 SE':>14}"
        f"{'可检测效应(点)':>18}{'可检测效应(块)':>18} |"
    )
    lines.append("|" + "|".join("-" * 12 for _ in range(6)) + "|")
    detectability = {}
    for name, stat in readings.items():
        block_stat = block_stats.get(name, {})
        point_se = (
            None
            if stat.get("lo") is None or stat.get("hi") is None
            else (float(stat["hi"]) - float(stat["lo"])) / 3.92
        )
        block_se = (
            None
            if block_stat.get("lo") is None or block_stat.get("hi") is None
            else (float(block_stat["hi"]) - float(block_stat["lo"])) / 3.92
        )
        mde_point = minimum_detectable_effect(stat)
        mde_block = minimum_detectable_effect(block_stat)
        detectability[name] = {
            "n": stat.get("n", 0),
            "se_point": point_se,
            "se_block": block_se,
            "mde_point": mde_point,
            "mde_block": mde_block,
        }
        se_point_text = "-" if point_se is None else f"{point_se:.3f}"
        se_block_text = "-" if block_se is None else f"{block_se:.3f}"
        mde_point_text = "-" if mde_point is None else f"{mde_point:.2f}"
        mde_block_text = "-" if mde_block is None else f"{mde_block:.2f}"
        lines.append(
            f"| {name:<30}{stat.get('n', 0):>6}{se_point_text:>14}{se_block_text:>14}"
            f"{mde_point_text:>18}{mde_block_text:>18} |"
        )
    lines.append("")
    lines.append(
        "这个量**不参与判读**，它只是给「含 0」一个参照："
        "读出「含 0」时，应当同时说明这个实验当时看不见多小的效应。"
        "块自助那一列一般更大——因为自相关让有效样本量变小，"
        "所以同一个规则对自相关稳健性的要求更高。"
    )
    lines.append("")

    summary["manipulation_check"] = {
        "per_constraint": {
            key: {"class": manip_classes[key], **stat}
            for key, stat in manip_stats.items()
        },
        "effective": manip_effective,
    }
    summary["autocorrelation"] = {
        "P1": {
            "n": len(p1_points),
            "lag1": lag1_autocorr(p1_points),
            "n_eff": effective_sample_size(p1_points),
        },
        "P2": {
            "n": len(p2_points),
            "lag1": lag1_autocorr(p2_points),
            "n_eff": effective_sample_size(p2_points),
        },
    }
    summary["detectability"] = detectability
    summary["verdict"] = {
        "window": primary,
        "P1": {"class": p1, **readings.get("P1 交互 prior (matched−broken)", {})},
        "P2": {"class": p2, **readings.get("P2 prior − anchor (matched)", {})},
        "conclusion": conclusion,
        "qualifier": qualifier,
        "manipulation_effective": manip_effective,
        "block_bootstrap": {
            "P1": {
                "class": block_classes.get("P1 交互 prior (matched−broken)", "无数据"),
                **block_stats.get("P1 交互 prior (matched−broken)", {}),
            },
            "P2": {
                "class": block_classes.get("P2 prior − anchor (matched)", "无数据"),
                **block_stats.get("P2 prior − anchor (matched)", {}),
            },
            "vetoed": vetoed,
            "reasons": veto_reasons,
        },
        "block_qualifier": block_qualifier,
        "pairing_alignment": {
            "ok": align_ok,
            "limit_steps": MISALIGN_LIMIT_STEPS,
            "misaligned": misaligned,
            "P1": align_stats["P1"],
            "P2": align_stats["P2"],
        },
        "align_qualifier": align_qualifier,
        "constraint_activity": {
            "ok": activity_ok,
            "inert_decisive_arms": inert_decisive,
            "per_arm": activity,
        },
        "activity_qualifier": activity_qualifier,
    }

    text = "\n".join(lines)
    print(text)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}")
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.markdown}")

    if missing:
        print(f"\nwarning: no data for {', '.join(missing)}")


if __name__ == "__main__":
    main()
