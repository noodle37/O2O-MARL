#!/usr/bin/env python3
# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Experiment 2b post-processing: per-state analysis of arm C's training probe.

Reads the probe.jsonl that pv/probe.py appends while arm C trains, then:

  * bins every recorded state by team deviation D(s) using the held-out
    thresholds fixed before any online data was seen;
  * reports, per bin, the online-state share, the offline-value error
    |Q^off_tot - G|, the memory-constraint trigger rate, and how often the
    memory target is closer to G than the TD target;
  * regresses log|Q^off_tot - G| on the D bins and on t, to separate "the
    knowledge has gone stale" from "the schedule simply ran out of time";
  * runs the shuffled-D negative control: within each probe point the D values
    are permuted across states and the whole bin analysis is repeated.  The
    dependence on D must disappear; if it does not, the dependence comes from
    the binning or the sampling rather than from the states;
  * reports the two mismatch rates from the design:
      stale retained   P(memory trigger AND D > q95)
      valid released   P(D <= q50 AND memory target closer to G AND
                        lambda already at its floor)

The per-state section requested by the design is also implemented here:

  * probe-point fixed effects: subtract each probe point's mean error before
    binning, so the D contrast is a cross-state difference at a fixed time;
  * cluster bootstrap over probe points (2000 draws), for the lambda-floor
    segment and for the full course;
  * novelty tertiles within each probe point, to separate team deviation from
    state novelty;
  * a second metric, the relative error |Q^off - G| - |TD - G|;
  * q90/q95/q99 cutoff sensitivity, a shuffled-D negative control after the
    fixed-effect transform, and the R1-R5 branch statement.

Outputs a JSON summary plus the aggregated rows that analyze_pv_exp2.py folds
into table 2 and figure 2.

Usage:
  python scripts/run_pv_exp2b.py --probe checkpoints/pv_exp1/5m_vs_6m/C_seed1/probe.jsonl \
      --thresholds checkpoints/pv_behaviour/5m_vs_6m/thresholds.json \
      --out /tmp/pv_exp2b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BINS = ("<=q50", "q50-q95", "q95-q99", ">q99")
EPS = 1e-6


def load_probe(path: str) -> List[dict]:
    rows: List[dict] = []
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


def bin_edges(thresholds: dict) -> List[float]:
    return [thresholds["q50"], thresholds["q95"], thresholds["q99"]]


def assign_bins(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Bin index per value; -1 where the value is not finite."""
    out = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    out[finite] = 0
    for k, edge in enumerate(edges):
        out[finite & (values > edge)] = k + 1
    return out


def _rate(mask: np.ndarray, cond: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(cond[mask].mean())


def analyse_rows(rows: Sequence[dict], edges: Sequence[float], lambda_floor: float) -> dict:
    """Pooled per-bin statistics plus the two mismatch rates."""
    dev, err, trig, closer, lam, t_env = [], [], [], [], [], []
    for row in rows:
        arrays = row.get("arrays") or {}
        d = np.asarray(arrays.get("team_deviation", []), dtype=np.float64)
        e = np.asarray(arrays.get("abs_off_err", []), dtype=np.float64)
        tr = np.asarray(arrays.get("memory_trigger", []), dtype=np.float64)
        cl = np.asarray(arrays.get("memory_closer", []), dtype=np.float64)
        n = min(len(d), len(e), len(tr), len(cl))
        if n == 0:
            continue
        dev.append(d[:n])
        err.append(e[:n])
        trig.append(tr[:n])
        closer.append(cl[:n])
        lam.append(np.full(n, float(row.get("lambda_memory", float("nan")))))
        t_env.append(np.full(n, float(row.get("t_env", 0))))

    if not dev:
        return {"n_states": 0, "bins": {}, "mismatch": {}}

    dev = np.concatenate(dev)
    err = np.concatenate(err)
    trig = np.concatenate(trig)
    closer = np.concatenate(closer)
    lam = np.concatenate(lam)
    t_env = np.concatenate(t_env)

    finite = np.isfinite(dev) & np.isfinite(err)
    dev_f, err_f = dev[finite], err[finite]
    trig_f, closer_f = trig[finite], closer[finite]
    lam_f, t_f = lam[finite], t_env[finite]

    bins = assign_bins(dev_f, edges)
    per_bin: Dict[str, dict] = {}
    total = max(int(finite.sum()), 1)
    for k, name in enumerate(BINS):
        sel = bins == k
        if sel.sum() == 0:
            per_bin[name] = {
                "share": 0.0,
                "n_states": 0,
                "mean_abs_err": float("nan"),
                "median_abs_err": float("nan"),
                "trigger_rate": float("nan"),
                "memory_closer_rate": float("nan"),
            }
            continue
        per_bin[name] = {
            "share": float(sel.sum()) / total,
            "n_states": int(sel.sum()),
            "mean_abs_err": float(err_f[sel].mean()),
            "median_abs_err": float(np.median(err_f[sel])),
            "trigger_rate": _rate(sel, trig_f),
            "memory_closer_rate": _rate(sel, closer_f),
            "closer_given_trigger": _rate(sel & (trig_f > 0.5), closer_f),
        }

    q95 = edges[1]
    stale = finite & (dev > q95) & (trig > 0.5)
    valid = (
        finite
        & (dev <= edges[0])
        & (closer > 0.5)
        & np.isfinite(lam)
        & (lam <= lambda_floor + 1e-9)
    )
    mismatch = {
        "stale_retained": float(stale.sum()) / max(int(finite.sum()), 1),
        "valid_released": float(valid.sum()) / max(int(finite.sum()), 1),
        "n_states": int(finite.sum()),
    }
    return {
        "n_states": int(finite.sum()),
        "bins": per_bin,
        "mismatch": mismatch,
        "coverage_lambda": [
            (float(row.get("lambda_memory", float("nan"))), float(row.get("coverage_frac", float("nan"))))
            for row in rows
            if np.isfinite(row.get("coverage_frac", float("nan")))
        ],
    }


def regression(rows: Sequence[dict], edges: Sequence[float]) -> dict:
    """log|Q^off - G| on D-bin dummies and on t (log-scaled steps)."""
    dev, err, t_env = [], [], []
    for row in rows:
        arrays = row.get("arrays") or {}
        d = np.asarray(arrays.get("team_deviation", []), dtype=np.float64)
        e = np.asarray(arrays.get("abs_off_err", []), dtype=np.float64)
        n = min(len(d), len(e))
        if n == 0:
            continue
        dev.append(d[:n])
        err.append(e[:n])
        t_env.append(np.full(n, float(row.get("t_env", 0))))
    if not dev:
        return {}
    dev = np.concatenate(dev)
    err = np.concatenate(err)
    t_env = np.concatenate(t_env)
    keep = np.isfinite(dev) & np.isfinite(err) & (err > EPS)
    if keep.sum() < 32:
        return {"n": int(keep.sum()), "note": "too few states for a stable fit"}
    y = np.log(err[keep])
    bins = assign_bins(dev[keep], edges)
    x_t = np.log1p(t_env[keep])
    # Design matrix: intercept, t, and the three non-baseline bin dummies.
    cols = [np.ones(int(keep.sum())), x_t]
    names = ["intercept", "log_t"]
    for k in range(1, len(BINS)):
        cols.append((bins == k).astype(np.float64))
        names.append(BINS[k])
    design = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = max(int(keep.sum()) - design.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(design.T @ design)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(design.shape[1], float("nan"))
    out = {"n": int(keep.sum()), "terms": {}}
    for name, value, err_se in zip(names, coef, se):
        out["terms"][name] = {
            "coef": float(value),
            "se": float(err_se),
            # One-sided: is the D-bin coefficient positive once t is controlled?
            "z": float(value / err_se) if err_se and err_se == err_se else float("nan"),
        }
    return out


def shuffled_control(rows: Sequence[dict], edges: Sequence[float], n_repeats: int = 20) -> dict:
    """Permute D within each probe point; the bin dependence must vanish."""
    real = analyse_rows(rows, edges, lambda_floor=float("inf"))
    if real.get("n_states", 0) == 0:
        return {}
    base = {
        name: real["bins"][name]["mean_abs_err"] for name in BINS
    }
    spread = [
        abs(base[b] - base[BINS[0]])
        for b in BINS[1:]
        if base[b] == base[b] and base[BINS[0]] == base[BINS[0]]
    ]
    observed = float(np.mean(spread)) if spread else float("nan")

    null_spreads = []
    rng = np.random.default_rng(0)
    for _ in range(n_repeats):
        shuffled = []
        for row in rows:
            arrays = row.get("arrays") or {}
            d = np.asarray(arrays.get("team_deviation", []), dtype=np.float64)
            if d.size == 0:
                continue
            perm = rng.permutation(d.size)
            copied = dict(row)
            copied_arrays = dict(arrays)
            copied_arrays["team_deviation"] = d[perm].tolist()
            copied["arrays"] = copied_arrays
            shuffled.append(copied)
        stats = analyse_rows(shuffled, edges, lambda_floor=float("inf"))
        if stats.get("n_states", 0) == 0:
            continue
        vals = [
            abs(stats["bins"][b]["mean_abs_err"] - stats["bins"][BINS[0]]["mean_abs_err"])
            for b in BINS[1:]
            if stats["bins"][b]["mean_abs_err"] == stats["bins"][b]["mean_abs_err"]
            and stats["bins"][BINS[0]]["mean_abs_err"] == stats["bins"][BINS[0]]["mean_abs_err"]
        ]
        if vals:
            null_spreads.append(float(np.mean(vals)))
    null_mean = float(np.mean(null_spreads)) if null_spreads else float("nan")
    return {
        "observed_spread": observed,
        "null_spread_mean": null_mean,
        "ratio": float(observed / null_mean) if null_mean and null_mean == null_mean else float("nan"),
        "n_repeats": len(null_spreads),
        "interpretation": (
            "dependence survives shuffling: it comes from the binning or sampling, not the states"
            if null_mean == null_mean and observed > 2.0 * null_mean
            else "dependence disappears under shuffling: it tracks the states"
        ),
    }


# ---------------------------------------------------------------------------
# Per-state analyses requested by the design (section 2.4):
#   1. probe-point fixed effects: remove each probe point's mean before binning,
#      so the D effect is the cross-state difference at a fixed time;
#   2. cluster bootstrap over probe points (2000 draws), for both the
#      lambda-floor segment and the full course;
#   3. novelty tertiles within each probe point, to separate team deviation
#      from state novelty;
#   4. a second metric, the relative error |Q^off - G| - |TD - G|.
# ---------------------------------------------------------------------------


def point_data(rows: Sequence[dict], lambda_floor: Optional[float], need_td: bool) -> List[dict]:
    """Per-probe-point arrays restricted to finite states (and to lambda floor)."""
    points: List[dict] = []
    for row in rows:
        lam = float(row.get("lambda_memory", float("nan")))
        if lambda_floor is not None and not (lam <= lambda_floor + 1e-9):
            continue
        arrays = row.get("arrays") or {}
        d = np.asarray(arrays.get("team_deviation", []), dtype=np.float64)
        e = np.asarray(arrays.get("abs_off_err", []), dtype=np.float64)
        td = np.asarray(arrays.get("abs_td_err", []), dtype=np.float64)
        nov = np.asarray(arrays.get("novelty", []), dtype=np.float64)
        n = min(len(d), len(e), len(td), len(nov))
        if n == 0:
            continue
        keep = np.isfinite(d[:n]) & np.isfinite(e[:n])
        if need_td:
            keep &= np.isfinite(td[:n])
        if not keep.any():
            continue
        points.append(
            {
                "t": float(row.get("t_env", 0.0)),
                "lambda": lam,
                "d": d[:n][keep],
                "e": e[:n][keep],
                "td": td[:n][keep] if need_td else None,
                "nov": nov[:n][keep],
            }
        )
    return points


def centred_points(points: Sequence[dict], metric: str) -> List[dict]:
    """Subtract each probe point's mean error (the fixed effect)."""
    out: List[dict] = []
    for p in points:
        if metric == "rel":
            if p["td"] is None:
                continue
            y = p["e"] - p["td"]
        else:
            y = p["e"]
        if y.size == 0:
            continue
        r = y - float(y.mean())
        out.append({"r": r, "d": p["d"], "nov": p["nov"], "t": p["t"]})
    return out


def tertile_labels(nov: np.ndarray) -> np.ndarray:
    labels = np.full(nov.shape, -1, dtype=np.int64)
    ok = np.isfinite(nov)
    if ok.sum() < 3:
        return labels
    q1, q2 = np.quantile(nov[ok], [1.0 / 3.0, 2.0 / 3.0])
    labels[ok & (nov <= q1)] = 0
    labels[ok & (nov > q1) & (nov <= q2)] = 1
    labels[ok & (nov > q2)] = 2
    return labels


def point_bin_stats(
    points: Sequence[dict], edges: Sequence[float], cutoff: float, with_strata: bool = False
) -> List[dict]:
    """Per-point sums/counts of the centred error for the four D bins."""
    out: List[dict] = []
    for p in points:
        d, r = p["d"], p["r"]
        bins = assign_bins(d, edges)
        valid = bins >= 0
        sums = np.bincount(bins[valid], weights=r[valid], minlength=len(BINS)).astype(float)
        counts = np.bincount(bins[valid], minlength=len(BINS)).astype(float)
        hi = d > cutoff
        lo = d <= edges[0]
        entry = {
            "sums": sums,
            "counts": counts,
            "hi_sum": float(r[hi].sum()),
            "hi_n": float(hi.sum()),
            "lo_sum": float(r[lo].sum()),
            "lo_n": float(lo.sum()),
        }
        if with_strata:
            labels = tertile_labels(p["nov"])
            s_sums = np.zeros((3, len(BINS)))
            s_counts = np.zeros((3, len(BINS)))
            s_hi_sum = np.zeros(3)
            s_hi_n = np.zeros(3)
            s_lo_sum = np.zeros(3)
            s_lo_n = np.zeros(3)
            for k in range(3):
                sel = labels == k
                if not sel.any():
                    continue
                bb = bins[sel]
                vv = bb >= 0
                s_sums[k] = np.bincount(bb[vv], weights=r[sel][vv], minlength=len(BINS))
                s_counts[k] = np.bincount(bb[vv], minlength=len(BINS))
                dd = d[sel]
                hh = dd > cutoff
                ll = dd <= edges[0]
                s_hi_sum[k] = float(r[sel][hh].sum())
                s_hi_n[k] = float(hh.sum())
                s_lo_sum[k] = float(r[sel][ll].sum())
                s_lo_n[k] = float(ll.sum())
            entry.update(
                {
                    "strata_sums": s_sums,
                    "strata_counts": s_counts,
                    "strata_hi_sum": s_hi_sum,
                    "strata_hi_n": s_hi_n,
                    "strata_lo_sum": s_lo_sum,
                    "strata_lo_n": s_lo_n,
                }
            )
        out.append(entry)
    return out


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def observed_stats(stats: Sequence[dict]) -> dict:
    if not stats:
        return {}
    sums = np.sum([s["sums"] for s in stats], axis=0)
    counts = np.sum([s["counts"] for s in stats], axis=0)
    means = _ratio(sums, counts)
    hi_sum = float(np.sum([s["hi_sum"] for s in stats]))
    hi_n = float(np.sum([s["hi_n"] for s in stats]))
    lo_sum = float(np.sum([s["lo_sum"] for s in stats]))
    lo_n = float(np.sum([s["lo_n"] for s in stats]))
    return {
        "bin_means": means,
        "bin_counts": counts,
        "hi_mean": hi_sum / hi_n if hi_n > 0 else float("nan"),
        "lo_mean": lo_sum / lo_n if lo_n > 0 else float("nan"),
        "hi_n": hi_n,
        "lo_n": lo_n,
    }


def cluster_bootstrap(
    stats: Sequence[dict], n_boot: int, rng: np.random.Generator, with_strata: bool = False
) -> dict:
    """Resample probe points with replacement; returns the bootstrap draws."""
    n = len(stats)
    if n == 0 or n_boot <= 0:
        return {}
    idx = rng.integers(0, n, size=(n_boot, n))
    sums = np.stack([s["sums"] for s in stats])
    counts = np.stack([s["counts"] for s in stats])
    bs = sums[idx].sum(axis=1)
    bc = counts[idx].sum(axis=1)
    bin_means = _ratio(bs, bc)
    hs = np.concatenate([[s["hi_sum"]] for s in stats])
    hn = np.concatenate([[s["hi_n"]] for s in stats])
    ls = np.concatenate([[s["lo_sum"]] for s in stats])
    ln = np.concatenate([[s["lo_n"]] for s in stats])
    hi_mean = _ratio(hs[idx].sum(axis=1), hn[idx].sum(axis=1))
    lo_mean = _ratio(ls[idx].sum(axis=1), ln[idx].sum(axis=1))
    out = {"bin_means": bin_means, "diff": hi_mean - lo_mean}
    if with_strata:
        ss = np.stack([s["strata_sums"] for s in stats])
        hs_s = np.stack([s["strata_hi_sum"] for s in stats])
        hn_s = np.stack([s["strata_hi_n"] for s in stats])
        ls_s = np.stack([s["strata_lo_sum"] for s in stats])
        ln_s = np.stack([s["strata_lo_n"] for s in stats])
        hi_s = _ratio(hs_s[idx].sum(axis=1), hn_s[idx].sum(axis=1))
        lo_s = _ratio(ls_s[idx].sum(axis=1), ln_s[idx].sum(axis=1))
        out["strata_diff"] = hi_s - lo_s
    return out


def ci(values: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> List[float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    low, high = np.percentile(vals, [lo, hi])
    return [float(low), float(high)]


def is_monotone(means: np.ndarray) -> bool:
    """Non-decreasing across the four bins, allowing one tie."""
    vals = means[np.isfinite(means)]
    if vals.size < 2:
        return False
    diffs = np.diff(vals)
    drops = int((diffs < -1e-12).sum())
    ties = int((np.abs(diffs) <= 1e-12).sum())
    return drops == 0 and ties <= 1


def shuffled_null(
    points: Sequence[dict], edges: Sequence[float], cutoff: float, n_repeats: int, rng: np.random.Generator
) -> dict:
    """Permute D within each probe point; rebuild the high-minus-low contrast."""
    vals: List[float] = []
    bin_means: List[np.ndarray] = []
    for _ in range(n_repeats):
        hi_sum = hi_n = lo_sum = lo_n = 0.0
        sums = np.zeros(len(BINS))
        counts = np.zeros(len(BINS))
        for p in points:
            d = rng.permutation(p["d"])
            hi = d > cutoff
            lo = d <= edges[0]
            hi_sum += float(p["r"][hi].sum())
            hi_n += float(hi.sum())
            lo_sum += float(p["r"][lo].sum())
            lo_n += float(lo.sum())
            bins = assign_bins(d, edges)
            valid = bins >= 0
            sums += np.bincount(bins[valid], weights=p["r"][valid], minlength=len(BINS))
            counts += np.bincount(bins[valid], minlength=len(BINS))
        vals.append(hi_sum / hi_n - lo_sum / lo_n if hi_n > 0 and lo_n > 0 else float("nan"))
        bin_means.append(_ratio(sums, counts))
    arr = np.asarray(vals, dtype=np.float64)
    band = ci(arr)
    bm = np.asarray(bin_means)
    return {
        "null_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
        "null_ci": band,
        "null_bin_mean": np.nanmean(bm, axis=0).tolist(),
        "covers_zero": bool(band[0] <= 0.0 <= band[1]) if np.isfinite(band).all() else False,
        "n_repeats": int(np.isfinite(arr).sum()),
    }


def segment_analysis(
    points: Sequence[dict],
    edges: Sequence[float],
    thresholds: dict,
    metric: str,
    n_boot: int,
    n_null: int,
    rng: np.random.Generator,
    with_strata: bool = False,
) -> dict:
    name = metric
    centred = centred_points(points, metric)
    if not centred:
        return {}
    cutoffs = {k: float(thresholds[k]) for k in ("q90", "q95", "q99") if k in thresholds}

    def run(cutoff: float, strata: bool) -> dict:
        stats = point_bin_stats(centred, edges, cutoff, with_strata=strata)
        obs = observed_stats(stats)
        boot = cluster_bootstrap(stats, n_boot, rng, with_strata=strata)
        return {"stats": stats, "obs": obs, "boot": boot}

    main_cut = cutoffs.get("q95", edges[1])
    main = run(main_cut, with_strata)
    obs = main["obs"]
    boot = main["boot"]
    diff_ci = ci(boot.get("diff", np.array([])))
    per_bin = {}
    for k, label in enumerate(BINS):
        share = obs["bin_counts"][k] / max(obs["bin_counts"].sum(), 1)
        per_bin[label] = {
            "mean": float(obs["bin_means"][k]),
            "ci": ci(boot["bin_means"][:, k]) if boot else [float("nan")] * 2,
            "n_states": int(obs["bin_counts"][k]),
            "share": float(share),
        }
    out = {
        "metric": name,
        "n_points": len(centred),
        "n_states": int(obs["bin_counts"].sum()),
        "bins": per_bin,
        "hi_minus_low": {
            "cutoff": float(main_cut),
            "value": float(obs["hi_mean"] - obs["lo_mean"]),
            "ci": diff_ci,
            "n_hi": int(obs["hi_n"]),
            "n_lo": int(obs["lo_n"]),
        },
        "monotone": is_monotone(obs["bin_means"]),
        "cutoff_sensitivity": {},
    }
    for key, cutoff in cutoffs.items():
        if abs(cutoff - main_cut) <= 1e-15:
            continue
        alt = run(cutoff, False)
        out["cutoff_sensitivity"][key] = {
            "cutoff": cutoff,
            "value": float(alt["obs"]["hi_mean"] - alt["obs"]["lo_mean"]),
            "ci": ci(alt["boot"].get("diff", np.array([]))),
        }

    null = shuffled_null(centred, edges, main_cut, n_null, rng)
    out["shuffled_d"] = {
        "observed": float(obs["hi_mean"] - obs["lo_mean"]),
        "null_mean": null["null_mean"],
        "null_ci": null["null_ci"],
        "null_bin_mean": null.get("null_bin_mean"),
        "covers_zero": null["covers_zero"],
        "n_repeats": null["n_repeats"],
    }

    if with_strata and boot.get("strata_diff") is not None:
        labels = ("novelty_low", "novelty_mid", "novelty_high")
        s_sums = np.sum([s["strata_sums"] for s in main["stats"]], axis=0)
        s_counts = np.sum([s["strata_counts"] for s in main["stats"]], axis=0)
        s_hi_sum = np.sum([s["strata_hi_sum"] for s in main["stats"]], axis=0)
        s_hi_n = np.sum([s["strata_hi_n"] for s in main["stats"]], axis=0)
        s_lo_sum = np.sum([s["strata_lo_sum"] for s in main["stats"]], axis=0)
        s_lo_n = np.sum([s["strata_lo_n"] for s in main["stats"]], axis=0)
        s_obs = _ratio(s_hi_sum, s_hi_n) - _ratio(s_lo_sum, s_lo_n)
        strata = {}
        for k, label in enumerate(labels):
            strata[label] = {
                "value": float(s_obs[k]),
                "ci": ci(boot["strata_diff"][:, k]),
                "n_states": int(s_counts[k].sum()),
                "n_hi": int(s_hi_n[k]),
                "n_lo": int(s_lo_n[k]),
            }
        out["novelty_strata"] = strata
    return out


def standardized_regression(points: Sequence[dict], edges: Sequence[float]) -> dict:
    """Pooled OLS of the error on D-bin dummies and log t, fully standardized.

    Time is constant within a probe point, so the fixed-effect transform would
    absorb it; this pooled fit is what compares the D and time coefficients.
    """
    err, dev, t_env = [], [], []
    for p in points:
        n = min(len(p["e"]), len(p["d"]))
        if n == 0:
            continue
        err.append(p["e"][:n])
        dev.append(p["d"][:n])
        t_env.append(np.full(n, p["t"]))
    if not err:
        return {}
    err = np.concatenate(err)
    dev = np.concatenate(dev)
    t_env = np.concatenate(t_env)
    keep = np.isfinite(err) & np.isfinite(dev)
    if keep.sum() < 32:
        return {"n": int(keep.sum()), "note": "too few states for a stable fit"}
    y = err[keep]
    bins = assign_bins(dev[keep], edges)
    log_t = np.log1p(t_env[keep])
    cols = [np.ones(int(keep.sum()))]
    names = ["intercept"]

    def z(col: np.ndarray) -> np.ndarray:
        sd = float(col.std())
        return (col - col.mean()) / sd if sd > 0 else col * 0.0

    cols.append(z(log_t))
    names.append("log_t")
    for k in range(1, len(BINS)):
        cols.append(z((bins == k).astype(np.float64)))
        names.append(BINS[k])
    design = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(design, z(y), rcond=None)
    terms = {name: float(value) for name, value in zip(names, coef)}
    d_names = BINS[1:]
    abs_d = max(abs(terms[name]) for name in d_names)
    abs_t = abs(terms["log_t"])
    return {
        "n": int(keep.sum()),
        "terms": terms,
        "abs_d_max": float(abs_d),
        "abs_log_t": float(abs_t),
        "state_dominant": bool(abs_d > abs_t),
        "note": "y, log_t and the bin dummies are all standardized (z-score) before the fit",
    }


def rule_checks(floor_abs: dict, full_abs: dict, rel_floor: dict) -> dict:
    if not floor_abs:
        return {}
    hi = floor_abs.get("hi_minus_low", {})
    r1 = bool(hi.get("ci", [float("nan")])[0] > 0 and floor_abs.get("monotone"))
    r2 = bool(floor_abs.get("shuffled_d", {}).get("covers_zero"))
    strata = floor_abs.get("novelty_strata", {})
    positive = [v for v in strata.values() if v["ci"][0] > 0 and v["value"] > 0]
    negative = [v for v in strata.values() if v["ci"][1] < 0 and v["value"] < 0]
    r3 = len(positive) >= 2 or len(negative) >= 2
    reg = full_abs.get("standardized_regression", {})
    r4 = bool(reg.get("state_dominant"))
    main_sign = np.sign(hi.get("value", float("nan")))
    cut_same = all(
        np.sign(v["value"]) == main_sign
        for v in floor_abs.get("cutoff_sensitivity", {}).values()
        if np.isfinite(v.get("value", float("nan")))
    )
    rel_same = np.sign(rel_floor.get("hi_minus_low", {}).get("value", float("nan"))) == main_sign
    r5 = bool(cut_same and rel_same)

    if not r1:
        branch = "C"
    elif not r2:
        branch = "C (negative control failed)"
    elif not r3:
        branch = "B"
    elif not r4:
        branch = "D"
    else:
        branch = "A"
    return {
        "R1_state_dependence": r1,
        "R2_negative_control": r2,
        "R3_separable_from_novelty": r3,
        "R4_state_over_time": r4,
        "R5_robustness": r5,
        "suggested_branch": branch,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--lambda-floor", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.thresholds, encoding="utf-8") as handle:
        thresholds = json.load(handle)
    edges = bin_edges(thresholds)
    rows = load_probe(args.probe)
    if not rows:
        raise SystemExit(f"FATAL: no probe rows in {args.probe}")

    rng = np.random.default_rng(args.seed)
    segments = {
        "floor": segment_analysis(
            point_data(rows, args.lambda_floor, need_td=False),
            edges, thresholds, "abs", args.n_boot, args.n_null, rng, with_strata=True,
        ),
        "full": segment_analysis(
            point_data(rows, None, need_td=False),
            edges, thresholds, "abs", args.n_boot, args.n_null, rng,
        ),
    }
    full_points = point_data(rows, None, need_td=False)
    segments["full"]["standardized_regression"] = standardized_regression(full_points, edges)
    segments["rel_floor"] = segment_analysis(
        point_data(rows, args.lambda_floor, need_td=True),
        edges, thresholds, "rel", args.n_boot, args.n_null, rng,
    )
    segments["rel_full"] = segment_analysis(
        point_data(rows, None, need_td=True),
        edges, thresholds, "rel", args.n_boot, args.n_null, rng,
    )

    summary = {
        "probe": args.probe,
        "thresholds": {k: thresholds.get(k) for k in ("q50", "q90", "q95", "q99")},
        "n_probe_points": len(rows),
        "n_boot": args.n_boot,
        "n_null": args.n_null,
        "seed": args.seed,
        "pooled": analyse_rows(rows, edges, args.lambda_floor),
        "regression": regression(rows, edges),
        "shuffled_control": shuffled_control(rows, edges),
        "fixed_effects": {
            "lambda_floor": segments["floor"],
            "full_course": segments["full"],
            "relative_error_lambda_floor": segments["rel_floor"],
            "relative_error_full_course": segments["rel_full"],
        },
        "novelty_strata": segments["floor"].get("novelty_strata", {}),
        "rule_checks": rule_checks(
            segments["floor"], segments["full"], segments["rel_floor"]
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    pooled = summary["pooled"]
    print(f"[2b] probe points={len(rows)} states={pooled.get('n_states')}")
    for name in BINS:
        cell = pooled["bins"].get(name, {})
        print(
            f"[2b]   {name:<9} share={cell.get('share', float('nan')):.3f} "
            f"|err|={cell.get('mean_abs_err', float('nan')):.3f} "
            f"trigger={cell.get('trigger_rate', float('nan')):.3f} "
            f"closer={cell.get('memory_closer_rate', float('nan')):.3f}"
        )
    mismatch = pooled.get("mismatch", {})
    print(
        f"[2b] stale_retained={mismatch.get('stale_retained', float('nan')):.4f} "
        f"valid_released={mismatch.get('valid_released', float('nan')):.4f}"
    )
    reg = summary["regression"].get("terms", {})
    for name in BINS[1:]:
        if name in reg:
            print(f"[2b]   regress {name}: coef={reg[name]['coef']:.4f} z={reg[name]['z']:.2f}")
    ctrl = summary["shuffled_control"]
    print(f"[2b] shuffle control: {ctrl.get('interpretation', 'n/a')}")
    for label, seg in (
        ("floor", summary["fixed_effects"]["lambda_floor"]),
        ("full", summary["fixed_effects"]["full_course"]),
        ("rel-floor", summary["fixed_effects"]["relative_error_lambda_floor"]),
    ):
        if not seg:
            continue
        hi = seg["hi_minus_low"]
        print(
            f"[2b] FE {label:<9} hi-lo={hi['value']:.4f} "
            f"CI=[{hi['ci'][0]:.4f}, {hi['ci'][1]:.4f}] "
            f"monotone={seg['monotone']} states={seg['n_states']}"
        )
        for s_name, s_val in seg.get("novelty_strata", {}).items():
            print(
                f"[2b]   {label:<9} {s_name:<12} diff={s_val['value']:.4f} "
                f"CI=[{s_val['ci'][0]:.4f}, {s_val['ci'][1]:.4f}] n={s_val['n_states']}"
            )
    reg = summary["fixed_effects"]["full_course"].get("standardized_regression", {})
    if reg:
        print(
            f"[2b] std regression (full): |D|max={reg.get('abs_d_max', float('nan')):.4f} "
            f"|log_t|={reg.get('abs_log_t', float('nan')):.4f} "
            f"state_dominant={reg.get('state_dominant')}"
        )
    checks = summary.get("rule_checks", {})
    if checks:
        print(f"[2b] rule checks: {checks}")
    print(f"[2b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
