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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--lambda-floor", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.thresholds, encoding="utf-8") as handle:
        thresholds = json.load(handle)
    edges = bin_edges(thresholds)
    rows = load_probe(args.probe)
    if not rows:
        raise SystemExit(f"FATAL: no probe rows in {args.probe}")

    summary = {
        "probe": args.probe,
        "thresholds": {k: thresholds.get(k) for k in ("q50", "q95", "q99")},
        "n_probe_points": len(rows),
        "pooled": analyse_rows(rows, edges, args.lambda_floor),
        "regression": regression(rows, edges),
        "shuffled_control": shuffled_control(rows, edges),
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
    print(f"[2b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
