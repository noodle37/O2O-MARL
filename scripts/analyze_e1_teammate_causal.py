#!/usr/bin/env python3
"""E1 read-off: does the fixed-focal reuse benefit depend on partner noise?

Input is the JSON written by scripts/run_e1_teammate_causal.py. For every
(checkpoint, focal set, eps) it pairs the per-episode returns of the two focal
branches, which share the env seed and the partner noise stream:

  Delta(eps) = R(focal offline, partners live+eps) - R(focal live, partners live+eps)

Reported quantities
  * per-checkpoint Delta(eps) with a paired bootstrap CI;
  * slope of Delta over eps, and Delta(0.3) - Delta(0), bootstrapped jointly over
    episode indices (episodes are common random numbers across eps);
  * the same quantities pooled over checkpoints;
  * a manipulation check: the live-branch return as a function of eps.

Pre-registered reading
  E1 holds if Delta(eps) is positive somewhere and moves with eps (change CI
  excludes 0) with the same sign in every checkpoint; a flat Delta means the
  reuse benefit does not depend on the teammate condition probed here.

Usage:
  python scripts/analyze_e1_teammate_causal.py \
    --input checkpoints/e1_e7_supplement/e1/*.json \
    --out checkpoints/e1_e7_supplement/e1_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def load_runs(paths: List[str]) -> Tuple[List[Dict], Dict]:
    runs: List[Dict] = []
    meta = {}
    for path in paths:
        with open(path) as handle:
            payload = json.load(handle)
        meta[os.path.basename(path)] = {
            "episodes": payload["episodes"],
            "eval_seed": payload["eval_seed"],
            "checkpoints": payload["checkpoints"],
        }
        runs.extend(payload["runs"])
    return runs, meta


def collect(runs: List[Dict]) -> Dict[Tuple[str, str], Dict[float, Dict[str, np.ndarray]]]:
    table: Dict[Tuple[str, str], Dict[float, Dict[str, np.ndarray]]] = defaultdict(dict)
    for row in runs:
        key = (row["checkpoint"], row["focal"])
        table[key].setdefault(float(row["eps"]), {})[row["branch"]] = np.asarray(
            row["returns"], dtype=np.float64
        )
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", nargs="+", required=True, help="E1 run JSONs or globs")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths: List[str] = []
    for pattern in args.input:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    runs, meta = load_runs(paths)
    table = collect(runs)
    if not table:
        raise SystemExit("no E1 runs found")

    checkpoints = sorted({ck for ck, _ in table})
    focals = sorted({f for _, f in table})
    rng = np.random.default_rng(7)
    payload: Dict = {"inputs": meta, "checkpoints": checkpoints, "focals": focals,
                     "per_checkpoint": {}, "pooled": {}, "pooled_change": {},
                     "verdict": []}

    # ---- per checkpoint / focal ----
    for focal in focals:
        print(f"\n=== focal {focal} ===")
        for ck in checkpoints:
            key = (ck, focal)
            if key not in table:
                continue
            eps_map = table[key]
            eps_list = sorted(eps_map)
            paired = [
                e for e in eps_list
                if "live" in eps_map[e] and "off" in eps_map[e]
                and eps_map[e]["live"].size == eps_map[e]["off"].size
            ]
            if not paired:
                continue
            n = min(eps_map[e]["live"].size for e in paired)
            diffs = np.stack([
                eps_map[e]["off"][:n] - eps_map[e]["live"][:n] for e in paired
            ])
            live_mean = np.array([eps_map[e]["live"][:n].mean() for e in paired])
            print(f"  {ck}: n={n} episodes per eps")
            print(f"    {'eps':>5s} {'live':>9s} {'off':>9s} {'Delta':>9s} {'95% CI':>20s}")
            rows = []
            for j, e in enumerate(paired):
                lo, hi = ci(diffs[j], n_boot=args.bootstrap, seed=1)
                print(f"    {e:5.2f} {live_mean[j]:9.3f} {eps_map[e]['off'][:n].mean():9.3f} "
                      f"{diffs[j].mean():+9.3f} [{lo:+7.3f},{hi:+7.3f}]")
                rows.append({"eps": e, "live_mean": float(live_mean[j]),
                             "off_mean": float(eps_map[e]["off"][:n].mean()),
                             "delta": float(diffs[j].mean()), "ci": [lo, hi]})
            # joint bootstrap over episode indices for slope and endpoint change
            if len(paired) >= 2:
                slopes = np.empty(args.bootstrap)
                changes = np.empty(args.bootstrap)
                x = np.asarray(paired, dtype=np.float64)
                for b in range(args.bootstrap):
                    idx = rng.integers(0, n, size=n)
                    d = diffs[:, idx].mean(axis=1)
                    slopes[b] = slope(x, d)
                    changes[b] = d[-1] - d[0]
                s_lo, s_hi = np.quantile(slopes, [0.025, 0.975])
                c_lo, c_hi = np.quantile(changes, [0.025, 0.975])
                s_pt = slope(x, diffs.mean(axis=1))
                c_pt = float(diffs[-1].mean() - diffs[0].mean())
                print(f"    slope {s_pt:+.3f} [{s_lo:+.3f},{s_hi:+.3f}]   "
                      f"Delta(last)-Delta(first) {c_pt:+.3f} [{c_lo:+.3f},{c_hi:+.3f}]")
                payload["per_checkpoint"].setdefault(focal, {})[ck] = {
                    "n": int(n), "rows": rows, "slope": s_pt,
                    "slope_ci": [float(s_lo), float(s_hi)],
                    "change": c_pt, "change_ci": [float(c_lo), float(c_hi)],
                }
            else:
                payload["per_checkpoint"].setdefault(focal, {})[ck] = {
                    "n": int(n), "rows": rows,
                }

    # ---- pooled over checkpoints ----
    print("\n=== pooled over checkpoints (equal weight) ===")
    for focal in focals:
        per_ck = payload["per_checkpoint"].get(focal, {})
        if len(per_ck) < 2:
            continue
        eps_list = sorted({r["eps"] for ck in per_ck.values() for r in ck["rows"]})
        rows = []
        boot = np.empty((args.bootstrap, len(eps_list)))
        for b in range(args.bootstrap):
            pick = rng.integers(0, len(per_ck), size=len(per_ck))
            vals = []
            for j, e in enumerate(eps_list):
                acc = []
                for pi in pick:
                    ck = list(per_ck)[pi]
                    match = [r for r in per_ck[ck]["rows"] if r["eps"] == e]
                    if match:
                        acc.append(match[0]["delta"])
                vals.append(float(np.mean(acc)) if acc else np.nan)
            boot[b] = vals
        for j, e in enumerate(eps_list):
            vals = [next(r for r in ck["rows"] if r["eps"] == e)["delta"]
                    for ck in per_ck.values() if any(r["eps"] == e for r in ck["rows"])]
            lo, hi = np.nanquantile(boot[:, j], [0.025, 0.975])
            same = int(np.sum(np.asarray(vals) > 0))
            print(f"  {focal:12s} eps={e:4.2f}  mean Delta {np.mean(vals):+7.3f} "
                  f"[{lo:+7.3f},{hi:+7.3f}]  positive {same}/{len(vals)}")
            rows.append({"eps": e, "delta": float(np.mean(vals)),
                         "ci": [float(lo), float(hi)],
                         "n_positive": same, "n": len(vals)})
        payload["pooled"][focal] = rows
        if len(rows) >= 2:
            # Paired over checkpoints: same resampled checkpoints at both ends.
            change_boot = boot[:, -1] - boot[:, 0]
            c_lo, c_hi = np.nanquantile(change_boot, [0.025, 0.975])
            payload["pooled_change"][focal] = {
                "change": rows[-1]["delta"] - rows[0]["delta"],
                "ci": [float(c_lo), float(c_hi)],
            }
            print(f"  {focal:12s} pooled change {rows[-1]['delta'] - rows[0]['delta']:+.3f} "
                  f"[{c_lo:+.3f},{c_hi:+.3f}]")

    # ---- pre-registered verdict ----
    print("\npre-registered reading:")
    for focal in focals:
        pooled = payload["pooled"].get(focal, [])
        if len(pooled) < 2:
            continue
        first, last = pooled[0], pooled[-1]
        pc = payload["pooled_change"].get(focal)
        sig_change = bool(pc) and (pc["ci"][0] > 0.0 or pc["ci"][1] < 0.0)
        slopes = [ck["slope"] for ck in payload["per_checkpoint"][focal].values()]
        n_neg = sum(1 for s in slopes if s < 0)
        same_slope = n_neg in (0, len(slopes))
        if sig_change and same_slope:
            reading = f"teammate-conditioned (all {len(slopes)} slopes same sign)"
        elif sig_change:
            reading = (f"pooled magnitude conditioned; direction heterogeneous "
                       f"({n_neg}/{len(slopes)} slopes negative)")
        else:
            reading = "no stable conditioning"
        line = (
            f"  {focal:12s} Delta goes {first['delta']:+.3f} -> {last['delta']:+.3f}; "
            f"change {pc['change']:+.3f} [{pc['ci'][0]:+.3f},{pc['ci'][1]:+.3f}]; "
            f"{reading}"
        )
        print(line)
        payload["verdict"].append(line)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
