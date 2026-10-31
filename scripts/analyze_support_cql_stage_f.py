#!/usr/bin/env python3
"""Stage F read-off: mass-normalized support-conditioned CQL placement.

Same design as Stage D, but gated/anti arms normalize per-batch CQL weight to
mean == 1, exactly matching uniform in total conservative mass. Any benefit can
only come from WHERE the penalty is placed.

Pre-registered read-off:
  gated_norm > uniform AND gated_norm > anti_norm, same sign across seeds -> mechanism holds
  gated_norm ~ uniform  -> placement does not matter, nothing beyond a tuned global
                           online_cql_weight
  gated_norm ~ anti_norm -> placement matters but the direction is wrong
  uniform ~ nocql       -> the conservative term is not the reason any arm moves

Usage:
  python scripts/analyze_support_cql_stage_f.py \
    --root checkpoints/support_cql_stage_f/5m_vs_6m --tail 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_support_gate_stage_c import (  # noqa: E402
    RET_KEYS,
    WIN_KEYS,
    align,
    curve,
    load_arm,
    pick_key,
)

ARMS = ["nocql", "uniform", "gated_norm", "anti_norm"]
FOCUS = "gated_norm"
OTHERS = ["uniform", "anti_norm", "nocql"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="checkpoints/support_cql_stage_f/5m_vs_6m")
    parser.add_argument("--tail", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    runs: Dict[int, Dict[str, List[Dict]]] = {}
    for path in sorted(glob.glob(os.path.join(args.root, "*_seed*", "online.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        arm, _, seed_part = name.partition("_seed")
        if arm not in ARMS:
            continue
        runs.setdefault(int(seed_part), {})[arm] = load_arm(path)
    if not runs:
        raise SystemExit(f"no stage F runs under {args.root}")

    all_rows = [r for seed in runs.values() for rows in seed.values() for r in rows]
    win_key = pick_key(all_rows, WIN_KEYS)
    ret_key = pick_key(all_rows, RET_KEYS)
    if ret_key is None:
        raise SystemExit("no return metric found in the logs")
    metrics = [("return", ret_key)] + ([("win", win_key)] if win_key else [])
    payload: Dict = {"root": args.root, "tail": args.tail, "seeds": {}, "paired": {}}

    for seed in sorted(runs):
        arms = runs[seed]
        present = [a for a in ARMS if a in arms]
        print(f"\n=== seed {seed}  arms: {', '.join(present)} ===")
        seed_out: Dict = {}
        for label, key in metrics:
            curves = {a: curve(arms[a], key) for a in present}
            curves = {a: c for a, c in curves.items() if c[0].size}
            if not curves:
                continue
            grid, aligned = align(curves)
            if grid.size == 0:
                print(f"  {label}: no common evaluation grid yet")
                continue
            tail = max(1, min(args.tail, grid.size))
            print(f"  {label}: {grid.size} paired points, t_env {grid[0]:.0f}..{grid[-1]:.0f}")
            print(f"    {'arm':10s} {'endpoint':>10s} {'tail mean':>10s} {'best':>8s}")
            for arm in present:
                if arm not in aligned:
                    continue
                v = aligned[arm]
                print(f"    {arm:10s} {v[-1]:10.3f} {v[-tail:].mean():10.3f} {v.max():8.3f}")
                seed_out.setdefault(label, {})[arm] = {
                    "endpoint": float(v[-1]),
                    "tail_mean": float(v[-tail:].mean()),
                    "best": float(v.max()),
                    "curve": [[float(g), float(x)] for g, x in zip(grid, v)],
                }
            if FOCUS in aligned:
                for other in OTHERS:
                    if other not in aligned:
                        continue
                    d_end = aligned[FOCUS][-1] - aligned[other][-1]
                    d_tail = aligned[FOCUS][-tail:].mean() - aligned[other][-tail:].mean()
                    print(f"    {FOCUS} - {other:10s} endpoint {d_end:+8.3f}   tail {d_tail:+8.3f}")
        payload["seeds"][str(seed)] = seed_out

        # Audit the gate actually applied: weight mean should be ~1 for normalized arms.
        for arm in ("gated_norm", "anti_norm"):
            w = [
                r["support/cql_weight_mean_interval_mean"]
                for r in arms.get(arm, [])
                if "support/cql_weight_mean_interval_mean" in r
            ]
            if w:
                print(
                    f"  {arm} cql weight: mean={np.mean(w):.3f} "
                    f"min={np.min(w):.3f} max={np.max(w):.3f}"
                )
                payload["seeds"][str(seed)].setdefault("gate", {})[arm] = {
                    "weight_mean": float(np.mean(w)),
                    "weight_min": float(np.min(w)),
                    "weight_max": float(np.max(w)),
                }

    print("\n=== across seeds (paired per seed) ===")
    for label, _ in metrics:
        for other in OTHERS:
            for stat in ("endpoint", "tail_mean"):
                diffs = []
                for seed_out in payload["seeds"].values():
                    block = seed_out.get(label, {})
                    if FOCUS in block and other in block:
                        diffs.append(block[FOCUS][stat] - block[other][stat])
                if not diffs:
                    continue
                arr = np.array(diffs)
                same = int(np.sum(arr > 0))
                print(
                    f"  {label:6s} {stat:9s} {FOCUS} - {other:10s}: mean {arr.mean():+7.3f}  "
                    f"per-seed {np.round(arr, 3).tolist()}  positive {same}/{arr.size}"
                )
                payload["paired"][f"{label}/{stat}/{FOCUS}-{other}"] = {
                    "mean": float(arr.mean()),
                    "per_seed": [float(x) for x in arr],
                    "n_positive": same,
                    "n": int(arr.size),
                }

    # Reference contrast: does the conservative term itself work?
    for stat in ("endpoint", "tail_mean"):
        diffs = []
        for seed_out in payload["seeds"].values():
            block = seed_out.get("return", {})
            if "uniform" in block and "nocql" in block:
                diffs.append(block["uniform"][stat] - block["nocql"][stat])
        if diffs:
            arr = np.array(diffs)
            print(
                f"  return {stat:9s} uniform - nocql   : mean {arr.mean():+7.3f}  "
                f"per-seed {np.round(arr, 3).tolist()}  positive {int(np.sum(arr > 0))}/{arr.size}"
            )
            payload["paired"][f"return/{stat}/uniform-nocql"] = {
                "mean": float(arr.mean()),
                "per_seed": [float(x) for x in arr],
                "n_positive": int(np.sum(arr > 0)),
                "n": int(arr.size),
            }

    print("\npre-registered read-off (return, tail_mean):")
    verdict = []
    for stat in ("endpoint", "tail_mean"):
        u = payload["paired"].get(f"return/{stat}/{FOCUS}-uniform")
        a = payload["paired"].get(f"return/{stat}/{FOCUS}-anti_norm")
        if not u or not a:
            verdict.append(f"  {stat}: incomplete (need gated_norm, uniform, anti_norm on every seed)")
            continue
        held = u["mean"] > 0 and a["mean"] > 0 and u["n_positive"] == u["n"] and a["n_positive"] == a["n"]
        verdict.append(
            f"  {stat}: gated_norm-uniform {u['mean']:+.3f} ({u['n_positive']}/{u['n']}), "
            f"gated_norm-anti_norm {a['mean']:+.3f} ({a['n_positive']}/{a['n']}) -> "
            f"{'MECHANISM HOLDS' if held else 'does not meet the pre-registered bar'}"
        )
    for line in verdict:
        print(line)
    payload["verdict"] = verdict

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
