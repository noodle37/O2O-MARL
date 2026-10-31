#!/usr/bin/env python3
"""Stage G read-off: do offline replay mixing and online CQL stack?

2x2 cells (see scripts/run_stage_g_stacking.sh):
    none = mixing 0.0, cql 0
    mix  = mixing 0.1, cql 0
    cql  = mixing 0.0, cql 50
    both = mixing 0.1, cql 50

`none` and `cql` are read from the Stage D logs (`nocql` / `uniform`), which used
the same checkpoint, recipe, seed set, budget and evaluation seed. Logs are
de-duplicated by t_env (the online loop emits t=0 twice, and Stage F inserted
extra copies later), then paired by evaluation index.

Reported:
  * per-seed tail-mean and endpoint for every arm;
  * mix - none, cql - none, both - none, both - mix;
  * interaction (both - mix) - (cql - none), the super-additivity test;
  * per-seed signs, because three seeds cannot support a CI across seeds.

Pre-registered reading (return, tail mean):
  both - none > 0 and 3/3 positive        -> the stack helps overall
  interaction > 0 and 3/3 positive        -> benefits are super-additive
  interaction < 0 and 3/3 negative        -> benefits overlap/substitute
  otherwise                               -> no stable interaction

Usage:
  python scripts/analyze_stage_g_stacking.py \
    --root checkpoints/stage_g_stacking/5m_vs_6m \
    --base-root checkpoints/support_cql_stage_d/5m_vs_6m \
    --tail 3 --out checkpoints/stage_g_stacking/5m_vs_6m/summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_support_gate_stage_c import (  # noqa: E402
    RET_KEYS,
    WIN_KEYS,
    load_arm,
    pick_key,
)

ARMS = ["none", "mix", "cql", "both"]
SOURCE = {"none": "nocql", "cql": "uniform", "mix": "mix", "both": "both"}


def dedupe(rows: List[Dict], key: str) -> List[Dict]:
    """Keep the last row per t_env: the loop evaluates t=0 twice."""
    by_t: Dict[float, Dict] = {}
    for row in rows:
        if key in row and "t_env" in row:
            by_t[float(row["t_env"])] = row
    return [by_t[t] for t in sorted(by_t)]


def curve(rows: List[Dict], key: str) -> Tuple[np.ndarray, np.ndarray]:
    pts = [(float(r["t_env"]), float(r[key])) for r in rows]
    return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))


def align(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], tol: float = 5000.0):
    n = min(t.size for t, _ in curves.values())
    grid = np.mean([t[:n] for t, _ in curves.values()], axis=0)
    drift = max(float(np.abs(t[:n] - grid).max()) for t, _ in curves.values())
    if drift > tol:
        raise SystemExit(f"eval grids diverge by {drift:.0f} env steps (> {tol:.0f})")
    return grid, {a: v[:n] for a, (_, v) in curves.items()}


def discover(root: str, base_root: str) -> Dict[int, Dict[str, str]]:
    runs: Dict[int, Dict[str, str]] = {}
    for arm in ARMS:
        src = SOURCE[arm]
        search = os.path.join(root if arm in ("mix", "both") else base_root,
                              f"{src}_seed*", "online.jsonl")
        for path in sorted(glob.glob(search)):
            name = os.path.basename(os.path.dirname(path))
            seed = int(name.partition("_seed")[2])
            runs.setdefault(seed, {})[arm] = path
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="checkpoints/stage_g_stacking/5m_vs_6m")
    ap.add_argument("--base-root", default="checkpoints/support_cql_stage_d/5m_vs_6m")
    ap.add_argument("--tail", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    runs = discover(args.root, args.base_root)
    if not runs:
        raise SystemExit("no stage G / stage D runs found")

    rows_by_seed = {
        seed: {arm: dedupe(load_arm(path), "evaluation/mean_episode_return")
               for arm, path in arms.items()}
        for seed, arms in runs.items()
    }
    all_rows = [r for arms in rows_by_seed.values() for rows in arms.values() for r in rows]
    win_key = pick_key(all_rows, WIN_KEYS)
    ret_key = pick_key(all_rows, RET_KEYS)
    if ret_key is None:
        raise SystemExit("no return metric in logs")
    metrics = [("return", ret_key)] + ([("win", win_key)] if win_key else [])

    payload: Dict = {"root": args.root, "base_root": args.base_root,
                     "tail": args.tail, "seeds": {}, "paired": {},
                     "interaction": {}, "verdict": []}

    for seed in sorted(rows_by_seed):
        arms = rows_by_seed[seed]
        present = [a for a in ARMS if a in arms]
        print(f"\n=== seed {seed}  arms: {', '.join(present)} ===")
        seed_out: Dict = {}
        for label, key in metrics:
            curves = {a: curve(arms[a], key) for a in present if arms[a]}
            curves = {a: c for a, c in curves.items() if c[0].size}
            if not curves:
                continue
            grid, aligned = align(curves)
            tail = max(1, min(args.tail, grid.size))
            print(f"  {label}: {grid.size} paired points, "
                  f"t_env {grid[0]:.0f}..{grid[-1]:.0f}")
            print(f"    {'arm':6s} {'endpoint':>10s} {'tail mean':>10s} {'best':>8s}")
            for arm in present:
                if arm not in aligned:
                    continue
                v = aligned[arm]
                print(f"    {arm:6s} {v[-1]:10.3f} {v[-tail:].mean():10.3f} {v.max():8.3f}")
                seed_out.setdefault(label, {})[arm] = {
                    "endpoint": float(v[-1]),
                    "tail_mean": float(v[-tail:].mean()),
                    "best": float(v.max()),
                }
        payload["seeds"][str(seed)] = seed_out

    pairs = [("mix", "none"), ("cql", "none"), ("both", "none"), ("both", "mix")]
    print("\n=== across seeds (paired per seed) ===")
    for label, _ in metrics:
        for a, b in pairs:
            for stat in ("endpoint", "tail_mean"):
                diffs = []
                for seed_out in payload["seeds"].values():
                    block = seed_out.get(label, {})
                    if a in block and b in block:
                        diffs.append(block[a][stat] - block[b][stat])
                if not diffs:
                    continue
                arr = np.asarray(diffs)
                print(f"  {label:6s} {stat:9s} {a:4s} - {b:4s}: mean {arr.mean():+7.3f}  "
                      f"per-seed {np.round(arr, 3).tolist()}  "
                      f"positive {int((arr > 0).sum())}/{arr.size}")
                payload["paired"][f"{label}/{stat}/{a}-{b}"] = {
                    "mean": float(arr.mean()),
                    "per_seed": [float(x) for x in arr],
                    "n_positive": int((arr > 0).sum()),
                    "n": int(arr.size),
                }

    print("\n=== interaction: (both - mix) - (cql - none) ===")
    for label, _ in metrics:
        cell = {}
        for stat in ("endpoint", "tail_mean"):
            inter = []
            for seed_out in payload["seeds"].values():
                block = seed_out.get(label, {})
                if all(a in block for a in ARMS):
                    inter.append(
                        block["both"][stat] - block["mix"][stat]
                        - block["cql"][stat] + block["none"][stat]
                    )
            if not inter:
                continue
            arr = np.asarray(inter)
            cell[stat] = {
                "mean": float(arr.mean()),
                "per_seed": [float(x) for x in arr],
                "n_positive": int((arr > 0).sum()),
                "n": int(arr.size),
            }
            print(f"  {label:6s} {stat:9s}: mean {arr.mean():+7.3f}  "
                  f"per-seed {np.round(arr, 3).tolist()}  "
                  f"positive {int((arr > 0).sum())}/{arr.size}")
        payload["interaction"][label] = cell

    print("\npre-registered reading (return, tail mean):")
    inter = payload["interaction"].get("return", {}).get("tail_mean")
    both_none = payload["paired"].get("return/tail_mean/both-none")
    if inter and both_none:
        if both_none["mean"] > 0 and both_none["n_positive"] == both_none["n"]:
            overall = "stack helps overall"
        else:
            overall = "stack does not help overall"
        if inter["mean"] > 0 and inter["n_positive"] == inter["n"]:
            shape = "super-additive"
        elif inter["mean"] < 0 and inter["n_positive"] == 0:
            shape = "sub-additive (benefits overlap)"
        else:
            shape = "no stable interaction"
        line = (
            f"  both-none {both_none['mean']:+.3f} "
            f"({both_none['n_positive']}/{both_none['n']}); "
            f"interaction {inter['mean']:+.3f} "
            f"({inter['n_positive']}/{inter['n']}) -> {overall}; {shape}"
        )
        print(line)
        payload["verdict"].append(line)
    else:
        print("  incomplete cells; cannot read the interaction")
        payload["verdict"].append("incomplete cells")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    main()
