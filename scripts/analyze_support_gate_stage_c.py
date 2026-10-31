#!/usr/bin/env python3
"""Stage C analysis: did the teammate-support gate repair the offline value memory?

Reads the per-arm online.jsonl files produced by
scripts/run_support_gate_stage_c.sh and reports the comparisons the
pre-registered table in documentation/experiment_reports/support_gate_stage_a_premise.md
needs:

    tcvm vs ovm   does gating the memory beat the published schedule
    tcvm vs anti  does it matter WHERE the memory is kept, or only how much
    tcvm vs none  is the memory worth keeping at all

Arms share the start checkpoint, the budget and the evaluation seed, so the
comparison is paired at every evaluation point. The headline number is the
endpoint; the tail mean over the last `--tail` points is reported alongside it
because a single 64-episode evaluation point is noisy. Seeds are paired too, so
the across-seed statistic is the mean of per-seed differences and the sign
consistency across seeds is reported rather than hidden inside an average.

Usage:
  python scripts/analyze_support_gate_stage_c.py --root checkpoints/support_gate_stage_c/5m_vs_6m
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

ARMS = ["none", "ovm", "tcvm", "anti"]
WIN_KEYS = ("evaluation/win_rate", "evaluation/won_mean", "evaluation/battle_won_mean")
RET_KEYS = (
    "evaluation/mean_episode_return",
    "evaluation/episode_return_mean",
    "evaluation/return_mean",
)


def pick_key(rows: List[Dict], candidates: Tuple[str, ...]) -> Optional[str]:
    for key in candidates:
        if any(key in row for row in rows):
            return key
    return None


def load_arm(path: str) -> List[Dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def curve(rows: List[Dict], key: str) -> Tuple[np.ndarray, np.ndarray]:
    pts = [(float(r["t_env"]), float(r[key])) for r in rows if key in r and "t_env" in r]
    pts.sort()
    if not pts:
        return np.array([]), np.array([])
    t = np.array([p[0] for p in pts])
    v = np.array([p[1] for p in pts])
    return t, v


def align(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]], tol: float = 5000.0
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Common evaluation grid across arms, so every comparison is paired.

    Arms share one eval schedule but each t_env is rounded to an episode
    boundary, so exact t_env values never intersect past 0. Pair by eval index
    and assert the paired steps agree to within `tol`.
    """
    n = min(t.size for t, _ in curves.values())
    grid = np.mean([t[:n] for t, _ in curves.values()], axis=0)
    drift = max(float(np.abs(t[:n] - grid).max()) for t, _ in curves.values()) if n else 0.0
    if drift > tol:
        raise SystemExit(f"eval grids diverge by {drift:.0f} env steps (> {tol:.0f}); not paired")
    return grid, {arm: v[:n] for arm, (_, v) in curves.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="checkpoints/support_gate_stage_c/5m_vs_6m")
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
        raise SystemExit(f"no stage C runs under {args.root}")

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
            print(f"    {'arm':6s} {'endpoint':>10s} {'tail mean':>10s} {'best':>8s}")
            for arm in present:
                if arm not in aligned:
                    continue
                v = aligned[arm]
                print(
                    f"    {arm:6s} {v[-1]:10.3f} {v[-tail:].mean():10.3f} {v.max():8.3f}"
                )
                seed_out.setdefault(label, {})[arm] = {
                    "endpoint": float(v[-1]),
                    "tail_mean": float(v[-tail:].mean()),
                    "best": float(v.max()),
                    "curve": [[float(g), float(x)] for g, x in zip(grid, v)],
                }
            if "tcvm" in aligned:
                for other in ("ovm", "anti", "none"):
                    if other not in aligned:
                        continue
                    d_end = aligned["tcvm"][-1] - aligned[other][-1]
                    d_tail = aligned["tcvm"][-tail:].mean() - aligned[other][-tail:].mean()
                    print(
                        f"    tcvm - {other:5s} endpoint {d_end:+8.3f}   tail {d_tail:+8.3f}"
                    )
        payload["seeds"][str(seed)] = seed_out

        gate = [
            r["support/gate_mean"]
            for r in arms.get("tcvm", [])
            if "support/gate_mean" in r
        ]
        theta = next(
            (r["support/theta"] for r in arms.get("tcvm", []) if "support/theta" in r), None
        )
        if gate:
            print(
                f"  tcvm gate: theta={theta:.4f} mean={np.mean(gate):.3f} "
                f"min={np.min(gate):.3f} max={np.max(gate):.3f}"
            )
            payload["seeds"][str(seed)]["gate"] = {
                "theta": theta,
                "mean": float(np.mean(gate)),
                "min": float(np.min(gate)),
                "max": float(np.max(gate)),
            }

    print("\n=== across seeds (paired per seed) ===")
    for label, _ in metrics:
        for other in ("ovm", "anti", "none"):
            for stat in ("endpoint", "tail_mean"):
                diffs = []
                for seed, seed_out in payload["seeds"].items():
                    block = seed_out.get(label, {})
                    if "tcvm" in block and other in block:
                        diffs.append(block["tcvm"][stat] - block[other][stat])
                if not diffs:
                    continue
                arr = np.array(diffs)
                same = int(np.sum(arr > 0))
                print(
                    f"  {label:6s} {stat:9s} tcvm - {other:5s}: mean {arr.mean():+7.3f}  "
                    f"per-seed {np.round(arr, 3).tolist()}  positive {same}/{arr.size}"
                )
                payload["paired"][f"{label}/{stat}/tcvm-{other}"] = {
                    "mean": float(arr.mean()),
                    "per_seed": [float(x) for x in arr],
                    "n_positive": same,
                    "n": int(arr.size),
                }

    verdict = []
    for stat in ("endpoint", "tail_mean"):
        key_ovm = payload["paired"].get(f"return/{stat}/tcvm-ovm")
        key_anti = payload["paired"].get(f"return/{stat}/tcvm-anti")
        if key_ovm and key_anti:
            verdict.append(
                f"{stat}: tcvm-ovm {key_ovm['mean']:+.3f} ({key_ovm['n_positive']}/{key_ovm['n']}), "
                f"tcvm-anti {key_anti['mean']:+.3f} ({key_anti['n_positive']}/{key_anti['n']})"
            )
    if verdict:
        print("\npre-registered read-off (return):")
        for line in verdict:
            print(f"  {line}")
        print(
            "  mechanism holds only if tcvm beats BOTH ovm and anti with a consistent sign"
        )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
