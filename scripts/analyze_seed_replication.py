#!/usr/bin/env python3
"""Cross-seed summary for the Insight A and E replication batches.

Reads the per-run A/E JSONs written by ``run_pv_exp2a.py`` and
``run_replay_gain.py`` (seed 1 from checkpoints/extra_20261024, seeds 2/3 from
checkpoints/extra_seeds_20261024) and reports:

* A: per-arm transplant benefit per seed, the best single/pair configuration,
     its paired CI vs the all-live team, and cross-seed agreement of the
     configuration ranking.  This is the "collaboration combination" view:
     which subset of offline agents is best, and is it stable across seeds?
* E: per-(arm, version) replay-group gain U, excess over random, and cross-seed
     agreement of the group ranking.

Nothing is recomputed from raw episodes beyond the definitions already used by
``analyze_extra_runs.py``.

Usage:
  python scripts/analyze_seed_replication.py \
      --roots checkpoints/extra_20261024 checkpoints/extra_seeds_20261024 \
      --out checkpoints/extra_seeds_20261024/seed_replication_summary.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_extra_runs import BOOT_SEED, analyse_a, analyse_e  # noqa: E402

N_AGENTS = 5


def load_runs(roots, prefix):
    paths = []
    for root in roots:
        paths.extend(sorted(glob.glob(os.path.join(root, f"{prefix}*.json"))))
    paths = [p for p in paths if not p.endswith("index.json")]
    return paths


def rank_stability(table: dict[str, dict[int, float]]) -> dict:
    """Pairwise Spearman of configuration scores across seeds."""
    seeds = sorted(table)
    out = {}
    for sa, sb in itertools.combinations(seeds, 2):
        names = sorted(set(table[sa]) & set(table[sb]))
        if len(names) < 3:
            continue
        a = np.array([table[sa][n] for n in names])
        b = np.array([table[sb][n] for n in names])
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            out[f"{sa}->{sb}"] = None
            continue
        out[f"{sa}->{sb}"] = float(spearmanr(a, b).statistic)
    return out


def summarise_a(paths, rng):
    by_arm = defaultdict(list)
    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        seed = int(payload.get("seed", 1))
        arm = str(payload.get("arm", "A"))
        t = "1000000"
        if t not in payload.get("natural", {}):
            continue
        row = analyse_a(payload, arm, t, rng)
        row["seed"] = seed
        by_arm[arm].append(row)

    summary = {}
    for arm, rows in sorted(by_arm.items()):
        rows = sorted(rows, key=lambda r: r["seed"])
        configs = sorted(
            {name for r in rows for name in r.get("paired_vs_all", {})}
        )
        gaps = {
            r["seed"]: {
                name: r["returns"][name] - r["all_return"]
                for name in configs
                if name in r.get("returns", {})
            }
            for r in rows
        }
        table = {}
        for name in configs:
            vals = [gaps[r["seed"]][name] for r in rows if name in gaps[r["seed"]]]
            table[name] = {
                "mean_gap": float(np.mean(vals)) if vals else None,
                "per_seed": {
                    str(r["seed"]): float(gaps[r["seed"]][name])
                    for r in rows
                    if name in gaps[r["seed"]]
                },
                "n_seeds": len(vals),
                "n_positive": int(sum(v > 0 for v in vals)),
            }
        ranking = sorted(
            (name for name in table if table[name]["mean_gap"] is not None),
            key=lambda name: -table[name]["mean_gap"],
        )
        best_per_seed = [
            {
                "seed": r["seed"],
                "name": r["best_transplant"]["name"],
                "gap": r["best_transplant"]["gap_vs_all"],
                "paired": r.get("best_paired"),
            }
            for r in rows
        ]
        summary[arm] = {
            "seeds": [r["seed"] for r in rows],
            "all_return": {str(r["seed"]): r["all_return"] for r in rows},
            "all_return_mean": float(np.mean([r["all_return"] for r in rows])),
            "best_per_seed": best_per_seed,
            "best_gap_mean": float(np.mean([r["best_transplant"]["gap_vs_all"] for r in rows])),
            "best_gap_range": [
                float(min(r["best_transplant"]["gap_vs_all"] for r in rows)),
                float(max(r["best_transplant"]["gap_vs_all"] for r in rows)),
            ],
            "q_rho": {str(r["seed"]): r.get("all_spearman_rho") for r in rows},
            "q_bias_norm": {str(r["seed"]): r.get("all_bias_normalized") for r in rows},
            "config_gap": table,
            "config_ranking_by_mean_gap": ranking,
            "config_rank_spearman": rank_stability(gaps),
        }
    return summary


def summarise_e(paths, rng):
    grouped = defaultdict(list)
    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        seed = int(payload.get("seed", 1))
        arm = str(payload.get("arm", "A"))
        row = analyse_e(payload, os.path.basename(path)[:-5], rng)
        row["seed"] = seed
        for t, version in row["versions"].items():
            key = (arm, str(t), int(row.get("updates") or 0), int(row.get("eval_episodes") or 0))
            grouped[key].append({"seed": seed, "version": version, "file": row["file"]})

    summary = {}
    for (arm, t, updates, eval_eps), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["seed"])
        groups = sorted(rows[0]["version"]["U"])
        table = {}
        for g in groups:
            vals = [r["version"]["U"][g] for r in rows if g in r["version"]["U"]]
            random_vals = [
                r["version"]["U"].get("random", float("nan")) for r in rows
            ]
            excess = [
                r["version"]["U"][g] - r["version"]["U"]["random"]
                for r in rows
                if g in r["version"]["U"] and "random" in r["version"]["U"]
            ]
            table[g] = {
                "mean_U": float(np.mean(vals)),
                "per_seed_U": {str(r["seed"]): r["version"]["U"][g] for r in rows if g in r["version"]["U"]},
                "mean_excess_over_random": float(np.mean(excess)) if excess else None,
                "n_seeds_not_below_random": int(sum(v >= 0 for v in excess)),
                "n_seeds": len(vals),
            }
        ranking = sorted(groups, key=lambda g: -table[g]["mean_U"])
        best_per_seed = [
            {
                "seed": r["seed"],
                "best_group": r["version"]["best_group"],
                "best_excess": r["version"]["best_excess_over_random"],
                "theta0_return": r["version"]["theta0_return"],
            }
            for r in rows
        ]
        u_table = {r["seed"]: r["version"]["U"] for r in rows}
        stability = rank_stability(u_table)
        summary[f"{arm}_t{t}_u{updates}_e{eval_eps}"] = {
            "arm": arm,
            "t": int(t),
            "updates": updates,
            "eval_episodes": eval_eps,
            "seeds": [r["seed"] for r in rows],
            "theta0_return": {str(r["seed"]): r["version"]["theta0_return"] for r in rows},
            "best_per_seed": best_per_seed,
            "groups": table,
            "group_ranking_by_mean_U": ranking,
            "group_rank_spearman": stability,
        }
    return summary


def print_a(summary):
    print("\n=== Insight A: transplant benefit by arm and seed ===")
    for arm, row in sorted(summary.items()):
        print(f"\narm {arm}: seeds={row['seeds']} all_return mean={row['all_return_mean']:.3f}")
        print("  seed  all_return  best transplant (gap)          best paired 95% CI")
        for b in row["best_per_seed"]:
            ci = b.get("paired") or {}
            lo, hi = (ci.get("ci") or [None, None])[:2]
            cis = "n/a" if lo is None else f"[{lo:+.3f},{hi:+.3f}]"
            print(
                f"  {b['seed']:>4}  {row['all_return'][str(b['seed'])]:9.3f}  "
                f"{b['name']:<28} {b['gap']:+.3f}  {cis}"
            )
        print("  config gap vs all-live (mean over seeds; +ve = transplant helps):")
        for name in row["config_ranking_by_mean_gap"]:
            cell = row["config_gap"][name]
            print(
                f"    {name:<14} mean={cell['mean_gap']:+.3f} "
                f"positive {cell['n_positive']}/{cell['n_seeds']} "
                f"per_seed={cell['per_seed']}"
            )
        print(f"  config rank Spearman across seeds: {row['config_rank_spearman']}")


def print_e(summary):
    print("\n=== Insight E: replay-group gain U by arm/version and seed ===")
    for key, row in sorted(summary.items()):
        print(
            f"\n{key}: seeds={row['seeds']} theta0={row['theta0_return']}"
        )
        print("  group     mean_U   excess/random  not-below-random  per_seed_U")
        for g in row["group_ranking_by_mean_U"]:
            cell = row["groups"][g]
            excess = cell["mean_excess_over_random"]
            excess_s = "n/a" if excess is None else f"{excess:+.3f}"
            print(
                f"  {g:<8} {cell['mean_U']:+8.3f}  {excess_s:>13}  "
                f"{cell['n_seeds_not_below_random']}/{cell['n_seeds']}  {cell['per_seed_U']}"
            )
        print(f"  best per seed: {[(b['seed'], b['best_group'], round(b['best_excess'], 3)) for b in row['best_per_seed']]}")
        print(f"  group rank Spearman across seeds: {row['group_rank_spearman']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--roots",
        nargs="+",
        default=["checkpoints/extra_20261024", "checkpoints/extra_seeds_20261024"],
    )
    ap.add_argument("--out", default="checkpoints/extra_seeds_20261024/seed_replication_summary.json")
    args = ap.parse_args()

    rng = np.random.default_rng(BOOT_SEED)
    a_paths = load_runs(args.roots, "A_")
    e_paths = load_runs(args.roots, "E_")
    print(f"A files: {len(a_paths)}  E files: {len(e_paths)}")

    summary = {
        "A": summarise_a(a_paths, rng),
        "E": summarise_e(e_paths, rng),
    }
    print_a(summary["A"])
    print_e(summary["E"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
