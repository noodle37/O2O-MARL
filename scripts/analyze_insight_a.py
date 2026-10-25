#!/usr/bin/env python3
"""Merge the Insight A shards and derive per-agent / per-pair effects.

For a policy version t (arm A's checkpoint after t environment steps), the
factorial evaluation gives J(config) for the live team and for every single and
pair of agents replaced by the frozen offline parameters:

    g_i(t)  = J(all@t) - J(i@off, rest@t)              (harm from freezing i)
    I_ij(t) = J(off_i, off_j | rest@t) - J(off_i | rest@t)
              - J(off_j | rest@t) + J(all@t)           (pair interaction)

Positive g_i means the teammate shift of agent i matters for team performance;
positive I_ij means the two freezes are sub-additive (they overlap), negative
means super-additive (they compound).  The script also reports the best
single/pair transplant and the version-to-version stability of the rankings.

Usage:
  python scripts/analyze_insight_a.py \
      --inputs checkpoints/pv_exp2a/5m_vs_6m/seed1_natural_shard1.json \
               checkpoints/pv_exp2a/5m_vs_6m/seed1_natural_shard2.json \
      --random checkpoints/pv_exp2a/5m_vs_6m/seed1_random.json \
      --out checkpoints/pv_exp2a/5m_vs_6m/seed1_analysis.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr


def load_natural(paths: List[str]) -> Dict[str, Dict]:
    merged: Dict[str, Dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        overlap = set(merged) & set(payload.get("natural", {}))
        if overlap:
            # Parallel shards may recompute a version; keep the later input.
            print(f"[merge] {os.path.basename(path)} replaces versions {sorted(overlap)}")
        merged.update(payload.get("natural", {}))
    return dict(sorted(merged.items(), key=lambda kv: int(kv[0])))


def version_effects(configs: Dict[str, Dict], n_agents: int):
    base = configs["all"]["mean_return"]
    g = np.array([base - configs[f"single_{i}"]["mean_return"] for i in range(n_agents)])
    pairs = list(itertools.combinations(range(n_agents), 2))
    I = np.array(
        [
            configs[f"pair_{i}_{j}"]["mean_return"]
            - configs[f"single_{i}"]["mean_return"]
            - configs[f"single_{j}"]["mean_return"]
            + base
            for i, j in pairs
        ]
    )
    names = [name for name in configs if name != "all"]
    returns = {name: configs[name]["mean_return"] for name in names}
    best_name = max(returns, key=returns.get)
    worst_name = min(returns, key=returns.get)
    return {
        "all_return": base,
        "all_win": configs["all"]["win_rate"],
        "g": g,
        "pairs": [list(p) for p in pairs],
        "I": I,
        "best_transplant": {
            "name": best_name,
            "return": returns[best_name],
            "gap_vs_all": returns[best_name] - base,
        },
        "worst_transplant": {"name": worst_name, "return": returns[worst_name]},
        "spread": returns[best_name] - returns[worst_name],
        "mean_team_deviation_all": configs["all"]["mean_team_deviation"],
        "q_off_spearman_all": configs["all"].get("spearman_rho"),
        "q_off_bias_norm_all": configs["all"].get("bias_normalized_mean"),
        "returns": returns,
    }


def stable_spearman(a, b):
    # t=0 is an exact null (g=I=0); a constant vector has no rank order.
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    return float(spearmanr(a, b).statistic)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--random", default=None)
    ap.add_argument("--n-agents", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    natural = load_natural(args.inputs)
    versions = {
        t: version_effects(configs, args.n_agents) for t, configs in natural.items()
    }
    times = sorted(versions, key=int)

    print("t          J(all)  win   best transplant               gap    g_i")
    for t in times:
        v = versions[t]
        best = v["best_transplant"]
        g_str = " ".join(f"{x:+.3f}" for x in v["g"])
        print(
            f"{t:>8}  {v['all_return']:6.3f}  {v['all_win']:.2f}  "
            f"{best['name']:<27} {best['gap_vs_all']:+.3f}  {g_str}"
        )

    print("\nPair interaction I_ij (positive = overlapping/sub-additive harm)")
    for t in times:
        v = versions[t]
        values = " ".join(
            f"{i}{j}:{val:+.3f}" for (i, j), val in zip(v["pairs"], v["I"])
        )
        print(f"  t={t:>8}: {values}")

    # Version-to-version stability of the two derived rankings.
    stability = {}
    for ta, tb in itertools.combinations(times, 2):
        key = f"{ta}->{tb}"
        stability[key] = {
            "g_spearman": stable_spearman(versions[ta]["g"], versions[tb]["g"]),
            "I_spearman": stable_spearman(versions[ta]["I"], versions[tb]["I"]),
        }
    print("\nVersion stability (Spearman; null = constant vector, e.g. t=0):")
    for key, val in stability.items():
        g_val, i_val = val["g_spearman"], val["I_spearman"]
        g_str = "null" if g_val is None else f"{g_val:+.3f}"
        i_str = "null" if i_val is None else f"{i_val:+.3f}"
        print(f"  {key}: g={g_str} I={i_str}")

    payload = {
        "versions": {
            t: {**v, "g": v["g"].tolist(), "I": v["I"].tolist()} for t, v in versions.items()
        },
        "version_stability": stability,
    }
    if args.random:
        with open(args.random, encoding="utf-8") as handle:
            payload["random_shift"] = json.load(handle).get("random", {})
        print("\nRandom teammate shift:")
        for eps, row in payload["random_shift"].items():
            print(
                f"  eps={eps:>4}: return={row['mean_return']:.3f} win={row['win_rate']:.2f} "
                f"D={row['mean_team_deviation']:.2f} rho={row.get('spearman_rho', float('nan')):+.3f} "
                f"bias_norm={row.get('bias_normalized_mean', float('nan')):+.3f}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
