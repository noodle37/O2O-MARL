#!/usr/bin/env python3
"""Summarise the Insight E replay-selection experiment.

Reads one or more run files (different update budgets) and prints, per policy
version, the short-horizon gain U_v(j) = J(theta_j^+) - J(theta_0) of each
experience group, the excess over random, and the cross-version stability of the
group ranking.  Also writes a compact JSON next to the inputs.

Usage:
  python scripts/analyze_insight_e.py \
      --runs checkpoints/replay_gain/5m_vs_6m.json \
             checkpoints/replay_gain/5m_vs_6m_u100.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    combined = {}
    for path in args.runs:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        budget = payload.get("updates")
        combined[str(budget)] = payload
        print(f"loaded {path} (updates={budget})")

    summary = {}
    for budget, payload in sorted(combined.items(), key=lambda kv: int(kv[0])):
        print(f"\n=== updates per group: {budget} ===")
        budget_summary = {}
        for t, version in sorted(payload["versions"].items(), key=lambda kv: int(kv[0])):
            groups = version["groups"]
            names = list(groups)
            us = np.array([groups[n]["U_return"] for n in names])
            wins = np.array([groups[n]["U_win"] for n in names])
            base = version["theta0"]
            random_u = groups["random"]["U_return"] if "random" in groups else np.nan
            print(
                f"\nt={t}  theta0 return={base['return']:.3f} win={base['win_rate']:.3f}"
            )
            print("  group     U_return   U_win   eval_return   excess_over_random")
            for n, u, w in zip(names, us, wins):
                print(
                    f"  {n:<8} {u:+8.3f}  {w:+.3f}  {groups[n]['eval']['return']:8.3f}"
                    f"  {u - random_u:+8.3f}"
                )
            e_index = names.index("random") if "random" in names else 0
            excess = us - us[e_index]
            budget_summary[t] = {
                "theta0": base,
                "u_return": {n: float(x) for n, x in zip(names, us)},
                "u_win": {n: float(x) for n, x in zip(names, wins)},
                "excess_over_random": {n: float(x) for n, x in zip(names, excess)},
                "rank_by_u": [n for n, _ in sorted(zip(names, us), key=lambda p: -p[1])],
            }
        times = list(budget_summary)
        stability = {}
        for ta, tb in itertools.combinations(times, 2):
            names = list(budget_summary[ta]["u_return"])
            ua = np.array([budget_summary[ta]["u_return"][n] for n in names])
            ub = np.array([budget_summary[tb]["u_return"][n] for n in names])
            stability[f"{ta}->{tb}"] = float(spearmanr(ua, ub).statistic)
        budget_summary["cross_version_spearman"] = stability
        if stability:
            print("\n  cross-version U Spearman: " + " ".join(
                f"{k}={v:+.3f}" for k, v in stability.items()
            ))
        summary[budget] = budget_summary

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.runs[0])), "5m_vs_6m_analysis.json"
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
