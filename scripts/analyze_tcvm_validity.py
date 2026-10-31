#!/usr/bin/env python3
"""Stage A-2: is the validity of offline ACTION PREFERENCES teammate-conditioned?

Stage A (scripts/analyze_tcvm_gate.py) showed the offline value *level* is
unusable after the earliest phase: both the online target and the frozen offline
Q_tot overestimate the realised return, so a value floor has no headroom (the
oracle per-step gate buys ~0.0). What may survive the level bias is the
offline network's *ordering* of actions, which is exactly "which offline
cooperative knowledge is still valid".

This measures that directly, per agent. For agent i at step t:

    dev_i    = max_a Qoff_i(a) - Qoff_i(a_exec)   >= 0, deviation from the
                                                   old team's preference
    agree_i  = mean_{j != i} 1[a_exec_j == argmax Qoff_j]
               the extent to which agent i's TEAMMATES are behaving like the
               old team, i.e. the condition under which i's offline preference
               was learned

and asks whether -dev_i predicts the realised return-to-go G_t, and whether
that predictive validity GROWS with agree_i. If offline preferences only hold
when teammates match the old team, corr(-dev_i, G) must increase in agree_i.

G is confounded by episode phase and by policy version, so G is residualised
within (version, phase decile) before any correlation is taken. Controls:
  * shuffle   - agree_i permuted within version, destroying the pairing only
  * online    - agree_i recomputed from the LIVE online net instead of the
                frozen offline net, so "teammates deviate from *something*"
                is distinguished from "teammates deviate from the old team"
  * self      - condition on agent i's own agreement instead of its teammates',
                which is not a cooperative-knowledge claim

Usage:
  python scripts/analyze_tcvm_validity.py --dir checkpoints/tcvm_premise/5m_vs_6m
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def residualise(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract the group mean, so no between-group variation survives."""
    out = values.astype(np.float64).copy()
    for key in np.unique(groups):
        sel = groups == key
        out[sel] -= out[sel].mean()
    return out


def phase_decile(t: np.ndarray, length: np.ndarray, n: int = 10) -> np.ndarray:
    frac = t.astype(np.float64) / np.maximum(length, 1)
    return np.clip((frac * n).astype(np.int64), 0, n - 1)


def others_mean(per_agent: np.ndarray) -> np.ndarray:
    """[T, N] -> [T, N] where entry (t, i) is the mean over agents j != i."""
    total = per_agent.sum(axis=1, keepdims=True)
    n = per_agent.shape[1]
    return (total - per_agent) / max(n - 1, 1)


def load_flat(path: str) -> Dict[str, np.ndarray]:
    """Flatten per-step/per-agent records into one row per (step, agent)."""
    rec = {key: value for key, value in np.load(path).items()}
    tag = os.path.splitext(os.path.basename(path))[0]
    n_agents = rec["off_margin"].shape[1]

    off_greedy = rec["off_greedy"].astype(np.float64)
    on_greedy = (rec["on_margin"].astype(np.float64) <= 1e-6).astype(np.float64)

    agree_others_off = others_mean(off_greedy)
    agree_others_on = others_mean(on_greedy)

    dec = phase_decile(rec["t"], rec["length"])
    rep = lambda v: np.repeat(v[:, None], n_agents, axis=1).reshape(-1)

    return {
        "tag": np.array([tag]),
        "dev": rec["off_margin"].astype(np.float64).reshape(-1),
        "dev_logp": (-rec["off_logp"].astype(np.float64)).reshape(-1),
        "agree_others": agree_others_off.reshape(-1),
        "agree_others_online": agree_others_on.reshape(-1),
        "agree_self": off_greedy.reshape(-1),
        "g": rep(rec["g"].astype(np.float64)),
        "episode": rep(rec["episode"]),
        "phase": rep(dec),
        "version": np.full(rec["g"].size * n_agents, tag),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 20 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def tercile_trend(
    dev: np.ndarray, g_resid: np.ndarray, cond: np.ndarray
) -> Tuple[List[Dict], float]:
    """corr(-dev, G_resid) inside each tercile of the conditioning variable."""
    edges = np.quantile(cond, [0.0, 1.0 / 3, 2.0 / 3, 1.0])
    rows = []
    for b in range(3):
        lo, hi = edges[b], edges[b + 1]
        sel = (cond >= lo) & (cond <= hi) if b == 2 else (cond >= lo) & (cond < hi)
        rows.append(
            {
                "tercile": b,
                "lo": float(lo),
                "hi": float(hi),
                "n": int(sel.sum()),
                "corr": corr(-dev[sel], g_resid[sel]),
                "mean_dev": float(dev[sel].mean()) if sel.any() else float("nan"),
            }
        )
    slope = rows[2]["corr"] - rows[0]["corr"]
    return rows, float(slope)


def bootstrap_slope(
    dev: np.ndarray,
    g_resid: np.ndarray,
    cond: np.ndarray,
    episode: np.ndarray,
    version: np.ndarray,
    iters: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    clusters = np.char.add(version.astype(str), episode.astype(str))
    uniq, inverse = np.unique(clusters, return_inverse=True)
    index = [np.flatnonzero(inverse == k) for k in range(uniq.size)]
    draws = np.empty(iters)
    for i in range(iters):
        pick = rng.integers(0, len(index), size=len(index))
        rows = np.concatenate([index[j] for j in pick])
        _, slope = tercile_trend(dev[rows], g_resid[rows], cond[rows])
        draws[i] = slope
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints/tcvm_premise/5m_vs_6m")
    parser.add_argument("--out", default="")
    parser.add_argument("--boot", type=int, default=400)
    parser.add_argument("--dev", default="dev", choices=["dev", "dev_logp"])
    args = parser.parse_args()

    paths = [
        p
        for p in sorted(glob.glob(os.path.join(args.dir, "*.npz")))
        if not os.path.basename(p).startswith("offlinedata")
    ]
    if not paths:
        raise SystemExit(f"no online record files in {args.dir}")
    flats = [load_flat(p) for p in paths]

    keys = [
        "dev",
        "dev_logp",
        "agree_others",
        "agree_others_online",
        "agree_self",
        "g",
        "episode",
        "phase",
        "version",
    ]
    data = {k: np.concatenate([f[k] for f in flats]) for k in keys}

    # Residualise the outcome within (version, phase decile): any correlation
    # that survives cannot be explained by a global schedule or by where in the
    # episode the step sits.
    groups = np.char.add(data["version"], data["phase"].astype(str))
    g_resid = residualise(data["g"], groups)
    dev = data[args.dev]

    print(f"rows={dev.size}  versions={len(flats)}  outcome=G residualised within (version, phase decile)")
    print(f"baseline corr(-{args.dev}, G_resid) = {corr(-dev, g_resid):+.4f}\n")

    rng = np.random.default_rng(0)
    shuffled = data["agree_others"].copy()
    for tag in np.unique(data["version"]):
        sel = data["version"] == tag
        shuffled[sel] = rng.permutation(shuffled[sel])

    conditions = {
        "agree_others(offline)": data["agree_others"],
        "agree_others(online)": data["agree_others_online"],
        "agree_self(offline)": data["agree_self"],
        "agree_others(shuffled)": shuffled,
    }

    results: Dict[str, Dict] = {}
    print(f"{'condition':26s} {'corr|low':>9s} {'corr|mid':>9s} {'corr|high':>10s} {'slope':>8s}")
    for name, cond in conditions.items():
        rows, slope = tercile_trend(dev, g_resid, cond)
        results[name] = {"terciles": rows, "slope": slope}
        print(
            f"{name:26s} {rows[0]['corr']:+9.4f} {rows[1]['corr']:+9.4f} "
            f"{rows[2]['corr']:+10.4f} {slope:+8.4f}"
        )

    if args.boot > 0:
        for name in ("agree_others(offline)", "agree_others(online)", "agree_others(shuffled)"):
            lo, hi = bootstrap_slope(
                dev,
                g_resid,
                conditions[name],
                data["episode"],
                data["version"],
                args.boot,
                0,
            )
            results[name]["ci"] = [lo, hi]
            print(
                f"  slope {name:26s} {results[name]['slope']:+.4f} "
                f"95% CI [{lo:+.4f}, {hi:+.4f}]"
            )

    print("\nper-version slope of corr(-dev, G_resid) across agree_others terciles")
    per_version = {}
    for flat in flats:
        tag = str(flat["tag"][0])
        sel = data["version"] == tag
        rows, slope = tercile_trend(dev[sel], g_resid[sel], data["agree_others"][sel])
        per_version[tag] = {"terciles": rows, "slope": slope}
        print(
            f"  {tag:24s} low={rows[0]['corr']:+.4f} mid={rows[1]['corr']:+.4f} "
            f"high={rows[2]['corr']:+.4f} slope={slope:+.4f}"
        )
    n_pos = sum(1 for v in per_version.values() if v["slope"] > 0)
    print(f"  versions with positive slope: {n_pos}/{len(per_version)}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "dev_signal": args.dev,
                    "rows": int(dev.size),
                    "baseline_corr": corr(-dev, g_resid),
                    "conditions": results,
                    "per_version": per_version,
                    "n_positive_slope": n_pos,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
