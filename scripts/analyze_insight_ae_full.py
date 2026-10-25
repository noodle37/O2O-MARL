#!/usr/bin/env python3
"""Merge every Insight A / Insight E run available on 5m_vs_6m.

Sources
-------
A:
  * main seed-1 version ladder   checkpoints/pv_exp2a/5m_vs_6m/seed1_natural_shard{1,2}.json
  * cross-arm at 1M (seed 1)     checkpoints/extra_20261024/A_arm{B,C,D}_t1M.json
  * missing cells                checkpoints/missing_ae_20261024/A_*.json
  * offline-team baseline        checkpoints/missing_ae_20261024/A_all_off_e200.json
E:
  * main seed-1 protocols        checkpoints/replay_gain/5m_vs_6m{,_u100,_lr1e4}.json
  * cross-arm at 1M (seed 1)     checkpoints/extra_20261024/E_*.json
  * missing cells                checkpoints/missing_ae_20261024/E_*.json

The script writes one machine-readable index with per-cell effects, paired
episode-level bootstrap intervals, and seed-level aggregates.  It does not
recompute training; everything is a reader for the saved per-episode returns.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

N_AGENTS = 5
PAIRS = list(itertools.combinations(range(N_AGENTS), 2))
N_BOOT = 2000
BOOT_SEED = 20261025


def mean_ci(x: np.ndarray, rng: np.random.Generator) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    boot = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_ci(diff: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    x = np.asarray(diff, dtype=float)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    boot = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
    }


def fmt_ci(est: Dict[str, float]) -> str:
    if est["lo"] is None:
        return f"{est['mean']:+.3f} [n/a]"
    return f"{est['mean']:+.3f} [{est['lo']:+.3f}, {est['hi']:+.3f}]"


# --------------------------------------------------------------------------- A


def load_a(natural_paths: List[str]) -> Dict[Tuple[str, int, int], Dict]:
    cells: Dict[Tuple[str, int, int], Dict] = {}
    for path in natural_paths:
        payload = json.load(open(path, encoding="utf-8"))
        arm = payload.get("arm") or "A"
        seed = int(payload.get("seed", 1))
        for t, configs in payload.get("natural", {}).items():
            cells[(arm, int(t), seed)] = configs
    return cells


def a_cell_metrics(configs: Dict, rng: np.random.Generator) -> Dict:
    has_episodes = all(
        isinstance(c.get("episode_returns"), list) and len(c["episode_returns"]) > 1
        for c in configs.values()
    )
    returns = {name: float(np.mean(c["episode_returns"])) if has_episodes else float(c["mean_return"])
               for name, c in configs.items()}
    base = np.asarray(configs["all"]["episode_returns"], dtype=float) if has_episodes else None

    g = {}
    for i in range(N_AGENTS):
        if has_episodes:
            diff = base - np.asarray(configs[f"single_{i}"]["episode_returns"], dtype=float)
            g[i] = paired_ci(diff, rng)
        else:
            g[i] = {"mean": returns["all"] - returns[f"single_{i}"], "lo": None, "hi": None}

    I = {}
    for i, j in PAIRS:
        if has_episodes:
            diff = (
                np.asarray(configs[f"pair_{i}_{j}"]["episode_returns"], dtype=float)
                - np.asarray(configs[f"single_{i}"]["episode_returns"], dtype=float)
                - np.asarray(configs[f"single_{j}"]["episode_returns"], dtype=float)
                + base
            )
            I[f"{i}_{j}"] = paired_ci(diff, rng)
        else:
            I[f"{i}_{j}"] = {
                "mean": returns[f"pair_{i}_{j}"] - returns[f"single_{i}"] - returns[f"single_{j}"] + returns["all"],
                "lo": None,
                "hi": None,
            }

    best_name = max((n for n in returns if n != "all"), key=returns.get)
    worst_name = min((n for n in returns if n != "all"), key=returns.get)
    if has_episodes:
        best_gap = paired_ci(np.asarray(configs[best_name]["episode_returns"], dtype=float) - base, rng)
        worst_gap = paired_ci(np.asarray(configs[worst_name]["episode_returns"], dtype=float) - base, rng)
    else:
        best_gap = {"mean": returns[best_name] - returns["all"], "lo": None, "hi": None}
        worst_gap = {"mean": returns[worst_name] - returns["all"], "lo": None, "hi": None}

    beneficial = []
    harmful = []
    if has_episodes:
        for name, c in configs.items():
            if name == "all":
                continue
            est = paired_ci(np.asarray(c["episode_returns"], dtype=float) - base, rng)
            if est["lo"] > 0:
                beneficial.append(name)
            if est["hi"] < 0:
                harmful.append(name)
        n_beneficial = len(beneficial)
        n_harmful = len(harmful)
    else:
        n_beneficial = sum(returns[n] > returns["all"] for n in returns if n != "all")
        n_harmful = sum(returns[n] < returns["all"] for n in returns if n != "all")

    return {
        "has_episode_data": has_episodes,
        "all_return": returns["all"],
        "all_win": float(configs["all"]["win_rate"]),
        "all_team_deviation": float(configs["all"]["mean_team_deviation"]),
        "q_spearman": configs["all"].get("spearman_rho"),
        "q_bias_norm": configs["all"].get("bias_normalized_mean"),
        "g": g,
        "I": I,
        "config_returns": returns,
        "best_transplant": best_name,
        "best_gap": best_gap,
        "worst_transplant": worst_name,
        "worst_gap": worst_gap,
        "beneficial_configs": beneficial,
        "harmful_configs": harmful,
        "n_beneficial": n_beneficial,
        "n_harmful": n_harmful,
    }


def a_aggregate(cells: Dict[Tuple[str, int, int], Dict], metrics: Dict) -> Dict:
    grouped: Dict[Tuple[str, int], List[int]] = {}
    for arm, t, seed in cells:
        grouped.setdefault((arm, t), []).append(seed)

    out = {}
    for (arm, t), seeds in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        seeds = sorted(seeds)
        rows = [metrics[(arm, t, s)] for s in seeds]
        g_mean = {i: float(np.mean([r["g"][i]["mean"] for r in rows])) for i in range(N_AGENTS)}
        g_pos = {i: int(sum(r["g"][i]["mean"] > 0 for r in rows)) for i in range(N_AGENTS)}
        I_mean = {k: float(np.mean([r["I"][k]["mean"] for r in rows])) for k in rows[0]["I"]}
        I_pos = {k: int(sum(r["I"][k]["mean"] > 0 for r in rows)) for k in rows[0]["I"]}
        best_gap_mean = float(np.mean([r["best_gap"]["mean"] for r in rows]))

        stability = {}
        if len(seeds) > 1:
            for sa, sb in itertools.combinations(seeds, 2):
                ga = np.array([metrics[(arm, t, sa)]["g"][i]["mean"] for i in range(N_AGENTS)])
                gb = np.array([metrics[(arm, t, sb)]["g"][i]["mean"] for i in range(N_AGENTS)])
                ia = np.array([metrics[(arm, t, sa)]["I"][k]["mean"] for k in rows[0]["I"]])
                ib = np.array([metrics[(arm, t, sb)]["I"][k]["mean"] for k in rows[0]["I"]])
                stability[f"{sa}->{sb}"] = {
                    "g_spearman": float(spearmanr(ga, gb).statistic),
                    "I_spearman": float(spearmanr(ia, ib).statistic),
                }

        out[f"{arm}@{t}"] = {
            "arm": arm,
            "t": t,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "mean_all_return": float(np.mean([r["all_return"] for r in rows])),
            "mean_g": g_mean,
            "seeds_g_positive": g_pos,
            "mean_I": I_mean,
            "seeds_I_positive": I_pos,
            "mean_best_gap": best_gap_mean,
            "best_transplant_by_seed": {s: metrics[(arm, t, s)]["best_transplant"] for s in seeds},
            "n_beneficial_by_seed": {s: metrics[(arm, t, s)]["n_beneficial"] for s in seeds},
            "n_harmful_by_seed": {s: metrics[(arm, t, s)]["n_harmful"] for s in seeds},
            "seed_stability": stability,
        }
    return out


# --------------------------------------------------------------------------- E


def e_cell_key(path: str, payload: Dict) -> Tuple:
    arm = payload.get("arm") or "A"
    seed = int(payload.get("seed", 1))
    lr = float(payload.get("lr", 1e-3))
    upd = int(payload["updates"])
    eps = int(payload["eval_episodes"])
    return (arm, seed, lr, upd, eps)


def load_e(paths: List[str]) -> Dict[Tuple, Dict]:
    cells: Dict[Tuple, Dict] = {}
    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        key = e_cell_key(path, payload)
        for t, version in payload["versions"].items():
            cell = {
                "arm": payload.get("arm") or "A",
                "seed": int(payload.get("seed", 1)),
                "t": int(t),
                "lr": float(payload.get("lr", 1e-3)),
                "updates": int(payload["updates"]),
                "eval_episodes": int(payload["eval_episodes"]),
                "theta0": version["theta0"],
                "groups": version["groups"],
                "source": path,
            }
            k = (cell["arm"], cell["seed"], cell["t"], cell["lr"], cell["updates"], cell["eval_episodes"])
            if k in cells:
                print(f"[warn] duplicate E cell {k}; keeping {path}")
            cells[k] = cell
    return cells


def e_cell_metrics(cell: Dict, rng: np.random.Generator) -> Dict:
    theta0 = cell["theta0"]
    theta_base = float(theta0["return"])
    groups = cell["groups"]
    names = list(groups)
    has_episodes = all(
        isinstance(g["eval"].get("episode_returns"), list) and len(g["eval"]["episode_returns"]) > 1
        for g in groups.values()
    ) and isinstance(theta0.get("episode_returns"), list)

    u = {}
    u_win = {}
    for name in names:
        g = groups[name]
        u[name] = float(g["eval"]["return"] - theta_base)
        u_win[name] = float(g["eval"]["win_rate"] - theta0["win_rate"])

    excess = {}
    for name in names:
        if name == "random":
            continue
        if has_episodes:
            est = paired_ci(
                np.asarray(groups[name]["eval"]["episode_returns"], dtype=float)
                - np.asarray(groups["random"]["eval"]["episode_returns"], dtype=float),
                rng,
            )
        else:
            est = {
                "mean": u[name] - u["random"],
                "lo": None,
                "hi": None,
            }
        excess[name] = est

    ranked = sorted(names, key=lambda n: -u[n])
    return {
        "has_episode_data": has_episodes,
        "theta0_return": theta_base,
        "theta0_win": float(theta0["win_rate"]),
        "u_return": u,
        "u_win": u_win,
        "excess_over_random": excess,
        "rank_by_u": ranked,
        "best_group": ranked[0],
        "n_positive_u": int(sum(v > 0 for v in u.values())),
        "n_groups": len(names),
        "train_loss": {n: {
            "first50": groups[n]["train"]["mean_loss_first50"],
            "last50": groups[n]["train"]["mean_loss_last50"],
        } for n in names},
    }


def e_aggregate(cells: Dict[Tuple, Dict], metrics: Dict) -> Dict:
    grouped: Dict[Tuple[str, int, float, int, int], List[int]] = {}
    for k in cells:
        arm, seed, t, lr, upd, eps = k
        grouped.setdefault((arm, t, lr, upd, eps), []).append(seed)

    out = {}
    for key, seeds in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        arm, t, lr, upd, eps = key
        seeds = sorted(seeds)
        rows = [metrics[(arm, s, t, lr, upd, eps)] for s in seeds]
        names = list(rows[0]["u_return"])
        u_mean = {n: float(np.mean([r["u_return"][n] for r in rows])) for n in names}
        u_sd = {n: float(np.std([r["u_return"][n] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for n in names}
        ex_mean = {
            n: float(np.mean([r["excess_over_random"][n]["mean"] for r in rows]))
            for n in names if n != "random"
        }
        ex_pos = {
            n: int(sum(r["excess_over_random"][n]["mean"] > 0 for r in rows))
            for n in names if n != "random"
        }
        rank_counts = {n: 0 for n in names}
        for r in rows:
            rank_counts[r["best_group"]] += 1
        mean_rank = {
            n: float(np.mean([r["rank_by_u"].index(n) + 1 for r in rows])) for n in names
        }

        stability = {}
        if len(seeds) > 1:
            for sa, sb in itertools.combinations(seeds, 2):
                ua = np.array([metrics[(arm, sa, t, lr, upd, eps)]["u_return"][n] for n in names])
                ub = np.array([metrics[(arm, sb, t, lr, upd, eps)]["u_return"][n] for n in names])
                stability[f"{sa}->{sb}"] = float(spearmanr(ua, ub).statistic)

        out[f"{arm}@{t}@lr{lr:g}@u{upd}@e{eps}"] = {
            "arm": arm,
            "t": t,
            "lr": lr,
            "updates": upd,
            "eval_episodes": eps,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "mean_theta0": float(np.mean([r["theta0_return"] for r in rows])),
            "mean_u": u_mean,
            "sd_u_across_seeds": u_sd,
            "mean_excess_over_random": ex_mean,
            "seeds_excess_positive": ex_pos,
            "best_group_count": rank_counts,
            "mean_rank": mean_rank,
            "seed_spearman": stability,
            "n_positive_u_by_seed": {s: metrics[(arm, s, t, lr, upd, eps)]["n_positive_u"] for s in seeds},
        }
    return out


def main() -> int:
    global N_BOOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="checkpoints/insight_ae_full_20261025")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    N_BOOT = args.n_boot

    rng = np.random.default_rng(BOOT_SEED)

    a_paths = [
        "checkpoints/pv_exp2a/5m_vs_6m/seed1_natural_shard1.json",
        "checkpoints/pv_exp2a/5m_vs_6m/seed1_natural_shard2.json",
    ]
    a_paths += sorted(glob.glob("checkpoints/extra_20261024/A_arm*_t*.json"))
    a_paths += sorted(glob.glob("checkpoints/missing_ae_20261024/A_arm*.json"))
    a_cells = load_a(a_paths)

    base_path = "checkpoints/missing_ae_20261024/A_all_off_e200.json"
    all_off = json.load(open(base_path, encoding="utf-8"))["random"]["0.0"]
    offline_baseline = float(all_off["mean_return"])

    a_metrics = {k: a_cell_metrics(v, rng) for k, v in a_cells.items()}
    a_agg = a_aggregate(a_cells, a_metrics)

    e_paths = sorted(glob.glob("checkpoints/replay_gain/5m_vs_6m*.json"))
    e_paths = [p for p in e_paths if not p.endswith("analysis.json")]
    e_paths += sorted(glob.glob("checkpoints/extra_20261024/E_*.json"))
    e_paths += sorted(glob.glob("checkpoints/missing_ae_20261024/E_*.json"))
    e_cells = load_e(e_paths)
    e_metrics = {k: e_cell_metrics(v, rng) for k, v in e_cells.items()}
    e_agg = e_aggregate(e_cells, e_metrics)

    # ---- headline summaries -------------------------------------------------
    a_summary = {
        "n_cells": len(a_cells),
        "n_cells_with_episode_data": int(sum(m["has_episode_data"] for m in a_metrics.values())),
        "n_arms": len({k[0] for k in a_cells}),
        "offline_baseline": offline_baseline,
        "cells_with_any_significant_beneficial_transplant": int(
            sum(m["has_episode_data"] and m["n_beneficial"] > 0 for m in a_metrics.values())
        ),
        "cells_with_any_significant_harmful_transplant": int(
            sum(m["has_episode_data"] and m["n_harmful"] > 0 for m in a_metrics.values())
        ),
        "cells_best_above_all_descriptive": int(
            sum(m["best_gap"]["mean"] > 0 for m in a_metrics.values())
        ),
        "cells_by_arm_version_seeds": {
            f"{arm}@{t}": sorted(s for a, tt, s in a_cells if a == arm and tt == t)
            for arm, t in sorted({(k[0], k[1]) for k in a_cells})
        },
        "mean_best_gap_by_arm_version": {
            k: v["mean_best_gap"] for k, v in a_agg.items()
        },
        "max_all_return": max(m["all_return"] for m in a_metrics.values()),
        "min_all_return": min(m["all_return"] for m in a_metrics.values()),
        "all_return_above_offline": int(sum(m["all_return"] >= offline_baseline for m in a_metrics.values())),
        "all_return_below_offline": int(sum(m["all_return"] < offline_baseline for m in a_metrics.values())),
    }

    e_summary = {
        "n_cells": len(e_cells),
        "n_group_observations": len(e_cells) * 4,
        "u_positive": int(sum(v > 0 for m in e_metrics.values() for v in m["u_return"].values())),
        "u_negative": int(sum(v < 0 for m in e_metrics.values() for v in m["u_return"].values())),
        "best_group_counts": {
            n: int(sum(m["best_group"] == n for m in e_metrics.values()))
            for n in ["td", "density", "return", "random"]
        },
        "excess_positive": {
            n: int(sum(m["excess_over_random"][n]["mean"] > 0 for m in e_metrics.values()))
            for n in ["td", "density", "return"]
        },
        "excess_ci_below_zero": {
            n: int(sum(
                m["excess_over_random"][n]["hi"] is not None and m["excess_over_random"][n]["hi"] < 0
                for m in e_metrics.values()
            ))
            for n in ["td", "density", "return"]
        },
        "excess_ci_above_zero": {
            n: int(sum(
                m["excess_over_random"][n]["lo"] is not None and m["excess_over_random"][n]["lo"] > 0
                for m in e_metrics.values()
            ))
            for n in ["td", "density", "return"]
        },
        "cells_by_arm_version_protocol": {
            k: v["seeds"] for k, v in e_agg.items()
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "meta": {
            "map": "5m_vs_6m",
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "bootstrap": "paired episode-level percentile bootstrap over the common 200/128/64-episode evaluation sequence",
            "sources": {"A": a_paths, "E": e_paths},
        },
        "A": {
            "summary": a_summary,
            "cells": {
                f"{arm}@{t}@s{seed}": {
                    **m,
                    "g": {str(k): v for k, v in m["g"].items()},
                    "I": m["I"],
                }
                for (arm, t, seed), m in sorted(a_metrics.items())
            },
            "aggregates": a_agg,
        },
        "E": {
            "summary": e_summary,
            "cells": {
                f"{arm}@{t}@lr{lr:g}@u{u}@e{eps}@s{seed}": m
                for (arm, seed, t, lr, u, eps), m in sorted(e_metrics.items())
            },
            "aggregates": e_agg,
        },
    }
    out = os.path.join(args.out_dir, "index.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # ---- console report -----------------------------------------------------
    print(f"[A] cells={len(a_cells)} offline_baseline={offline_baseline:.3f}")
    for (arm, t, seed), m in sorted(a_metrics.items()):
        g_str = " ".join("{:+.2f}".format(m["g"][i]["mean"]) for i in range(N_AGENTS))
        print(
            f"  {arm} t={t:>7} s{seed}: J={m['all_return']:6.3f} "
            f"best={m['best_transplant']:<10} gap={fmt_ci(m['best_gap'])} "
            f"g=[{g_str}]"
        )
    print("\n[A] arm/version aggregates")
    for k, v in a_agg.items():
        print(
            f"  {k:<14} seeds={v['seeds']} J={v['mean_all_return']:6.3f} "
            f"best_gap={v['mean_best_gap']:+.3f} "
            f"g+={[v['seeds_g_positive'][i] for i in range(5)]}/{v['n_seeds']} "
            f"I+={sum(1 for x in v['seeds_I_positive'].values() if x == v['n_seeds'])}/10"
        )

    print(f"\n[E] cells={len(e_cells)} group-observations={len(e_cells)*4}")
    for (arm, seed, t, lr, u, eps), m in sorted(e_metrics.items()):
        best = m["best_group"]
        u_str = ", ".join(
            "{}:{:+.2f}".format(n, m["u_return"][n]) for n in m["u_return"]
        )
        print(
            f"  {arm} t={t:>7} lr={lr:g} u={u:>3} e={eps:>3} s{seed}: "
            f"U={{{u_str}}} best={best}"
        )
    print("\n[E] aggregates")
    for k, v in e_agg.items():
        print(
            f"  {k:<38} seeds={v['seeds']} best={v['best_group_count']} "
            f"excess+={v['seeds_excess_positive']} rho={ {kk: round(vv,2) for kk,vv in v['seed_spearman'].items()} }"
        )

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
