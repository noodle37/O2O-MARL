#!/usr/bin/env python3
"""Analyze teammate-conditioned prior value and pairwise interactions."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> Dict:
    return {
        "mean": float(values.mean()),
        "ci95": list(ci(values, n_boot, rng)),
        "episode_values": [float(x) for x in values],
    }


def checkpoint_key(name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", name)
    return (int(match.group(1)) if match else -1, name)


def load_inputs(specs: List[str]) -> Dict[str, Dict]:
    payloads = {}
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"bad --input '{spec}'; expected seed_label=json_path")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if float(payload["eps"][0]) != 0.0 or len(payload["eps"]) != 1:
            raise SystemExit(f"{path}: analysis requires exactly eps=0")
        payloads[label] = payload
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="seed_label=run_e1 JSON")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payloads = load_inputs(args.input)
    rng = np.random.default_rng(20261110)
    result = {
        "analysis": "Delta_i = J_i - J_empty; I_ij = J_ij - J_i - J_j + J_empty",
        "episode_bootstrap": args.bootstrap,
        "seeds": {},
    }
    cross_seed = defaultdict(list)

    for seed, payload in sorted(payloads.items()):
        rows = [r for r in payload["runs"] if float(r["eps"]) == 0.0]
        by_cell = defaultdict(dict)
        for row in rows:
            key = (row["checkpoint"], row["focal"])
            if row["branch"] in by_cell[key]:
                raise SystemExit(f"duplicate cell {seed} {key} {row['branch']}")
            by_cell[key][row["branch"]] = np.asarray(row["returns"], dtype=np.float64)

        checkpoints = sorted({r["checkpoint"] for r in rows}, key=checkpoint_key)
        focals = {r["focal"]: tuple(r["focal_agents"]) for r in rows}
        singles = {name: agents[0] for name, agents in focals.items() if len(agents) == 1}
        pairs = {name: agents for name, agents in focals.items() if len(agents) == 2}
        if len(singles) < 2 or not pairs:
            raise SystemExit(f"{seed}: need all singleton and pair focal evaluations")

        seed_out = {"checkpoints": {}, "dynamic_contrasts": {}}
        metric_history = defaultdict(dict)
        candidate_history = {}
        for checkpoint in checkpoints:
            baseline = None
            for focal in sorted(focals):
                cell = by_cell.get((checkpoint, focal), {})
                if "live" not in cell or "off" not in cell:
                    raise SystemExit(f"{seed}: incomplete cell {checkpoint} {focal}")
                if baseline is None:
                    baseline = cell["live"]
                elif not np.array_equal(baseline, cell["live"]):
                    raise SystemExit(f"{seed}: unpaired live baselines at {checkpoint}")
            n_episodes = baseline.size
            if n_episodes < 2:
                raise SystemExit(f"{seed}: insufficient episodes at {checkpoint}")

            ck_out = {
                "n_episodes": int(n_episodes),
                "J_empty": summarize(baseline, args.bootstrap, rng),
                "singles": {},
                "pairs": {},
            }
            candidates = {"empty": baseline}
            for focal, agent in sorted(singles.items()):
                off = by_cell[(checkpoint, focal)]["off"]
                if off.size != n_episodes:
                    raise SystemExit(f"{seed}: episode count mismatch {checkpoint} {focal}")
                delta = off - baseline
                ck_out["singles"][str(agent)] = {
                    "J": float(off.mean()),
                    "delta": summarize(delta, args.bootstrap, rng),
                }
                metric_history[f"delta_{agent}"][checkpoint] = delta
                candidates[f"single_{agent}"] = off

            for focal, (i, j) in sorted(pairs.items()):
                if i not in singles.values() or j not in singles.values():
                    raise SystemExit(f"{seed}: pair {focal} lacks singleton measurements")
                off_i = by_cell[(checkpoint, f"single_{i}")]["off"]
                off_j = by_cell[(checkpoint, f"single_{j}")]["off"]
                off_ij = by_cell[(checkpoint, focal)]["off"]
                if off_ij.size != n_episodes:
                    raise SystemExit(f"{seed}: episode count mismatch {checkpoint} {focal}")
                interaction = off_ij - off_i - off_j + baseline
                pair_name = f"{i}_{j}"
                ck_out["pairs"][pair_name] = {
                    "J": float(off_ij.mean()),
                    "delta_pair": summarize(off_ij - baseline, args.bootstrap, rng),
                    "interaction": summarize(interaction, args.bootstrap, rng),
                }
                metric_history[f"interaction_{pair_name}"][checkpoint] = interaction
                candidates[f"pair_{pair_name}"] = off_ij

            best = max(candidates, key=lambda name: float(candidates[name].mean()))
            ck_out["best_candidate"] = best
            ck_out["candidate_mean_returns"] = {
                name: float(values.mean()) for name, values in candidates.items()
            }
            candidate_history[checkpoint] = candidates
            seed_out["checkpoints"][checkpoint] = ck_out

        if len(checkpoints) >= 2:
            first, last = checkpoints[0], checkpoints[-1]
            for metric, history in sorted(metric_history.items()):
                change = history[last] - history[first]
                seed_out["dynamic_contrasts"][metric] = {
                    "first": first,
                    "last": last,
                    **summarize(change, args.bootstrap, rng),
                }
                cross_seed[metric].append(float(change.mean()))

            candidate_names = sorted(candidate_history[first])
            static_means = {
                name: float(np.mean([
                    candidate_history[ck][name].mean() for ck in checkpoints
                ]))
                for name in candidate_names
            }
            oracle = float(np.mean([
                max(values.mean() for values in candidate_history[ck].values())
                for ck in checkpoints
            ]))
            best_static = max(static_means, key=static_means.get)
            seed_out["dynamic_summary"] = {
                "best_static_candidate": best_static,
                "best_static_mean_return": static_means[best_static],
                "per_checkpoint_oracle_mean_return": oracle,
                "oracle_minus_static": oracle - static_means[best_static],
            }
            cross_seed["oracle_minus_static"].append(oracle - static_means[best_static])

        result["seeds"][seed] = seed_out

    result["cross_seed"] = {
        metric: {
            "n_seeds": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "seed_values": values,
        }
        for metric, values in sorted(cross_seed.items())
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    for seed, seed_out in sorted(result["seeds"].items()):
        print(f"\n=== {seed} ===")
        for checkpoint, ck in seed_out["checkpoints"].items():
            interactions = [v["interaction"]["mean"] for v in ck["pairs"].values()]
            deltas = [v["delta"]["mean"] for v in ck["singles"].values()]
            print(
                f"{checkpoint}: best={ck['best_candidate']} "
                f"Delta_range=[{min(deltas):+.3f},{max(deltas):+.3f}] "
                f"I_range=[{min(interactions):+.3f},{max(interactions):+.3f}]"
            )
        if "dynamic_summary" in seed_out:
            summary = seed_out["dynamic_summary"]
            print(f"oracle-static={summary['oracle_minus_static']:+.4f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
