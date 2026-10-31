#!/usr/bin/env python3
"""Validate joint pair selection against additive single-agent selection.

Both methods retain exactly two offline agents and receive the same total number
of selection episodes. Joint selection estimates all ten pair returns directly.
Independent selection estimates five single-agent returns and adds the two
single-agent scores for each pair. The all-online return is omitted from the
additive score because it is constant across pairs and cannot change the choice.

Repeated random split-half validation keeps selection and evaluation episodes
disjoint. The main endpoint is the selected pair's validation return under the
joint method minus that under the independent method. Aggregate intervals
resample arm/version clusters rather than treating repeated splits or training
seeds as independent observations.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import zlib

import numpy as np

N_AGENTS = 5
PAIRS = list(itertools.combinations(range(N_AGENTS), 2))
PAIR_NAMES = [f"pair_{i}_{j}" for i, j in PAIRS]
SINGLE_NAMES = [f"single_{i}" for i in range(N_AGENTS)]
REQUIRED_CONFIGS = {"all", *SINGLE_NAMES, *PAIR_NAMES}
SPLIT_SEED = 20261027
BOOT_SEED = 20261028


def input_paths() -> list[str]:
    paths = [
        "checkpoints/pv_exp2a/5m_vs_6m/seed1_natural_shard1.json",
        "checkpoints/pv_exp2a/5m_vs_6m/seed1_natural_shard2.json",
    ]
    paths += sorted(glob.glob("checkpoints/extra_20261024/A_arm*_t*.json"))
    paths += sorted(glob.glob("checkpoints/missing_ae_20261024/A_arm*.json"))
    return paths


def load_cells(paths: list[str]) -> tuple[dict, list[str]]:
    cells = {}
    skipped = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("map") != "5m_vs_6m":
            raise ValueError(f"unexpected map in {path}: {payload.get('map')}")
        arm = str(payload.get("arm", "A"))
        train_seed = int(payload.get("seed", 1))
        for t, configs in payload.get("natural", {}).items():
            key = f"{arm}@{int(t)}@s{train_seed}"
            lengths = {
                len(configs[name].get("episode_returns", []))
                for name in REQUIRED_CONFIGS
                if name in configs
            }
            valid = REQUIRED_CONFIGS.issubset(configs) and len(lengths) == 1 and lengths.pop() > 1
            if not valid:
                skipped.append(key)
                continue
            if key in cells:
                raise ValueError(f"duplicate Insight A cell: {key}")
            cells[key] = {
                "arm": arm,
                "t": int(t),
                "seed": train_seed,
                "configs": configs,
                "source": path,
            }
    return dict(sorted(cells.items())), skipped


def evaluate_cell(cell: dict, pair_budget: int, n_splits: int) -> dict:
    configs = cell["configs"]
    n_episodes = len(configs["all"]["episode_returns"])
    selection_size = n_episodes // 2
    if pair_budget < 1 or 2 * pair_budget > selection_size:
        raise ValueError(
            f"pair budget {pair_budget} is incompatible with {selection_size} selection episodes"
        )

    pair_returns = np.asarray(
        [configs[name]["episode_returns"] for name in PAIR_NAMES], dtype=np.float64
    )
    single_returns = np.asarray(
        [configs[name]["episode_returns"] for name in SINGLE_NAMES], dtype=np.float64
    )
    if not np.isfinite(pair_returns).all() or not np.isfinite(single_returns).all():
        raise ValueError("non-finite episode return")

    seed = SPLIT_SEED + zlib.crc32(
        f"{cell['arm']}@{cell['t']}@s{cell['seed']}".encode("utf-8")
    )
    rng = np.random.default_rng(seed)
    joint_values = np.empty(n_splits)
    independent_values = np.empty(n_splits)
    oracle_values = np.empty(n_splits)
    joint_choices = np.empty(n_splits, dtype=np.int64)
    independent_choices = np.empty(n_splits, dtype=np.int64)

    for repetition in range(n_splits):
        permutation = rng.permutation(n_episodes)
        selection = permutation[:selection_size]
        validation = permutation[selection_size:]

        pair_scores = pair_returns[:, selection[:pair_budget]].mean(axis=1)
        single_scores = single_returns[:, selection[: 2 * pair_budget]].mean(axis=1)
        additive_scores = np.asarray(
            [single_scores[i] + single_scores[j] for i, j in PAIRS]
        )

        joint_choice = int(np.argmax(pair_scores))
        independent_choice = int(np.argmax(additive_scores))
        validation_means = pair_returns[:, validation].mean(axis=1)

        joint_choices[repetition] = joint_choice
        independent_choices[repetition] = independent_choice
        joint_values[repetition] = validation_means[joint_choice]
        independent_values[repetition] = validation_means[independent_choice]
        oracle_values[repetition] = validation_means.max()

    differences = joint_values - independent_values
    return {
        "arm": cell["arm"],
        "t": cell["t"],
        "seed": cell["seed"],
        "source": cell["source"],
        "pair_budget": pair_budget,
        "single_budget": 2 * pair_budget,
        "selection_episodes_per_method": len(PAIRS) * pair_budget,
        "validation_episodes_per_method": n_episodes - selection_size,
        "n_splits": n_splits,
        "joint_validation_return": float(joint_values.mean()),
        "independent_validation_return": float(independent_values.mean()),
        "difference": float(differences.mean()),
        "difference_split_sd": float(differences.std(ddof=1)),
        "joint_better_rate": float((differences > 0).mean()),
        "independent_better_rate": float((differences < 0).mean()),
        "tie_rate": float((differences == 0).mean()),
        "same_pair_rate": float((joint_choices == independent_choices).mean()),
        "joint_regret": float((oracle_values - joint_values).mean()),
        "independent_regret": float((oracle_values - independent_values).mean()),
        "joint_choice_frequency": {
            name: float((joint_choices == index).mean())
            for index, name in enumerate(PAIR_NAMES)
        },
        "independent_choice_frequency": {
            name: float((independent_choices == index).mean())
            for index, name in enumerate(PAIR_NAMES)
        },
    }


def aggregate(results: dict, n_boot: int, seed: int) -> dict:
    keys = sorted(results)
    differences = np.asarray([results[key]["difference"] for key in keys])
    clusters = sorted({key.rsplit("@s", 1)[0] for key in keys})
    cluster_indices = [
        [index for index, key in enumerate(keys) if key.rsplit("@s", 1)[0] == cluster]
        for cluster in clusters
    ]

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_boot)
    for repetition in range(n_boot):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([differences[cluster_indices[index]] for index in sampled])
        bootstrap[repetition] = values.mean()

    cluster_differences = {
        cluster: float(differences[indices].mean())
        for cluster, indices in zip(clusters, cluster_indices)
    }
    return {
        "n_cells": len(keys),
        "n_arm_version_clusters": len(clusters),
        "mean_difference": float(differences.mean()),
        "difference_cluster_bootstrap_95ci": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "median_cell_difference": float(np.median(differences)),
        "cells_with_positive_difference": int((differences > 0).sum()),
        "cells_with_negative_difference": int((differences < 0).sum()),
        "clusters_with_positive_difference": int(sum(value > 0 for value in cluster_differences.values())),
        "clusters_with_negative_difference": int(sum(value < 0 for value in cluster_differences.values())),
        "cluster_differences": cluster_differences,
        "mean_joint_validation_return": float(
            np.mean([results[key]["joint_validation_return"] for key in keys])
        ),
        "mean_independent_validation_return": float(
            np.mean([results[key]["independent_validation_return"] for key in keys])
        ),
        "mean_joint_better_rate": float(
            np.mean([results[key]["joint_better_rate"] for key in keys])
        ),
        "mean_same_pair_rate": float(
            np.mean([results[key]["same_pair_rate"] for key in keys])
        ),
        "mean_joint_regret": float(np.mean([results[key]["joint_regret"] for key in keys])),
        "mean_independent_regret": float(
            np.mean([results[key]["independent_regret"] for key in keys])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-budgets", type=int, nargs="+", default=[5, 10, 25, 50])
    parser.add_argument("--main-pair-budget", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=500)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument(
        "--out", default="checkpoints/joint_selection_validation_5m_vs_6m/results.json"
    )
    args = parser.parse_args()
    if args.main_pair_budget not in args.pair_budgets:
        parser.error("--main-pair-budget must be included in --pair-budgets")

    paths = input_paths()
    cells, skipped = load_cells(paths)
    if not cells:
        raise RuntimeError("no Insight A cells with per-episode returns were found")
    print(f"loaded {len(cells)} cells; skipped {len(skipped)} without complete episode data", flush=True)

    results_by_budget = {}
    summary_by_budget = {}
    for budget in args.pair_budgets:
        results = {
            key: evaluate_cell(cell, budget, args.n_splits)
            for key, cell in cells.items()
        }
        results_by_budget[str(budget)] = results
        summary_by_budget[str(budget)] = aggregate(
            results, args.n_boot, BOOT_SEED + budget
        )
        summary = summary_by_budget[str(budget)]
        print(
            f"pair budget={budget:>2} total selection={10 * budget:>3}: "
            f"joint-independent={summary['mean_difference']:+.4f} "
            f"95% CI {summary['difference_cluster_bootstrap_95ci']} "
            f"positive cells={summary['cells_with_positive_difference']}/{summary['n_cells']}",
            flush=True,
        )

    main_results = results_by_budget[str(args.main_pair_budget)]
    print("\nmain-budget cell results", flush=True)
    for key, result in main_results.items():
        print(
            f"  {key:<18} joint={result['joint_validation_return']:7.3f} "
            f"independent={result['independent_validation_return']:7.3f} "
            f"diff={result['difference']:+.3f} "
            f"joint_better={result['joint_better_rate']:.3f} "
            f"same_pair={result['same_pair_rate']:.3f}",
            flush=True,
        )

    payload = {
        "meta": {
            "map": "5m_vs_6m",
            "comparison": "direct pair-return selection minus additive single-agent selection",
            "retained_agents_per_method": 2,
            "pair_candidates": PAIR_NAMES,
            "selection_episodes": "10*m per joint method; 5*2*m per independent method",
            "validation": "disjoint split-half episodes; repeated random splits",
            "aggregate_interval": "95% percentile bootstrap over arm/version clusters",
            "tie_breaking": "first pair in lexicographic candidate order",
            "n_splits": args.n_splits,
            "n_boot": args.n_boot,
            "split_seed": SPLIT_SEED,
            "boot_seed": BOOT_SEED,
            "main_pair_budget": args.main_pair_budget,
            "sources": paths,
            "skipped_cells": skipped,
        },
        "summary_by_budget": summary_by_budget,
        "main_results": main_results,
        "results_by_budget": results_by_budget,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
