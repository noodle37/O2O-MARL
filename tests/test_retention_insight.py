#!/usr/bin/env python3
"""Unit tests for the H1/H2 retention-insight measurement helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.retention_insight_core import (
    RETURN_TIE,
    _hamming,
    behaviour_candidates,
    build_deviations,
    candidate_metrics,
    cluster_bootstrap_diff,
    effect_bin,
    random_candidates,
    sample_random_pairs,
)
from scripts.verify_retention_insight import merge_shards, select_probe_steps, summarize


class _StubSidx:
    def __init__(self, marginal, counts=None):
        self.marginal = np.asarray(marginal, dtype=np.int64)
        self.counts = counts or {}


def _pair(cluster, outcome, abs_gap, kind="random_pair"):
    return {
        "cluster": cluster,
        "outcome": outcome,
        "abs_return_gap": abs_gap,
        "abs_q_gap": abs_gap,
        "kind": kind,
    }


def test_candidate_metrics_inverted_and_perfect():
    keys = [(0, 0), (0, 1), (1, 0)]
    q = [3.0, 2.0, 1.0]
    inverted = candidate_metrics(keys, q, [1.0, 2.0, 3.0])
    assert inverted["top1_correct"] is False
    assert inverted["top2_correct"] is False
    assert inverted["pair_concordance"] == 0.0
    assert inverted["kendall_tau"] == -1.0
    assert inverted["regret"] == 2.0
    assert inverted["true_best_qrank"] == 2
    assert inverted["q_top1_truerank"] == 2

    perfect = candidate_metrics(keys, q, [1.0, 2.0, 3.0][::-1])
    assert perfect["top1_correct"] is True
    assert perfect["top2_correct"] is True
    assert perfect["pair_concordance"] == 1.0
    assert perfect["regret"] == 0.0
    assert perfect["true_best_qrank"] == 0


def test_candidate_metrics_ties_and_small_sets():
    tied = candidate_metrics([(0,), (1,), (2,)], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0])
    assert tied["n_tie"] == 3
    assert tied["n_decided"] == 0
    assert tied["pair_concordance"] is None
    assert tied["top1_correct"] is True

    assert candidate_metrics([(0,)], [1.0], [0.0]) == {
        "k": 1,
        "n_pairs": 0,
        "n_decided": 0,
        "n_tie": 0,
    }


def test_build_deviations_is_one_agent_primary_and_hard():
    legal = torch.ones(2, 3, dtype=torch.bool)
    near = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int64)
    q = torch.tensor([[0.0, 5.0, 1.0], [0.0, 1.0, 9.0]])
    sidx = _StubSidx([[5, 1, 0], [5, 1, 0]])
    rng = np.random.default_rng(0)

    devs = build_deviations(
        (0, 0), legal, near, q, sidx, ("primary", "hard", "random"), rng
    )
    assert set(devs) == {
        ("primary", 0),
        ("primary", 1),
        ("hard", 0),
        ("hard", 1),
        ("random", 0),
        ("random", 1),
    }
    for (kind, agent), (joint, alt) in devs.items():
        assert _hamming(joint.tolist(), (0, 0)) == 1
        assert int(joint[agent]) == int(alt)
        if kind == "primary":
            assert int(alt) == 1
    assert int(devs[("hard", 0)][1]) == 1
    assert int(devs[("hard", 1)][1]) == 2


def test_build_deviations_skips_agents_without_alternative():
    legal = torch.tensor([[True, False], [True, True]])
    near = np.array([[0, 1]], dtype=np.int64)
    q = torch.zeros(2, 2)
    sidx = _StubSidx([[1, 0], [1, 1]])
    devs = build_deviations((0, 0), legal, near, q, sidx, ("primary",), np.random.default_rng(0))
    assert set(devs) == {("primary", 1)}


def test_behaviour_candidates_respects_legality_and_hamming():
    legal = torch.tensor([[True, True, False], [True, True, True]])
    near = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    sidx = _StubSidx([[3, 1, 1], [3, 1, 1]])
    cands = behaviour_candidates(
        (0, 0), legal, near, sidx, k=8, per_agent=2, max_hamming=1, n_recorded=8
    )
    keys = [tuple(int(x) for x in c.tolist()) for c in cands]
    assert keys[0] == (0, 0)
    assert len(keys) == len(set(keys))
    for key in keys:
        for agent, action in enumerate(key):
            assert bool(legal[agent, action])
    assert (1, 1) not in keys  # Hamming 2 from the base
    assert (0, 1) in keys and (1, 0) in keys

    loose = behaviour_candidates(
        (0, 0), legal, near, sidx, k=8, per_agent=2, max_hamming=2, n_recorded=8
    )
    loose_keys = [tuple(int(x) for x in c.tolist()) for c in loose]
    assert (1, 1) in loose_keys


def test_random_candidates_terminates_when_space_is_exhausted():
    legal = torch.tensor([[True, False], [True, False]])
    support = torch.tensor([[0, 0]])
    picks = random_candidates((0, 0), legal, support, k=4, rng=np.random.default_rng(0))
    assert len(picks) == 1
    assert tuple(int(x) for x in picks[0].tolist()) == (0, 0)

    legal_empty = torch.zeros(1, 1, dtype=torch.bool)
    picks = random_candidates((0,), legal_empty, torch.tensor([[0]]), k=3, rng=np.random.default_rng(0))
    assert len(picks) == 1


def test_random_candidates_are_legal_and_distinct():
    legal = torch.ones(2, 3, dtype=torch.bool)
    support = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    picks = random_candidates((0, 0), legal, support, k=4, rng=np.random.default_rng(1))
    keys = [tuple(int(x) for x in p.tolist()) for p in picks]
    assert len(keys) == 4
    assert len(set(keys)) == 4
    assert keys[0] == (0, 0)


def test_sample_random_pairs_distinct_and_handles_tiny_support():
    feasible = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    pairs = sample_random_pairs(feasible, 6, np.random.default_rng(0))
    assert len(pairs) == 6
    seen = set()
    for a, b in pairs:
        ka, kb = tuple(a.tolist()), tuple(b.tolist())
        assert ka != kb
        seen.add((min(ka, kb), max(ka, kb)))
    assert len(seen) == 6
    assert sample_random_pairs(torch.tensor([[0, 0]]), 4, np.random.default_rng(0)) == []


def test_cluster_bootstrap_diff_is_paired_by_cluster():
    a = [_pair("s1", "agree", 1.0), _pair("s2", "agree", 1.0)]
    b = [_pair("s1", "invert", 1.0), _pair("s2", "invert", 1.0)]
    out = cluster_bootstrap_diff(a, b, lambda r: 1.0 if r["outcome"] == "agree" else 0.0, n_boot=50)
    assert out["n_clusters"] == 2
    assert out["mean_diff"] == 1.0
    assert out["ci95"] == [1.0, 1.0]

    assert cluster_bootstrap_diff(a, [_pair("s3", "agree", 1.0)], lambda r: 1.0)["n_clusters"] == 0


def test_select_probe_steps_clamped_and_ordered():
    assert select_probe_steps(20, [0.3, 0.7], 2) == [6, 14]
    assert select_probe_steps(2, [0.3, 0.7], 2) == [1]
    assert select_probe_steps(10, [0.9, 0.1], 1) == [1]


def test_summarize_and_merge_shards(tmp_path):
    dev = [
        {
            "cluster": "1:greedy",
            "dev_kind": "primary",
            "dev_agent": 0,
            "abs_return_gap": 3.0,
            "abs_q_gap": 2.0,
            "signed_return_gap": 3.0,
            "outcome": "agree",
        },
        {
            "cluster": "2:greedy",
            "dev_kind": "hard",
            "dev_agent": 0,
            "abs_return_gap": 3.0,
            "abs_q_gap": 2.0,
            "signed_return_gap": -3.0,
            "outcome": "invert",
        },
    ]
    rnd = [
        {**_pair("1:greedy", "agree", 3.0)},
        {**_pair("2:greedy", "invert", 3.0)},
    ]
    sel = [
        {
            "cluster": "1:greedy",
            "set": "support_neigh",
            "k": 4,
            "skipped": False,
            "top1_correct": True,
            "top2_correct": True,
            "regret": 0.0,
            "true_best_qrank": 0,
            "q_top1_truerank": 0,
        },
        {
            "cluster": "2:greedy",
            "set": "support_neigh",
            "k": 4,
            "skipped": False,
            "top1_correct": False,
            "top2_correct": True,
            "regret": 2.0,
            "true_best_qrank": 2,
            "q_top1_truerank": 1,
        },
    ]
    cpairs = [
        {**_pair("1:greedy", "agree", 3.0, kind="support_neigh_pair"), "set": "support_neigh"},
        {**_pair("2:greedy", "invert", 3.0, kind="support_neigh_pair"), "set": "support_neigh"},
    ]
    gdiag = [
        {**_pair("1:greedy", "agree", 3.0, kind="global_top"), "pair_group": "global_diag"},
        {**_pair("2:greedy", "invert", 3.0, kind="global_long"), "pair_group": "global_diag"},
    ]
    records = {
        "episodes": [{"won": True}, {"won": False}],
        "states": [{"base_replay_gap": 0.0}, {"base_replay_gap": 0.1}],
        "deviations": dev,
        "random_pairs": rnd,
        "global_pairs": gdiag,
        "selectability": sel,
        "candidate_pairs": cpairs,
    }
    summary = summarize(records, None)
    assert summary["n_episodes"] == 2
    assert summary["n_wins"] == 1
    assert summary["n_global_pairs"] == 2
    assert summary["base_replay_gap_max"] == 0.1
    assert summary["selectability_by_set"]["support_neigh"]["n_sets"] == 2
    assert "deviation_minus_random_by_kind" in summary
    assert "selectability_minus_random" in summary

    payload = {
        "config": "configs/x.yaml",
        "frozen_checkpoint": "x.pt",
        "population": "win",
        "waves": ["greedy"],
        "eps": 0.3,
        "probe_fracs": [0.3],
        "dev_kinds": "primary,hard",
        "candidate_sets": "support_neigh",
        "cand_k": 8,
        "k_near": 64,
        "parameter_updates": 0,
        **records,
        "summary": summary,
    }
    (tmp_path / "shard_1.json").write_text(json.dumps(payload))
    (tmp_path / "driver.log").write_text("not json")
    merged, meta = merge_shards(tmp_path)
    assert meta["config"] == "configs/x.yaml"
    assert len(merged["deviations"]) == 2
    assert len(merged["states"]) == 2


def test_effect_bin_boundaries():
    assert effect_bin(0.0) == "[0.0,0.5)"
    assert effect_bin(0.49) == "[0.0,0.5)"
    assert effect_bin(0.5) == "[0.5,2.0)"
    assert effect_bin(10.0) == "[5.0,1000000000.0)"
    assert effect_bin(1e9) == ">=1000000000.0"
    assert RETURN_TIE > 0
