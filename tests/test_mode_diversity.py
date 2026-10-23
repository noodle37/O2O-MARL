#!/usr/bin/env python3
"""Unit tests for the H3 mode-diversity measurement helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.analyze_mode_diversity import (
    bootstrap_offdiag,
    cluster_episodes,
    entropy,
    js_divergence,
    js_matrix,
    mean_offdiag,
    normalized_entropy,
    occupancy,
    parse_manifest,
)


def test_js_divergence_bounds_and_symmetry():
    a = {(0,): 0.5, (1,): 0.5}
    assert js_divergence(a, a) == 0.0
    b = {(2,): 1.0}
    assert abs(js_divergence(a, b) - 1.0) < 1e-12
    assert abs(js_divergence(a, b) - js_divergence(b, a)) < 1e-12
    c = {(0,): 1.0}
    assert abs(js_divergence(a, c) - 0.31127812445913283) < 1e-9


def test_js_matrix_symmetric_zero_diagonal():
    dists = [{(0,): 1.0}, {(1,): 1.0}, {(0,): 0.5, (1,): 0.5}]
    matrix = js_matrix(dists)
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.allclose(matrix, matrix.T)
    assert matrix[0, 1] == 1.0
    assert 0.0 < matrix[0, 2] < 1.0


def test_entropy_and_normalized():
    assert entropy({}) == 0.0
    assert entropy({0: 1.0}) == 0.0
    assert abs(entropy({0: 0.5, 1: 0.5}) - 1.0) < 1e-12
    assert normalized_entropy({0: 1.0}) == 0.0
    assert abs(normalized_entropy({0: 0.5, 1: 0.5}) - 1.0) < 1e-12
    # Non-uniform over three outcomes stays in (0, 1).
    value = normalized_entropy({0: 0.8, 1: 0.1, 2: 0.1})
    assert 0.0 < value < 1.0


def test_mean_offdiag_and_bootstrap():
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert mean_offdiag(matrix) == 1.0
    assert mean_offdiag(np.zeros((1, 1))) is None
    boot = bootstrap_offdiag(matrix, n_boot=50, rng=np.random.default_rng(0))
    assert boot["mean"] == 1.0
    # Resampling can pick the same episode twice, which contributes a zero pair.
    assert boot["ci95"][0] <= 1.0 <= boot["ci95"][1]


def test_occupancy_counts_and_effective_modes():
    occ = occupancy([0, 0, 0, 1], 2)
    assert occ["counts"] == [3, 1]
    assert np.allclose(occ["probs"], [0.75, 0.25])
    assert occ["dominant_share"] == 0.75
    assert 1.0 < occ["effective_modes"] < 2.0
    single = occupancy([0, 0], 1)
    assert single["entropy"] == 0.0
    assert single["effective_modes"] == 1.0
    assert single["dominant_share"] == 1.0


def test_cluster_episodes_recovers_two_groups():
    # Two tight groups of three episodes each, far apart.
    n = 6
    distance = np.full((n, n), 2.0)
    np.fill_diagonal(distance, 0.0)
    for group in ((0, 1, 2), (3, 4, 5)):
        for i in group:
            for j in group:
                if i != j:
                    distance[i, j] = 0.1
    result = cluster_episodes(distance, k_min=2, k_max=3)
    labels = result["labels"]
    assert sorted(set(labels)) == [0, 1]
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert result["best_k"] == 2
    assert result["silhouette"] is not None and result["silhouette"] > 0.5


def test_cluster_episodes_single_point_fallback():
    result = cluster_episodes(np.zeros((1, 1)), k_min=2, k_max=4)
    assert result["best_k"] == 1
    assert result["labels"] == [0]
    assert result["silhouette"] is None


def test_parse_manifest(tmp_path):
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    ckpt_a.write_bytes(b"")
    ckpt_b.write_bytes(b"")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "# comment\n"
        f"teacher_best offline 100 {ckpt_a}\n"
        "\n"
        f"uot_t1m uot 500 {ckpt_b}  # inline\n",
        encoding="utf-8",
    )
    specs = parse_manifest(str(manifest))
    assert [s.tag for s in specs] == ["teacher_best", "uot_t1m"]
    assert specs[0].family == "offline" and specs[0].t_env == 100
    assert specs[1].family == "uot" and specs[1].t_env == 500
    assert specs[1].path == str(ckpt_b.resolve())
