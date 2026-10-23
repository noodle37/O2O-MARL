# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Figure rendering for the problem-validation experiments.

Figure 1 (experiment 1), five panels:
  (a) full learning curve, median solid, IQR shaded
  (b) 0-150k zoom with the early-collapse depth and arm A's recovery marked
  (c) value-scale bias versus training step
  (d) individual preference retention versus training step
  (e) in-coverage ratio rho_cov versus training step

Figure 2 (experiment 2), six panels:
  (a) Spearman rho by D bin, natural and random shift
  (b) normalised bias by D bin, natural and random shift
  (c) g and I bar charts grouped by D
  (d) error versus D at t = 50k, 250k, 1M
  (e) in-coverage fraction versus lambda(t), with the two mismatch zones marked
  (f) memory-trigger rate by D bin (bars) with the "memory closer to G" rate
      (line) and a 0.5 reference line

Only matplotlib is used; figures are written as PNG so they can be dropped into
the report without further conversion.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARM_LABELS = {
    "A": "A direct fine-tune",
    "B": "B continual conservative",
    "C": "C OVMSE",
    "D": "D scale correction",
    "Z": "Z from scratch",
}
ARM_ORDER = ("A", "B", "C", "D", "Z")


def _out(path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent or ".", exist_ok=True)
    return path


def _panel_curves(ax, curves: Dict[str, List[Tuple[int, float]]], window=None, marks=None):
    """Plot one metric per arm: median line with an IQR band across seeds."""
    for arm in ARM_ORDER:
        if arm not in curves or not curves[arm]:
            continue
        arrays = [np.asarray(curves[arm])]
        # A single seed has no spread; the band collapses to the line.
        stacked = np.stack(arrays, axis=0)
        ts = stacked[0, :, 0]
        if window is not None:
            keep = ts <= window
            ts, stacked = ts[keep], stacked[:, keep, :]
        if ts.size == 0:
            continue
        med = np.median(stacked[:, :, 1], axis=0)
        ax.plot(ts, med, label=ARM_LABELS[arm], linewidth=1.6)
        if stacked.shape[0] > 1:
            q1 = np.quantile(stacked[:, :, 1], 0.25, axis=0)
            q3 = np.quantile(stacked[:, :, 1], 0.75, axis=0)
            ax.fill_between(ts, q1, q3, alpha=0.2)
    ax.set_xlabel("environment steps")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)


def figure1(
    curves: Dict[str, Dict[str, List[Tuple[int, float]]]],
    out_path: str,
    *,
    t_off: Optional[int] = None,
    drop_mark: Optional[Tuple[str, float]] = None,
    recovery_mark: Optional[Tuple[str, int]] = None,
    title: str = "",
) -> str:
    """curves[metric][arm] -> [(t, value)] for win_rate/return, bias, pref, cov."""
    metrics = [
        ("win_rate", "(a) learning curve"),
        ("win_rate", "(b) 0-150k zoom"),
        ("value_bias", "(c) value-scale bias"),
        ("preference_retention", "(d) preference retention"),
        ("rho_cov", "(e) in-coverage ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, metrics):
        _panel_curves(ax, curves.get(metric, {}), window=150_000 if "zoom" in label else None)
        ax.set_title(label, fontsize=10)
        if metric == "win_rate" and t_off is not None:
            ax.axhline(t_off, color="grey", linestyle=":", linewidth=1, label="offline start")
        if "zoom" in label and drop_mark is not None:
            arm, value = drop_mark
            ax.annotate(
                f"{arm} drop {value:.2f}",
                xy=(0.02, 0.06),
                xycoords="axes fraction",
                fontsize=8,
                color="darkred",
            )
        if "zoom" in label and recovery_mark is not None:
            arm, steps = recovery_mark
            ax.axvline(steps, color="darkred", linestyle="--", linewidth=1)
            ax.annotate(
                f"{arm} recovers @{steps // 1000}k",
                xy=(steps, 0.9),
                xycoords=("data", "axes fraction"),
                fontsize=8,
                color="darkred",
            )
    axes[5].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = _out(out_path)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure2(
    *,
    bins: Sequence[str],
    rho_natural: Sequence[float],
    rho_random: Sequence[float],
    bias_natural: Sequence[float],
    bias_random: Sequence[float],
    g_by_bin: Dict[str, Sequence[float]],
    i_by_bin: Dict[str, Sequence[float]],
    err_by_t: Dict[int, Tuple[Sequence[float], Sequence[float]]],
    coverage_lambda: Sequence[Tuple[float, float]],
    trigger_by_bin: Sequence[float],
    closer_by_bin: Sequence[float],
    out_path: str,
    title: str = "",
) -> str:
    """Render the six experiment-2 panels from pre-aggregated arrays."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    x = np.arange(len(bins))

    ax = axes[0]
    ax.plot(x, rho_natural, marker="o", label="natural shift N")
    ax.plot(x, rho_random, marker="s", label="random shift R")
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=8)
    ax.set_ylabel("Spearman rho")
    ax.set_title("(a) rank quality by teammate deviation", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    width = 0.38
    ax.bar(x - width / 2, bias_natural, width, label="natural shift N")
    ax.bar(x + width / 2, bias_random, width, label="random shift R")
    ax.axhline(0.0, color="grey", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=8)
    ax.set_ylabel("(Q^off - G) / R_max")
    ax.set_title("(b) normalised bias by teammate deviation", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    width = 0.38
    g_means = [float(np.mean(v)) if len(v) else float("nan") for v in g_by_bin.values()]
    i_means = [float(np.mean(v)) if len(v) else float("nan") for v in i_by_bin.values()]
    ax.bar(x - width / 2, g_means, width, label="g (single transplant)")
    ax.bar(x + width / 2, i_means, width, label="I (pair interaction)")
    ax.axhline(0.0, color="grey", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=8)
    ax.set_ylabel("normalised return difference")
    ax.set_title("(c) transplant gap and pairwise synergy", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    for t, (dev, err) in sorted(err_by_t.items()):
        order = np.argsort(dev)
        ax.plot(np.asarray(dev)[order], np.asarray(err)[order], marker="o", markersize=3,
                label=f"t={t // 1000}k")
    ax.set_xlabel("team deviation D(s)")
    ax.set_ylabel("|Q^off - G|")
    ax.set_title("(d) offline-value error versus deviation", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[4]
    if coverage_lambda:
        lam = np.asarray([p[0] for p in coverage_lambda])
        cov = np.asarray([p[1] for p in coverage_lambda])
        order = np.argsort(lam)
        ax.plot(lam[order], cov[order], marker="o", markersize=3)
    ax.set_xlabel("lambda(t)")
    ax.set_ylabel("in-coverage fraction P(D <= q95)")
    ax.set_title("(e) coverage versus memory weight", fontsize=10)
    ax.axvspan(0.0, 0.2, color="red", alpha=0.08)
    ax.axvspan(0.8, 1.0, color="blue", alpha=0.08)
    ax.annotate("stale knowledge retained", xy=(0.03, 0.05), xycoords="axes fraction",
                fontsize=7, color="darkred")
    ax.annotate("valid knowledge released", xy=(0.62, 0.85), xycoords="axes fraction",
                fontsize=7, color="darkblue")
    ax.grid(alpha=0.3)

    ax = axes[5]
    ax.bar(x, trigger_by_bin, 0.6, label="memory trigger rate")
    ax.plot(x, closer_by_bin, marker="o", color="darkorange",
            label="memory target closer to G")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("(f) memory constraint by deviation bin", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = _out(out_path)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
