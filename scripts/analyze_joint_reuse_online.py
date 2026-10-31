"""Aggregate final 640-episode evaluations for the joint_reuse_online validation experiment.

Reads eval640_initial.jsonl / eval640_final.jsonl under
checkpoints/joint_reuse_online_5m_vs_6m/C_t250k/ and prints per-seed and
cross-seed summaries. Chunk-level (32-episode group) paired differences use
the t distribution with 19 degrees of freedom.
"""

import json
import math
from pathlib import Path

import scipy.stats as st

ROOT = Path("checkpoints/joint_reuse_online_5m_vs_6m/C_t250k")
SEEDS = [1, 2, 3]
BRANCHES = ["joint", "independent", "none"]


def load_eval(path: Path):
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    chunks = [l for l in lines if "chunk" in l]
    summary = [l for l in lines if "chunk" not in l][-1]
    assert len(chunks) == 20 and summary["episodes"] == 640, path
    return chunks, summary


def paired_ci(a, b):
    """Paired difference a - b: mean, 95% CI, and one-sided p (diff > 0)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    tcrit = st.t.ppf(0.975, n - 1)
    half = tcrit * sd / math.sqrt(n)
    tstat = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
    p_one_sided = 1 - st.t.cdf(tstat, n - 1)
    return mean, mean - half, mean + half, p_one_sided


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def main():
    data = {}
    for seed in SEEDS:
        data[seed] = {"initial": load_eval(ROOT / f"seed{seed}/eval640_initial.jsonl")}
        for br in BRANCHES:
            data[seed][br] = load_eval(ROOT / f"seed{seed}/{br}/eval640_final.jsonl")

    print("== Per-branch final 640-episode eval ==")
    header = f"{'seed':>4} {'branch':>12} {'mean_return':>11} {'win_rate':>9}"
    print(header)
    for seed in SEEDS:
        for br in ["initial"] + BRANCHES:
            _, s = data[seed][br]
            print(f"{seed:>4} {br:>12} {fmt(s['mean_return']):>11} {fmt(s['win_rate'], 4):>9}")

    print("\n== Paired chunk differences (20 groups of 32 episodes) ==")
    print(f"{'seed':>4} {'contrast':>16} {'dReturn':>9} {'CI95 low':>9} {'CI95 high':>9} {'p(>0)':>7} {'dWinRate':>9}")
    contrasts = [("joint", "independent"), ("joint", "none"), ("independent", "none")]
    per_seed_diffs = {c: [] for c in contrasts}
    per_seed_wdiffs = {c: [] for c in contrasts}
    for seed in SEEDS:
        for a, b in contrasts:
            ca, _ = data[seed][a]
            cb, _ = data[seed][b]
            ra = [c["chunk_mean_return"] for c in ca]
            rb = [c["chunk_mean_return"] for c in cb]
            wa = [c["chunk_win_rate"] for c in ca]
            wb = [c["chunk_win_rate"] for c in cb]
            m, lo, hi, p = paired_ci(ra, rb)
            wm, _, _, _ = paired_ci(wa, wb)
            per_seed_diffs[(a, b)].append(m)
            per_seed_wdiffs[(a, b)].append(wm)
            print(f"{seed:>4} {a + '-' + b:>16} {fmt(m):>9} {fmt(lo):>9} {fmt(hi):>9} {fmt(p, 4):>7} {fmt(wm, 4):>9}")

    print("\n== Cross-seed mean of per-seed paired differences (n=3 seeds) ==")
    print(f"{'contrast':>16} {'mean dRet':>10} {'sd':>7} {'mean dWin':>10}")
    for c in contrasts:
        ds = per_seed_diffs[c]
        m = sum(ds) / len(ds)
        sd = math.sqrt(sum((x - m) ** 2 for x in ds) / (len(ds) - 1))
        ws = per_seed_wdiffs[c]
        wm = sum(ws) / len(ws)
        print(f"{c[0] + '-' + c[1]:>16} {fmt(m):>10} {fmt(sd):>7} {fmt(wm, 4):>10}")

    print("\n== Final minus initial (same-seed 640-eval paired) ==")
    for seed in SEEDS:
        ci, si = data[seed]["initial"]
        for br in BRANCHES:
            cb, sb = data[seed][br]
            ri = [c["chunk_mean_return"] for c in ci]
            rb = [c["chunk_mean_return"] for c in cb]
            m, lo, hi, p = paired_ci(rb, ri)
            print(f"seed{seed} {br:>12}: dReturn={fmt(m)} CI=[{fmt(lo)},{fmt(hi)}] p(>0)={fmt(p,4)}")


if __name__ == "__main__":
    main()
