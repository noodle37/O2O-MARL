"""Analysis of the offline joint-action ranking reliability results.

Reads logs/rank_reliability_full.json (per-pair records) and answers the five
questions of task 2 with cluster-robust (per trajectory) statistics:

  Q1 overall + stratified concordance vs. chance 0.5
  Q2 concordance vs. |dQ| bins
  Q3 concordance vs. the four support measures
  Q4 A (frozen seed1 greedy) vs. B (seed2 greedy) continuation, paired per pair
  Q5 controlling for |dQ|: joint conditional support vs. per-agent marginal support

Usage: python scripts/analyze_rank_reliability.py [--in logs/rank_reliability_full.json]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def cluster_stats(rows, key="outcome_agree"):
    """rows: list of dicts with 'cluster'. Returns mean, se, ci95, n, n_clusters."""
    per = defaultdict(list)
    for r in rows:
        per[r["cluster"]].append(1.0 if r[key] else 0.0)
    if not per:
        return None
    means = [sum(v) / len(v) for v in per.values()]
    n = sum(len(v) for v in per.values())
    m = sum(means) / len(means)
    if len(means) < 2:
        return {"n": n, "n_clusters": len(means), "mean": m, "se": None, "ci95": None}
    var = sum((x - m) ** 2 for x in means) / (len(means) - 1)
    se = math.sqrt(var / len(means))
    return {"n": n, "n_clusters": len(means), "mean": m, "se": se,
            "ci95": [m - 1.96 * se, m + 1.96 * se]}


def fmt(s):
    if s is None:
        return "n/a"
    ci = "[%.3f, %.3f]" % tuple(s["ci95"]) if s["ci95"] else ""
    return "n=%5d cl=%2d  %.3f  se=%.3f %s" % (s["n"], s["n_clusters"], s["mean"], s["se"] or 0, ci)


def paired_cluster_diff(rows_a, rows_b, key="outcome_agree"):
    """Pair rows by (cluster, cont-independent pair id) and diff A-B per cluster."""
    def idx(rows):
        d = {}
        for r in rows:
            d[(r["cluster"], r["kind"], tuple(r["joint_a"]), tuple(r["joint_b"]))] = 1.0 if r[key] else 0.0
        return d
    da, db = idx(rows_a), idx(rows_b)
    common = sorted(set(da) & set(db))
    per = defaultdict(list)
    for k in common:
        per[k[0]].append(db[k] - da[k])
    means = [sum(v) / len(v) for v in per.values()]
    n = sum(len(v) for v in per.values())
    m = sum(means) / len(means)
    var = sum((x - m) ** 2 for x in means) / (len(means) - 1) if len(means) > 1 else 0.0
    se = math.sqrt(var / len(means))
    return {"n_pairs": n, "n_clusters": len(means), "diff_B_minus_A": m, "se": se,
            "ci95": [m - 1.96 * se, m + 1.96 * se],
            "flip_rate": sum(1 for k in common if da[k] != db[k]) / max(len(common), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="logs/rank_reliability_full.json")
    args = ap.parse_args()

    d = json.load(open(args.inp))
    recs = [r for r in d["records"] if r["outcome"] in ("agree", "invert")]
    ties = len(d["records"]) - len(recs)
    for r in recs:
        r["outcome_agree"] = r["outcome"] == "agree"
        r["sup_min"] = r.get("sup_b") or r.get("sup_a") or {}
        r["pair_key"] = (r["cluster"], r["kind"], tuple(r["joint_a"]), tuple(r["joint_b"]))

    print("=" * 78)
    print("Q1  overall concordance (frozen seed1 critic ordering vs. true discounted G)")
    print("=" * 78)
    print("  decided pairs %d (ties dropped %d)" % (len(recs), ties))
    print("  ALL      ", fmt(cluster_stats(recs)))
    for k in ("algorithm", "top", "long", "small_gap", "large_gap"):
        sub = [r for r in recs if r["kind"] == k]
        print("  %-9s " % k, fmt(cluster_stats(sub)))
    for w in ("greedy", "epsgreedy"):
        sub = [r for r in recs if r["wave"] == w]
        print("  wave=%-10s" % w, fmt(cluster_stats(sub)))
    per_win = cluster_stats([r for r in recs if r["won"]])
    per_loss = cluster_stats([r for r in recs if not r["won"]])
    print("  win      ", fmt(per_win))
    print("  loss     ", fmt(per_loss))

    print()
    print("=" * 78)
    print("Q2  concordance by |dQ|")
    print("=" * 78)
    bins = [(0, 0.1), (0.1, 1), (1, 5), (5, 20), (20, 1e9)]
    for lo, hi in bins:
        sub = [r for r in recs if lo <= r["abs_q_gap"] < hi]
        print("  |dQ| in [%g,%g)  " % (lo, hi), fmt(cluster_stats(sub)))
    # Q2 within the algorithm pairs only (what UOT actually compares)
    algo = [r for r in recs if r["kind"] == "algorithm"]
    print("  -- algorithm pairs only --")
    for lo, hi in bins:
        sub = [r for r in algo if lo <= r["abs_q_gap"] < hi]
        print("  |dQ| in [%g,%g)  " % (lo, hi), fmt(cluster_stats(sub)))
    # Q2 within the diagnostic pairs only
    diag = [r for r in recs if r["kind"] != "algorithm"]
    print("  -- diagnostic pairs only --")
    for lo, hi in bins:
        sub = [r for r in diag if lo <= r["abs_q_gap"] < hi]
        print("  |dQ| in [%g,%g)  " % (lo, hi), fmt(cluster_stats(sub)))

    print()
    print("=" * 78)
    print("Q3  concordance by data support (min over the two actions of the pair)")
    print("=" * 78)
    supports = [
        ("global count", lambda r: min(r["sup_a"]["global_count"], r["sup_b"]["global_count"])),
        ("marginal global", lambda r: min(min(r["sup_a"]["marginal_global"]), min(r["sup_b"]["marginal_global"]))),
        ("state-cond joint", lambda r: min(r["sup_a"]["state_cond_joint_count"], r["sup_b"]["state_cond_joint_count"])),
        ("cond freq", lambda r: min(r["sup_a"]["cond_freq"], r["sup_b"]["cond_freq"])),
        ("marginal near(k=64)", lambda r: min(min(r["sup_a"]["marginal_near"]), min(r["sup_b"]["marginal_near"]))),
    ]
    for name, fn in supports:
        vals = sorted({fn(r) for r in recs})
        print("  %-20s distinct values: %s" % (name, vals[:8] + (["..."] if len(vals) > 8 else [])))
    for name, fn in supports:
        vals = [fn(r) for r in recs]
        if len(set(vals)) <= 1:
            print("  %-20s CONSTANT (%s) -> no explanatory power" % (name, vals[0] if vals else None))
            continue
        qs = sorted(vals)
        # terciles by value
        cuts = [qs[int(len(qs) / 3)], qs[int(2 * len(qs) / 3)]]
        for lo, hi, lab in ((-1, cuts[0], "low"), (cuts[0], cuts[1], "mid"), (cuts[1], 1e18, "high")):
            sub = [r for r in recs if lo < fn(r) <= hi]
            print("  %-20s %-5s (<=%s) " % (name, lab, hi if hi < 1e18 else "inf"), fmt(cluster_stats(sub)))

    print()
    print("=" * 78)
    print("Q4  continuation A (frozen seed1 greedy) vs B (seed2 greedy), paired")
    print("=" * 78)
    ra = [r for r in recs if r["cont"] == "A"]
    rb = [r for r in recs if r["cont"] == "B"]
    print("  A        ", fmt(cluster_stats(ra)))
    print("  B        ", fmt(cluster_stats(rb)))
    print("  paired   ", json.dumps(paired_cluster_diff(ra, rb)))
    # same, restricted to algorithm pairs
    ra2 = [r for r in ra if r["kind"] == "algorithm"]
    rb2 = [r for r in rb if r["kind"] == "algorithm"]
    print("  paired(algorithm only)", json.dumps(paired_cluster_diff(ra2, rb2)))
    # how big are the |G_A - G_B| differences on the same branch?
    ga = {(r["cluster"], tuple(r["joint_a"])): r["g_a"] for r in d["records"] if r["cont"] == "A"}
    gb = {(r["cluster"], tuple(r["joint_a"])): r["g_a"] for r in d["records"] if r["cont"] == "B"}
    common = sorted(set(ga) & set(gb))
    diffs = [abs(ga[k] - gb[k]) for k in common]
    diffs.sort()
    if diffs:
        print("  |G_A - G_B| on same branch: n=%d median=%.3f p90=%.3f max=%.3f"
              % (len(diffs), diffs[len(diffs) // 2], diffs[int(0.9 * len(diffs))], diffs[-1]))

    print()
    print("=" * 78)
    print("Q5  controlling for |dQ|: joint-conditional support vs per-agent marginal")
    print("=" * 78)
    joint_fn = lambda r: min(r["sup_a"]["state_cond_joint_count"], r["sup_b"]["state_cond_joint_count"])
    marg_fn = lambda r: min(min(r["sup_a"]["marginal_global"]), min(r["sup_b"]["marginal_global"]))
    glob_fn = lambda r: min(r["sup_a"]["global_count"], r["sup_b"]["global_count"])
    print("  joint support distribution:", end=" ")
    jc = defaultdict(int)
    for r in recs:
        jc[joint_fn(r)] += 1
    print(dict(sorted(jc.items())))
    # stratify by |dQ|, then compare support tertiles inside each stratum
    for lo, hi in [(0.1, 1), (1, 5), (5, 20), (20, 1e9)]:
        sub = [r for r in recs if lo <= r["abs_q_gap"] < hi]
        if len(sub) < 30:
            continue
        print("  -- |dQ| in [%g,%g), n=%d --" % (lo, hi, len(sub)))
        for name, fn in (("global count", glob_fn), ("marginal global", marg_fn)):
            vals = sorted(fn(r) for r in sub)
            if len(set(vals)) <= 1:
                print("     %-16s CONSTANT" % name)
                continue
            cuts = [vals[int(len(vals) / 3)], vals[int(2 * len(vals) / 3)]]
            for l2, h2, lab in ((-1, cuts[0], "low"), (cuts[0], cuts[1], "mid"), (cuts[1], 1e18, "high")):
                s2 = [r for r in sub if l2 < fn(r) <= h2]
                print("     %-16s %-5s " % (name, lab), fmt(cluster_stats(s2)))
        jz = [r for r in sub if joint_fn(r) == 0]
        jn = [r for r in sub if joint_fn(r) > 0]
        print("     %-16s ==0  " % "joint(state-cond)", fmt(cluster_stats(jz)) if jz else "n/a")
        print("     %-16s >0   " % "joint(state-cond)", fmt(cluster_stats(jn)) if jn else "n/a (no pairs with support)")

    print()
    print("=" * 78)
    print("extra: does the UOT margin |dQ|>=0.1 filter help?")
    print("=" * 78)
    below = [r for r in recs if r["abs_q_gap"] < 0.1]
    above = [r for r in recs if r["abs_q_gap"] >= 0.1]
    print("  |dQ|<0.1 (would be filtered) ", fmt(cluster_stats(below)))
    print("  |dQ|>=0.1 (kept)            ", fmt(cluster_stats(above)))
    print()
    print("extra: per-trajectory concordance spread (algorithm pairs, cont A)")
    pa = [r for r in ra if r["kind"] == "algorithm"]
    per = defaultdict(list)
    for r in pa:
        per[r["cluster"]].append(1.0 if r["outcome_agree"] else 0.0)
    rows = sorted((sum(v) / len(v), k, len(v)) for k, v in per.items())
    for m, k, n in rows:
        print("   %-16s n=%3d  %.3f" % (k, n, m))


if __name__ == "__main__":
    main()
