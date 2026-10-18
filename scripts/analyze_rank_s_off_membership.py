"""Follow-up analysis: does ranking reliability depend on being inside S_off?

The per-pair table in logs/rank_reliability_full.json records the two compared
joint actions but not their relation to the offline recorded set S_off. This
script rebuilds S_off from the vault (3283 recorded joints) and, for every
compared pair, marks whether each action is

  * exactly recorded (in S_off, Hamming 0),
  * within Hamming <= 2 of a recorded joint (the UOT candidate rule), or
  * outside (extrapolation).

and then reports concordance by that relation, plus a paired greedy vs
eps-greedy comparison that does not require identical probe steps.

Usage: python scripts/analyze_rank_support_smembership.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from scripts.verify_rank_reliability import ensure_project_root, resolve_path  # noqa: E402
from training.factory import load_vault_dataset  # noqa: E402
from uot.support import RecordedJoints  # noqa: E402


def cluster_stats(rows, key="agree"):
    per = defaultdict(list)
    for r in rows:
        per[r["cluster"]].append(1.0 if r[key] else 0.0)
    if not per:
        return None
    ms = [sum(v) / len(v) for v in per.values()]
    n = sum(len(v) for v in per.values())
    m = sum(ms) / len(ms)
    if len(ms) < 2:
        return {"n": n, "cl": len(ms), "m": m, "se": None}
    se = math.sqrt(sum((x - m) ** 2 for x in ms) / (len(ms) - 1) / len(ms))
    return {"n": n, "cl": len(ms), "m": m, "se": se}


def f(s):
    if s is None:
        return "n/a"
    if s["se"] is None:
        return "n=%5d cl=%2d  %.3f" % (s["n"], s["cl"], s["m"])
    return "n=%5d cl=%2d  %.3f  se=%.3f  ci=[%.3f,%.3f]" % (
        s["n"], s["cl"], s["m"], s["se"], s["m"] - 1.96 * s["se"], s["m"] + 1.96 * s["se"])


def main() -> None:
    ensure_project_root()
    cfg = OmegaConf.to_container(OmegaConf.load("configs/5m_vs_6m.yaml"), resolve=True)
    cfg["system"]["data_device"] = "cpu"
    cfg["system"]["device"] = "cpu"
    dataset = load_vault_dataset(cfg)
    n_agents = dataset.stats.n_agents
    n_actions = dataset.stats.n_actions
    rj = RecordedJoints.from_episode_actions(dataset.actions, dataset.filled, n_actions)
    codes = rj.sorted_codes.numpy()
    print("S_off size:", len(codes), " n_agents:", n_agents, " n_actions:", n_actions)
    del dataset

    import numpy as np

    mults = (n_actions ** np.arange(n_agents)).astype(np.int64)
    rec = codes.reshape(-1, 1) // mults % n_actions  # [M, N] decoded actions
    rec = rec.astype(np.int64)

    def decode(joint):
        c = int(np.dot(np.asarray(joint, dtype=np.int64), mults))
        return np.array([(c // mults[i]) % n_actions for i in range(n_agents)], dtype=np.int64)

    def min_hamming(joint):
        a = decode(joint)
        d = (rec != a[None, :]).sum(axis=1)
        return int(d.min())

    d = json.load(open("logs/rank_reliability_full.json"))
    recs = [r for r in d["records"] if r["outcome"] in ("agree", "invert")]
    for r in recs:
        r["agree"] = r["outcome"] == "agree"
        ha, hb = min_hamming(r["joint_a"]), min_hamming(r["joint_b"])
        r["ha"], r["hb"] = ha, hb
        r["hmin"] = min(ha, hb)
        r["in_s_off"] = ha == 0 or hb == 0
        r["both_s_off"] = (ha == 0) and (hb == 0)
        r["within2"] = max(ha, hb) <= 2

    print()
    print("=== relation of the compared pair to S_off (min Hamming to a recorded joint) ===")
    for lab, fn in [
        ("min Hamming of the pair", lambda r: r["hmin"]),
        ("max Hamming of the pair", lambda r: max(r["ha"], r["hb"])),
    ]:
        vals = sorted({fn(r) for r in recs})
        for v in vals:
            print("  %-24s = %d  " % (lab, v), f(cluster_stats([r for r in recs if fn(r) == v])))
    print("  both exactly recorded     ", f(cluster_stats([r for r in recs if r["both_s_off"]])))
    print("  at least one recorded     ", f(cluster_stats([r for r in recs if r["in_s_off"]])))
    print("  neither recorded          ", f(cluster_stats([r for r in recs if not r["in_s_off"]])))
    print("  both within H<=2 (UOT rule)", f(cluster_stats([r for r in recs if r["within2"]])))
    print("  some action H>2           ", f(cluster_stats([r for r in recs if not r["within2"]])))

    print()
    print("=== same split, restricted to diagnostic pairs (algorithm pairs are S_off by construction) ===")
    diag = [r for r in recs if r["kind"] != "algorithm"]
    print("  diag: at least one recorded", f(cluster_stats([r for r in diag if r["in_s_off"]])))
    print("  diag: neither recorded     ", f(cluster_stats([r for r in diag if not r["in_s_off"]])))
    for v in sorted({max(r["ha"], r["hb"]) for r in diag}):
        print("  diag: max Hamming = %d      " % v, f(cluster_stats([r for r in diag if max(r["ha"], r["hb"]) == v])))

    print()
    print("=== within S_off pairs only: |dQ| still matters ===")
    s = [r for r in recs if r["in_s_off"]]
    for lo, hi in [(0, 1), (1, 5), (5, 20), (20, 1e9)]:
        sub = [r for r in s if lo <= r["abs_q_gap"] < hi]
        if sub:
            print("  |dQ| in [%g,%g)  " % (lo, hi), f(cluster_stats(sub)))

    print()
    print("=== greedy vs eps-greedy (trajectory level, cont A, algorithm pairs) ===")
    for w in ("greedy", "epsgreedy"):
        sub = [r for r in recs if r["cont"] == "A" and r["kind"] == "algorithm" and r["wave"] == w]
        print("  %-10s" % w, f(cluster_stats(sub)))
    per = defaultdict(dict)
    for r in recs:
        if r["cont"] != "A" or r["kind"] != "algorithm":
            continue
        per[(r["seed"], r["wave"])].setdefault("n", 0)
        per[(r["seed"], r["wave"])]["n"] += 1
        per[(r["seed"], r["wave"])]["k"] = per[(r["seed"], r["wave"])].get("k", 0) + (1 if r["agree"] else 0)
    diffs = []
    for seed in sorted({k[0] for k in per}):
        g, e = per.get((seed, "greedy")), per.get((seed, "epsgreedy"))
        if g and e:
            diffs.append(e["k"] / e["n"] - g["k"] / g["n"])
    if diffs:
        m = sum(diffs) / len(diffs)
        se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs))
        print("  per-seed epsgreedy - greedy diff = %+.3f  se=%.3f  ci=[%.3f,%.3f] (n_seeds=%d)"
              % (m, se, m - 1.96 * se, m + 1.96 * se, len(diffs)))

    print()
    print("=== phase / alive units (fixed, all records) ===")
    for lab, lo, hi in [("early", 0, 0.3), ("mid", 0.3, 0.7), ("late", 0.7, 1.01)]:
        print("  %-6s" % lab, f(cluster_stats([r for r in recs if lo <= r["frac"] < hi])))
    for lab, key in [("alive_allies", "alive_allies"), ("alive_enemies", "alive_enemies")]:
        for v in sorted({r[key] for r in recs}):
            sub = [r for r in recs if r[key] == v]
            if len(sub) >= 30:
                print("  %s=%d" % (lab, v), f(cluster_stats(sub)))


if __name__ == "__main__":
    main()
