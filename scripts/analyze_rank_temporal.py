#!/usr/bin/env python3
"""Aggregate a task-2 temporal rank-reliability measurement into report tables.

Reads a merged `logs/rank_temporal_<tag>.json` (written by
scripts/run_rank_temporal_auto.sh via scripts/verify_rank_temporal.py) and
prints the tables the report needs:

  * per continuation arm: concordance with trajectory-clustered 95% CI,
    algorithm pairs vs diagnostic pairs, tie counts;
  * breakdowns by pair kind, |dQ| bin, episode phase and rollout wave;
  * paired differences against the teacher arm (and between fork arms), with
    the fraction of pairs whose verdict flips between arms;
  * branch-return gaps as background (how far each continuation policy's
    realised returns sit from the teacher's own on the same forced branches).

The schema is identical for every batch, so batch 2 reuses this script
unchanged: point --in at the new merged json and --out at a new archive.

Usage:
  python scripts/analyze_rank_temporal.py \
      --in logs/rank_temporal_batch1.json \
      --out logs/rank_temporal_batch1_analysis.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_rank_reliability import bucket, clustered_rate

QGAP_EDGES = [0, 0.1, 1, 5, 20, 1e9]
PHASE = lambda r: "early" if r["frac"] < 0.34 else ("mid" if r["frac"] < 0.67 else "late")


def _fmt(stat: dict | None) -> str:
    if not stat or stat.get("concordance") is None:
        return "n/a"
    ci = "[%.3f, %.3f]" % tuple(stat["ci95"]) if stat.get("ci95") else ""
    return "%.3f %s (n=%d, ties=%d)" % (stat["concordance"], ci, stat["n_decided"], stat["n_tie"])


def _pair_key(r: dict) -> tuple:
    # t (probe step) is part of state identity: two probe states in the same
    # seed:wave cluster can produce the same diagnostic pair otherwise.
    return (r["cluster"], r["t"], r["kind"], tuple(r["joint_a"]), tuple(r["joint_b"]))


def _verdict(rows: list[dict]) -> dict:
    return {_pair_key(r): r["outcome"] for r in rows}


def paired_diff(rows_a: list[dict], rows_b: list[dict]) -> dict | None:
    """Per-cluster paired difference (B - A) over identical state/pair keys."""
    da, db = _verdict(rows_a), _verdict(rows_b)
    common = [k for k in da.keys() & db.keys() if da[k] in ("agree", "invert") and db[k] in ("agree", "invert")]
    if not common:
        return None
    per: dict[str, list[float]] = defaultdict(list)
    for k in common:
        va = 1.0 if da[k] == "agree" else 0.0
        vb = 1.0 if db[k] == "agree" else 0.0
        per[k[0]].append(vb - va)
    means = [sum(v) / len(v) for v in per.values()]
    n = sum(len(v) for v in per.values())
    m = sum(means) / len(means)
    var = sum((x - m) ** 2 for x in means) / (len(means) - 1) if len(means) > 1 else 0.0
    se = math.sqrt(var / len(means))
    flip = sum(1 for k in common if da[k] != db[k]) / len(common)
    return {"n": n, "n_clusters": len(means), "diff": m, "se": se,
            "ci95": [m - 1.96 * se, m + 1.96 * se], "flip": flip}


def _gap_stat(rows: list[dict]) -> str:
    gaps = sorted(r["abs_return_gap"] for r in rows)
    if not gaps:
        return "n/a"
    p90 = gaps[min(len(gaps) - 1, int(0.9 * len(gaps)))]
    return "median %.2f, p90 %.2f, max %.2f" % (gaps[len(gaps) // 2], p90, gaps[-1])


def build_report(doc: dict) -> str:
    recs = doc["records"]
    arms = [c["tag"] for c in doc["conts"]]
    cont_t = {c["tag"]: c.get("t") for c in doc["conts"]}
    offline = {"A", "B"}
    by_arm = {a: [r for r in recs if r["cont"] == a] for a in arms}
    algo = {a: [r for r in by_arm[a] if r["kind"] == "algorithm"] for a in arms}

    out: list[str] = []
    w = out.append
    w("task-2 temporal rank reliability: %s" % doc.get("mode", "?"))
    w("frozen teacher: %s" % doc.get("frozen_checkpoint"))
    w("records=%d  arms=%s" % (len(recs), arms))
    w("")

    w("== 1. concordance per continuation arm (trajectory-clustered 95% CI) ==")
    w("%-9s %-9s %-11s %s" % ("arm", "cont_t", "all pairs", "algorithm pairs"))
    for a in arms:
        w("%-9s %-9s %-11s %s" % (a, cont_t.get(a), _fmt(clustered_rate(by_arm[a])),
                                  _fmt(clustered_rate(algo[a]))))
    w("")
    w("online arms only, sorted by training step:")
    online = sorted((a for a in arms if a not in offline), key=lambda a: (cont_t.get(a) or 0))
    for a in online:
        w("  t=%-9s %-9s %s" % (cont_t.get(a), a, _fmt(clustered_rate(algo[a]))))
    w("")

    w("== 2. algorithm pairs by |dQ| bin ==")
    bins_raw = sorted(set(bucket(r["abs_q_gap"], QGAP_EDGES) for r in algo[arms[0]]),
                      key=lambda s: QGAP_EDGES.index(float(s.split(",")[0][1:])) if s.startswith("[") else 99)
    w("%-14s %s" % ("|dQ| bin", " ".join("%-9s" % a for a in arms)))
    for b in bins_raw:
        cells = []
        for a in arms:
            s = clustered_rate([r for r in algo[a] if bucket(r["abs_q_gap"], QGAP_EDGES) == b])
            cells.append("%-9s" % ("%.3f" % s["concordance"] if s.get("concordance") is not None else "n/a"))
        w("%-14s %s" % (b.replace("1000000000.0", "inf"), " ".join(cells)))
    w("")

    w("== 3. all pairs by kind ==")
    kinds = ["algorithm", "top", "long", "small_gap", "large_gap"]
    w("%-11s %s" % ("kind", " ".join("%-9s" % a for a in arms)))
    for k in kinds:
        cells = []
        for a in arms:
            s = clustered_rate([r for r in by_arm[a] if r["kind"] == k])
            cells.append("%-9s" % ("%.3f" % s["concordance"] if s.get("concordance") is not None else "n/a"))
        w("%-11s %s" % (k, " ".join(cells)))
    w("")

    w("== 4. all pairs by episode phase and wave ==")
    for name, key_fn in (("phase", PHASE), ("wave", lambda r: r["wave"])):
        groups = sorted({key_fn(r) for r in recs})
        w("%-11s %s" % (name, " ".join("%-9s" % a for a in arms)))
        for g in groups:
            cells = []
            for a in arms:
                s = clustered_rate([r for r in by_arm[a] if key_fn(r) == g])
                cells.append("%-9s" % ("%.3f" % s["concordance"] if s.get("concordance") is not None else "n/a"))
            w("%-11s %s" % (g, " ".join(cells)))
        w("")

    w("== 5. paired differences vs teacher arm A (algorithm pairs) ==")
    w("%-9s %-22s %-24s %s" % ("arm", "diff (B-A) 95% CI", "flip rate vs A", "n pairs"))
    for a in arms:
        if a == "A":
            continue
        pd = paired_diff(by_arm["A"], by_arm[a])
        if pd is None:
            w("%-9s n/a" % a)
            continue
        w("%-9s %+0.3f [%+0.3f, %+0.3f]%14.1f%%      %d"
          % (a, pd["diff"], pd["ci95"][0], pd["ci95"][1], 100 * pd["flip"], pd["n"]))
    w("")
    w("== 5b. paired differences between online arms (algorithm pairs) ==")
    w("%-26s %-22s %s" % ("pair", "diff 95% CI", "flip rate"))
    cont_path = {c["tag"]: c.get("path") or "" for c in doc["conts"]}
    fork_pairs = [("t0", "fork1M"), ("fork1M", "cont3M"), ("fork1M", "off3M"),
                  ("fork1M", "decay3M"), ("cont3M", "off3M"), ("cont3M", "decay3M"),
                  ("off3M", "decay3M")]
    # Main-line arms are the periodic checkpoints of the standalone UOT online
    # run; consecutive pairs along that line (plus its ends against the fork
    # arms) show whether reliability drifts with training on the main line.
    main_arms = sorted((a for a in online if "uot_temporal" in cont_path.get(a, "")),
                       key=lambda a: (cont_t.get(a) or 0))
    main_pairs = list(zip(main_arms, main_arms[1:]))
    if main_arms and "t0" in by_arm:
        main_pairs.append(("t0", main_arms[0]))
    for a in main_arms:
        for b in online:
            if b not in main_arms and b != "t0":
                main_pairs.append((a, b))
    seen_pairs = set()
    for x, y in fork_pairs + main_pairs:
        if x not in by_arm or y not in by_arm or (x, y) in seen_pairs:
            continue
        seen_pairs.add((x, y))
        pd = paired_diff(by_arm[x], by_arm[y])
        if pd is None:
            continue
        w("%-26s %+0.3f [%+0.3f, %+0.3f]%14.1f%%"
          % ("%s vs %s" % (y, x), pd["diff"], pd["ci95"][0], pd["ci95"][1], 100 * pd["flip"]))
    w("")

    w("== 6. background: branch-return gaps per arm (|G_a - G_b| over all pairs) ==")
    for a in arms:
        w("%-9s %s" % (a, _gap_stat(by_arm[a])))
    w("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", action="append", required=True,
                    help="merged rank_temporal_<tag>.json (repeatable; later files add arms)")
    ap.add_argument("--out", default=None, help="archive path; prints to stdout when omitted")
    args = ap.parse_args()

    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.inp]
    if len(docs) == 1:
        doc = docs[0]
    else:
        # Same frozen teacher / states / action pairs across batches, so merging is
        # a concatenation; the first doc supplies the shared header fields.
        doc = dict(docs[0])
        doc["conts"] = [c for d in docs for c in d["conts"]]
        doc["records"] = [r for d in docs for r in d["records"]]
    text = build_report(doc)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print("wrote %s" % args.out)
    print(text)


if __name__ == "__main__":
    main()
