#!/usr/bin/env python3
"""Paired 640-episode comparisons for the TDR / PAD+TDR arms initialised from
the matched-protocol offline checkpoint.

Every eval640 run uses `scripts/eval_640.py` with `seed=<train seed>`, so two
runs with the same seed share the same 640 evaluation scenarios while runs with
different seeds do not.  This script only pairs runs with the same eval seed and
reports a paired chunk bootstrap (20 chunks of 32 episodes) so the comparison is
not diluted by the Wilson interval of a single arm.

Usage:
  python scripts/analyze_tdr_ovmse_init.py
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Dict, List, Tuple

BOOT = 10000
ALPHA = 0.05
SEED = 20261022

RUNS: Dict[str, str] = {
    "TDR_10Mfinal_s1": "checkpoints/uot_controls_5m_vs_6m/tdr/eval640.jsonl",
    "TDR_5Mbest_s1": "checkpoints/uot_tdr_ovmse_init_5m_vs_6m/tdr/eval640.jsonl",
    "TDR_5Mbest_s2": "checkpoints/uot_tdr_ovmse_init_5m_vs_6m/tdr_seed2/eval640.jsonl",
    "PADTDR_5Mbest_s1": "checkpoints/uot_pad_tdr_ovmse_init_5m_vs_6m/pad_tdr_seed1/eval640.jsonl",
    "PADTDR_5Mbest_s2": "checkpoints/uot_pad_tdr_ovmse_init_5m_vs_6m/pad_tdr_seed2/eval640.jsonl",
    "A_5Mbest_s1": "checkpoints/matched_protocol_20261020/A_seed1/eval640_final.jsonl",
    "B_5Mbest_s1": "checkpoints/matched_protocol_20261020/B_seed1/eval640_final.jsonl",
    "C_5Mbest_s1": "checkpoints/matched_protocol_20261020/C_seed1/eval640_final.jsonl",
    "D_5Mbest_s1": "checkpoints/matched_protocol_20261020/D_seed1/eval640_final.jsonl",
    "E_5Mbest_s1": "checkpoints/matched_protocol_20261020/E_seed1/eval640_final.jsonl",
}

PAIRS: List[Tuple[str, str, str]] = [
    ("TDR_10Mfinal_s1", "TDR_5Mbest_s1", "init effect on TDR (same seed/eval set)"),
    ("TDR_5Mbest_s1", "A_5Mbest_s1", "decay vs fixed ranking, 5.08M init"),
    ("TDR_5Mbest_s1", "B_5Mbest_s1", "decayed ranking vs no ranking"),
    ("A_5Mbest_s1", "B_5Mbest_s1", "fixed ranking vs no ranking"),
    ("TDR_5Mbest_s1", "PADTDR_5Mbest_s1", "joint vs per-agent, same schedule, seed 1"),
    ("TDR_5Mbest_s2", "PADTDR_5Mbest_s2", "joint vs per-agent, same schedule, seed 2"),
    ("PADTDR_5Mbest_s1", "B_5Mbest_s1", "per-agent decay vs no ranking, seed 1"),
    ("C_5Mbest_s1", "A_5Mbest_s1", "OVMSE vs UOT fixed ranking"),
    ("D_5Mbest_s1", "B_5Mbest_s1", "OVM alone vs no ranking"),
    ("E_5Mbest_s1", "B_5Mbest_s1", "mixing alone vs no ranking"),
]


def load(path: str) -> Tuple[List[float], float, str]:
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    chunks = [r["chunk_win_rate"] for r in rows if "chunk" in r]
    summary = [r for r in rows if "chunk" not in r]
    return chunks, summary[-1]["win_rate"], summary[-1].get("tag", "")


def paired(chunks_a: List[float], chunks_b: List[float]) -> Tuple[float, float, float]:
    n = min(len(chunks_a), len(chunks_b))
    diffs = [chunks_a[i] - chunks_b[i] for i in range(n)]
    rng = random.Random(SEED)
    boots = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo = boots[int(ALPHA / 2 * BOOT)]
    hi = boots[int((1 - ALPHA / 2) * BOOT) - 1]
    return sum(diffs) / n, lo, hi


def main() -> None:
    data = {}
    for name, path in RUNS.items():
        if os.path.isfile(path):
            data[name] = load(path)
    print("| arm | eval seed tag | 640-ep win rate | n chunks |")
    print("|---|---|---:|---:|")
    for name, (chunks, win, tag) in data.items():
        print(f"| {name} | {tag} | {win:.4f} | {len(chunks)} |")
    print()
    print("| comparison | Δwin (a−b) | 95% CI (paired chunk bootstrap) | note |")
    print("|---|---:|---|---|")
    for a, b, note in PAIRS:
        if b not in data or a not in data:
            continue
        if len(data[a][0]) != len(data[b][0]):
            continue
        d, lo, hi = paired(data[a][0], data[b][0])
        print(f"| {a} − {b} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {note} |")


if __name__ == "__main__":
    main()
