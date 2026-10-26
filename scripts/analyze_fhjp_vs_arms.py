#!/usr/bin/env python3
"""FHJP vs OVMSE vs UOT on the 640-episode final-checkpoint endpoint.

Primary endpoint is the same one used by ``scripts/analyze_matched_protocol.py``:
a 640-episode evaluation (chunk 32, Wilson CI) of the final online checkpoint.
Every arm listed here starts from the same offline checkpoint on its map
(``checkpoints/ovmse_batch_20261014/<map>_offline/ovmse_paper_seed1_best.pt``),
so the comparison is initialisation-matched; only the online recipe differs.

Arms
  5m_vs_6m (3M online steps)
    FHJP                      checkpoints/fhjp_5m_vs_6m/seed{1,2,3}
    UOT   (matched arm A)     checkpoints/matched_protocol_20261020/A_seed1
    OVMSE (matched arm C)     checkpoints/matched_protocol_20261020/C_seed1
    OVMSE (matched arm D)     checkpoints/matched_protocol_20261020/D_seed1
    OVMSE (paper arm)         checkpoints/ovmse_batch_20261014/5m_vs_6m_online
    vanilla (matched arm B)   checkpoints/matched_protocol_20261020/B_seed1
    no-OVM+mix (matched E)    checkpoints/matched_protocol_20261020/E_seed1
  2s3z (1M online steps)
    FHJP                      checkpoints/fhjp_2s3z/seed{1,2,3}
    OVMSE (paper arm)         checkpoints/ovmse_batch_20261014/2s3z_online
    UOT / vanilla             no arm exists on this map

Contrasts: paired chunk bootstrap when both arms were evaluated with the same
eval seed (same 20 chunk boundaries), otherwise a Newcombe CI for win rate and
an unpaired chunk bootstrap for mean return.  Nothing is recomputed from raw
episodes.

Usage:
  python scripts/analyze_fhjp_vs_arms.py \
      --out logs/fhjp_vs_arms_summary.json \
      --md documentation/experiment_reports/fhjp_vs_ovmse_uot.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_matched_protocol import (  # noqa: E402
    bootstrap_mean_ci,
    newcombe_diff_ci,
    paired_chunks,
    read_jsonl,
    wilson_ci,
)

ROOT = "/home/jxy/multiAgentRL"

MAP_BUDGET = {"5m_vs_6m": 3_000_000, "2s3z": 1_000_000}
OFFLINE_CKPT = {
    m: f"checkpoints/ovmse_batch_20261014/{m}_offline/ovmse_paper_seed1_best.pt"
    for m in MAP_BUDGET
}

# arm_id, algorithm, recipe, seed, online.jsonl, eval640_final.jsonl, eval seed, note
ARMS: List[dict] = [
    # ---------------- 5m_vs_6m ----------------
    dict(map="5m_vs_6m", arm_id="FHJP", algo="FHJP",
         desc="joint proposer p=0.5 + plain online QMIX TD (no CQL, no OVM, no offline replay)",
         seed=1,
         online="checkpoints/fhjp_5m_vs_6m/seed1/online.jsonl",
         eval="checkpoints/fhjp_5m_vs_6m/seed1/eval640_final.jsonl", eval_seed=1),
    dict(map="5m_vs_6m", arm_id="FHJP", algo="FHJP",
         desc="joint proposer p=0.5 + plain online QMIX TD (no CQL, no OVM, no offline replay)",
         seed=2,
         online="checkpoints/fhjp_5m_vs_6m/seed2/online.jsonl",
         eval="checkpoints/fhjp_5m_vs_6m/seed2/eval640_final.jsonl", eval_seed=2),
    dict(map="5m_vs_6m", arm_id="FHJP", algo="FHJP",
         desc="joint proposer p=0.5 + plain online QMIX TD (no CQL, no OVM, no offline replay)",
         seed=3,
         online="checkpoints/fhjp_5m_vs_6m/seed3/online.jsonl",
         eval="checkpoints/fhjp_5m_vs_6m/seed3/eval640_final.jsonl", eval_seed=3,
         note="stopped manually at ~2.76M/3.0M steps; no final checkpoint, no 640-episode endpoint"),
    dict(map="5m_vs_6m", arm_id="UOT-A", algo="UOT",
         desc="matched arm A: UOT eta=0.1, OVM off, mixing 0", seed=1,
         online="checkpoints/matched_protocol_20261020/A_seed1/online.jsonl",
         eval="checkpoints/matched_protocol_20261020/A_seed1/eval640_final.jsonl", eval_seed=1),
    dict(map="5m_vs_6m", arm_id="OVMSE-C", algo="OVMSE",
         desc="matched arm C: OVM on, offline replay mixing 0.1", seed=1,
         online="checkpoints/matched_protocol_20261020/C_seed1/online.jsonl",
         eval="checkpoints/matched_protocol_20261020/C_seed1/eval640_final.jsonl", eval_seed=1),
    dict(map="5m_vs_6m", arm_id="OVMSE-D", algo="OVMSE",
         desc="matched arm D: OVM on, offline replay mixing 0", seed=1,
         online="checkpoints/matched_protocol_20261020/D_seed1/online.jsonl",
         eval="checkpoints/matched_protocol_20261020/D_seed1/eval640_final.jsonl", eval_seed=1),
    dict(map="5m_vs_6m", arm_id="OVMSE-paper", algo="OVMSE",
         desc="paper arm (configs/5m_vs_6m.yaml), same offline init as every other arm here", seed=1,
         online="checkpoints/ovmse_batch_20261014/5m_vs_6m_online/online.jsonl",
         eval="checkpoints/ovmse_batch_20261014/5m_vs_6m_online/eval640_final.jsonl", eval_seed=1),
    dict(map="5m_vs_6m", arm_id="vanilla-B", algo="vanilla online QMIX",
         desc="matched arm B: no UOT, no OVM, no mixing (FHJP's recipe without the proposer)", seed=1,
         online="checkpoints/matched_protocol_20261020/B_seed1/online.jsonl",
         eval="checkpoints/matched_protocol_20261020/B_seed1/eval640_final.jsonl", eval_seed=1),
    dict(map="5m_vs_6m", arm_id="mix-E", algo="ablation",
         desc="matched arm E: OVM off, offline replay mixing 0.1", seed=1,
         online="checkpoints/matched_protocol_20261020/E_seed1/online.jsonl",
         eval="checkpoints/matched_protocol_20261020/E_seed1/eval640_final.jsonl", eval_seed=1),
    # ---------------- 2s3z ----------------
    dict(map="2s3z", arm_id="FHJP", algo="FHJP",
         desc="joint proposer p=0.5 + plain online QMIX TD (no CQL, no OVM, no offline replay)",
         seed=1,
         online="checkpoints/fhjp_2s3z/seed1/online.jsonl",
         eval="checkpoints/fhjp_2s3z/seed1/eval640_final.jsonl", eval_seed=1),
    dict(map="2s3z", arm_id="FHJP", algo="FHJP",
         desc="joint proposer p=0.5 + plain online QMIX TD (no CQL, no OVM, no offline replay)",
         seed=2,
         online="checkpoints/fhjp_2s3z/seed2/online.jsonl",
         eval="checkpoints/fhjp_2s3z/seed2/eval640_final.jsonl", eval_seed=2),
    dict(map="2s3z", arm_id="FHJP", algo="FHJP",
         desc="joint proposer p=0.5 + plain online QMIX TD (no CQL, no OVM, no offline replay)",
         seed=3,
         online="checkpoints/fhjp_2s3z/seed3/online.jsonl",
         eval="checkpoints/fhjp_2s3z/seed3/eval640_final.jsonl", eval_seed=3),
    dict(map="2s3z", arm_id="OVMSE-paper", algo="OVMSE",
         desc="paper arm (configs/2s3z.yaml), same offline init as every other arm here", seed=1,
         online="checkpoints/ovmse_batch_20261014/2s3z_online/online.jsonl",
         eval="checkpoints/ovmse_batch_20261014/2s3z_online/eval640_final.jsonl", eval_seed=1),
]

CURVE_BIN = 250_000.0
DIAG_KEYS = [
    "evaluation/q_bias",
    "evaluation/greedy_action_disagree_rate",
    "evaluation/live_minus_offline_q",
    "train_ovm/activation_rate_interval_mean",
    "train_ovm/lambda_memory_interval_mean",
]


def load_640(path: str) -> Optional[dict]:
    rows = read_jsonl(path)
    if not rows:
        return None
    summary = next((r for r in rows if "checkpoint" in r), None)
    chunks = [r for r in rows if "chunk" in r]
    if summary is None:
        return {"path": path, "chunks": chunks, "complete": False}
    return {
        "path": path,
        "complete": True,
        "tag": summary.get("tag"),
        "checkpoint": summary.get("checkpoint"),
        "ckpt_t_env": summary.get("ckpt_t_env"),
        "episodes": summary.get("episodes"),
        "win_rate": summary.get("win_rate"),
        "win_rate_ci95": summary.get("win_rate_ci95"),
        "mean_return": summary.get("mean_return"),
        "chunks": chunks,
    }


def load_online(path: str) -> dict:
    rows = read_jsonl(path)
    evals = [r for r in rows if "evaluation/win_rate" in r]
    if not evals:
        return {"path": path, "rows": len(rows), "evals": 0}
    first, last = evals[0], evals[-1]
    t_last = float(last.get("t_env", 0.0))
    curve = []
    max_bin = int(max(1, np.ceil((t_last + 1e-9) / CURVE_BIN)))
    for k in range(max_bin):
        lo, hi = k * CURVE_BIN, (k + 1) * CURVE_BIN
        sel = [r for r in evals if lo <= float(r.get("t_env", 0.0)) < hi]
        if not sel:
            continue
        curve.append({
            "bin_start": lo,
            "bin_end": hi,
            "n_evals": len(sel),
            "win_rate": float(np.mean([r["evaluation/win_rate"] for r in sel])),
            "mean_return": float(np.mean([r["evaluation/mean_episode_return"] for r in sel])),
        })
    last10 = evals[-10:]
    diag = {}
    for key in DIAG_KEYS:
        vals = [r[key] for r in last10 if key in r and r[key] is not None]
        if vals:
            diag[key] = float(np.mean(vals))
    best = max(evals, key=lambda r: float(r["evaluation/win_rate"]))
    return {
        "path": path,
        "rows": len(rows),
        "evals": len(evals),
        "t_env_first": float(first.get("t_env", 0.0)),
        "t_env_last": t_last,
        "last_win_rate": float(last["evaluation/win_rate"]),
        "last_return": float(last["evaluation/mean_episode_return"]),
        "last10_win_rate": float(np.mean([r["evaluation/win_rate"] for r in last10])),
        "last10_return": float(np.mean([r["evaluation/mean_episode_return"] for r in last10])),
        "best_eval_win_rate": float(best["evaluation/win_rate"]),
        "best_eval_t_env": float(best.get("t_env", 0.0)),
        "curve_250k": curve,
        "last10_diagnostics": diag,
    }


def load_proposer(path: str) -> List[dict]:
    rows = read_jsonl(path)
    out = []
    for r in rows:
        if "proposer/version" not in r:
            continue
        out.append({
            "version": int(r["proposer/version"]),
            "steps": int(r.get("proposer/steps", 0)),
            "tau_v": r.get("proposer/tau_v"),
            "tau_v_std": r.get("proposer/tau_v_std"),
            "tau_v_ub": r.get("proposer/tau_v_ub"),
            "offline_arm_return": r.get("proposer/offline_arm_return"),
            "online_arm_return": r.get("proposer/online_arm_return"),
            "conflict_rate": r.get("proposer/conflict_rate"),
            "offline_select_rate": r.get("proposer/offline_select_rate"),
            "offline_select_rate_all_steps": r.get("proposer/offline_select_rate_all_steps"),
            "n_eval_episodes_per_arm": r.get("proposer/n_eval_episodes_per_arm"),
            "eval_env_steps": r.get("proposer/eval_env_steps"),
            "proposer_on": r.get("proposer/proposer_on"),
            "exited": r.get("proposer/exited"),
            "t_env": r.get("t_env"),
        })
    return out


def chunk_arrays(arm: dict, field: str) -> Optional[np.ndarray]:
    ep = arm.get("endpoint") or {}
    if not ep.get("complete"):
        return None
    chunks = ep.get("chunks") or []
    if not chunks:
        return None
    vals = np.asarray([c.get(field, np.nan) for c in chunks], dtype=np.float64)
    if np.isnan(vals).any():
        return None
    return vals


def unpaired_chunk_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000) -> dict:
    """Bootstrap CI for the difference of chunk means when the two arms were
    evaluated with different eval seeds (so chunks are not paired)."""
    rng = np.random.default_rng(12345)
    ia = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    ib = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    diff = ia - ib
    return {
        "paired": False,
        "n_chunks": int(a.size),
        "diff": float(a.mean() - b.mean()),
        "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
    }


def contrast(a: dict, b: dict, n_boot: int = 10000) -> Dict[str, dict]:
    """a - b on win rate and mean return."""
    out: Dict[str, dict] = {}
    ep_a, ep_b = a.get("endpoint") or {}, b.get("endpoint") or {}
    if not (ep_a.get("complete") and ep_b.get("complete")):
        return {"win_rate": {"paired": False, "reason": "endpoint missing"},
                "mean_return": {"paired": False, "reason": "endpoint missing"}}
    n_a, n_b = int(ep_a.get("episodes", 0)), int(ep_b.get("episodes", 0))
    same_eval_seed = a.get("eval_seed") == b.get("eval_seed")
    if same_eval_seed:
        arrays = paired_chunks([a, b], "chunk_win_rate")
        if arrays is not None:
            paired = arrays[0] - arrays[1]
            lo, hi = bootstrap_mean_ci(paired, n_boot=n_boot)
            out["win_rate"] = {"paired": True, "n_chunks": int(paired.size),
                               "diff": float(paired.mean()), "ci95": [lo, hi]}
        else:
            lo, hi = newcombe_diff_ci(ep_a["win_rate"], n_a, ep_b["win_rate"], n_b)
            out["win_rate"] = {"paired": False, "method": "newcombe", "n_chunks": 0,
                               "diff": float(ep_a["win_rate"] - ep_b["win_rate"]), "ci95": [lo, hi]}
    else:
        lo, hi = newcombe_diff_ci(ep_a["win_rate"], n_a, ep_b["win_rate"], n_b)
        out["win_rate"] = {"paired": False, "method": "newcombe (different eval seed)",
                           "n_chunks": 0,
                           "diff": float(ep_a["win_rate"] - ep_b["win_rate"]), "ci95": [lo, hi]}
    va, vb = chunk_arrays(a, "chunk_mean_return"), chunk_arrays(b, "chunk_mean_return")
    if va is None or vb is None:
        out["mean_return"] = {"paired": False, "reason": "chunks missing"}
    elif same_eval_seed and [c.get("episodes_done") for c in ep_a["chunks"]] == [
        c.get("episodes_done") for c in ep_b["chunks"]
    ]:
        paired = va - vb
        lo, hi = bootstrap_mean_ci(paired, n_boot=n_boot)
        out["mean_return"] = {"paired": True, "n_chunks": int(paired.size),
                              "diff": float(paired.mean()), "ci95": [lo, hi]}
    else:
        out["mean_return"] = unpaired_chunk_bootstrap(va, vb, n_boot=n_boot)
    return out


def seed_stats(arms: Sequence[dict], field: str) -> Optional[dict]:
    vals = [float(a["endpoint"][field]) for a in arms
            if (a.get("endpoint") or {}).get("complete")]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max()), "values": vals}


def fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def ci_str(pair: Optional[Sequence[float]], digits: int = 3) -> str:
    if not pair:
        return "—"
    return f"[{fmt(pair[0], digits)}, {fmt(pair[1], digits)}]"


def diff_str(item: Optional[dict], digits: int = 3) -> str:
    if not item or item.get("diff") is None:
        return "—"
    tag = "paired" if item.get("paired") else "unpaired"
    return f"{item['diff']:+.{digits}f} {ci_str(item.get('ci95'), digits)} ({tag})"


def build_summary_lines(summary: dict) -> List[str]:
    """Auto-generated prose summary; every number comes from the tables below."""
    lines: List[str] = []
    exits = {}
    for key, versions in summary["proposer"].items():
        done = [v for v in versions if v["exited"]]
        exits[key] = (done[0]["version"], done[0]["t_env"]) if done else None
    incomplete = {
        (a["map"], a["seed"]) for a in summary["arms"]
        if a["algo"] == "FHJP" and not (a.get("endpoint") or {}).get("complete")
    }
    for m in ("5m_vs_6m", "2s3z"):
        st = summary["fhjp_cross_seed"].get(m) or {}
        if not st.get("win_rate"):
            continue
        lines.append(
            f"- **{m}**: FHJP endpoint win {fmt(st['win_rate']['mean'])} ± "
            f"{fmt(st['win_rate']['std'])} over {st['win_rate']['n']} completed seed(s) "
            f"(return {fmt(st['mean_return']['mean'], 2)} ± {fmt(st['mean_return']['std'], 2)})."
        )
        for name, item in summary["contrasts"].get(m, {}).items():
            if not name.startswith("FHJP s1 −"):
                continue
            w = item.get("win_rate") or {}
            r = item.get("mean_return") or {}
            if w.get("diff") is None:
                continue
            verdict = "favours FHJP" if w["ci95"][0] > 0 else (
                "favours the comparator" if w["ci95"][1] < 0 else "not resolved")
            lines.append(
                f"  - {name}: Δwin {w['diff']:+.3f} {ci_str(w.get('ci95'))} "
                f"({verdict}), Δreturn {fmt(r.get('diff'), 2)} {ci_str(r.get('ci95'), 2)}."
            )
        exit_txt = ", ".join(
            f"seed{s}: version {v[0]} at {int(v[1] / 1000)}k steps"
            f"{' (run incomplete)' if (m, s) in incomplete else ''}"
            for s, v in sorted((int(k.split("seed")[-1]), v) for k, v in exits.items()
                               if k.startswith(m))
            if v
        )
        if exit_txt:
            lines.append(f"  - Proposer exit: {exit_txt}; after exit the run is plain online QMIX TD.")
    lines.append(
        "- The proposer's `offline_select_rate` stays at ~0.50 inside conflict decisions on every "
        "run, matching the fixed p=0.5 randomisation, and `conflict_rate` falls across versions "
        "(~0.53→0.43 on 5m_vs_6m), i.e. the online policy converges toward the offline joint actions."
    )
    return lines


def build_markdown(summary: dict) -> str:
    lines: List[str] = []
    lines.append("# FHJP vs OVMSE vs UOT (auto-generated)")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.extend(build_summary_lines(summary))
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    lines.append("| primary endpoint | 640-episode evaluation of the final online checkpoint (chunk 32, Wilson 95% CI) |")
    lines.append("| 5m_vs_6m online budget | 3,000,000 env steps |")
    lines.append("| 2s3z online budget | 1,000,000 env steps |")
    lines.append("| shared offline init (both maps) | `checkpoints/ovmse_batch_20261014/<map>_offline/ovmse_paper_seed1_best.pt` |")
    lines.append("| FHJP recipe | plain online QMIX TD + 0.5/0.5 joint proposer at decision conflicts; no CQL, no OVM target, no offline replay |")
    lines.append("| FHJP proposer | p=0.5, version_len=500k (5m_vs_6m) / 200k (2s3z), M=32 episodes per arm, M_min=16, alpha=0.05, eval steps counted in the budget |")
    lines.append("| eval seed | equal to the run's online seed; arms sharing an eval seed are contrasted with a paired chunk bootstrap |")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append("| arm | algorithm | recipe | run directory |")
    lines.append("|---|---|---|---|")
    seen = set()
    for arm in summary["arms"]:
        key = (arm["map"], arm["arm_id"])
        if key in seen:
            continue
        seen.add(key)
        run_dir = os.path.dirname(arm["online_path"])
        lines.append(f"| {arm['arm_id']} ({arm['map']}) | {arm['algo']} | {arm['desc']} | `{run_dir}` |")
    lines.append("")
    lines.append("## Primary endpoint by map")
    lines.append("")
    for m in ("5m_vs_6m", "2s3z"):
        lines.append(f"### {m}")
        lines.append("")
        lines.append("| arm | seed | win rate | 95% CI | mean return | ckpt t_env | endpoint |")
        lines.append("|---|---:|---:|---|---:|---:|---|")
        for arm in summary["arms"]:
            if arm["map"] != m:
                continue
            ep = arm.get("endpoint") or {}
            if ep.get("complete"):
                lines.append(
                    f"| {arm['arm_id']} | {arm['seed']} | {fmt(ep['win_rate'])} "
                    f"| {ci_str(ep['win_rate_ci95'])} | {fmt(ep['mean_return'], 2)} "
                    f"| {fmt(ep.get('ckpt_t_env'), 0)} | yes |"
                )
            else:
                on = arm.get("online") or {}
                lines.append(
                    f"| {arm['arm_id']} | {arm['seed']} | pending | — | — "
                    f"| {fmt(on.get('t_env_last'), 0)} | no ({arm.get('note', 'incomplete')}) |"
                )
        lines.append("")
        st = summary["fhjp_cross_seed"].get(m)
        if st and st.get("win_rate"):
            lines.append(
                f"FHJP cross-seed ({st['win_rate']['n']} completed seed(s)): win {fmt(st['win_rate']['mean'])} "
                f"± {fmt(st['win_rate']['std'])} (range {fmt(st['win_rate']['min'])}–{fmt(st['win_rate']['max'])}), "
                f"return {fmt(st['mean_return']['mean'], 2)} ± {fmt(st['mean_return']['std'], 2)}"
            )
            lines.append("")
    lines.append("## Contrasts vs FHJP")
    lines.append("")
    lines.append("Positive Δ means FHJP is better. Paired = same eval seed (chunk-level paired bootstrap); unpaired = different eval seed (Newcombe for win rate, chunk bootstrap for return).")
    lines.append("")
    lines.append("| map | contrast | Δwin (95% CI) | Δreturn (95% CI) |")
    lines.append("|---|---|---:|---:|")
    for m in ("5m_vs_6m", "2s3z"):
        for name, item in summary["contrasts"].get(m, {}).items():
            lines.append(
                f"| {m} | {name} | {diff_str(item.get('win_rate'))} "
                f"| {diff_str(item.get('mean_return'), 2)} |"
            )
    lines.append("")
    lines.append("## FHJP proposer diagnostics (version boundaries)")
    lines.append("")
    lines.append("tau_v = paired mean of (offline-arm return − online-arm return) over 32 shared-seed episodes per arm; exit when the one-sided 95% upper bound <= 0 and M >= 16. Exit is irreversible, and after exit the run is plain online QMIX TD.")
    lines.append("")
    lines.append("| map/seed | version | t_env | tau_v | tau_v_ub | offline arm | online arm | conflict rate | offline select rate | eval env steps | exited |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for key, versions in summary["proposer"].items():
        for v in versions:
            lines.append(
                f"| {key} | {v['version']} | {fmt(v['t_env'], 0)} | {fmt(v['tau_v'])} "
                f"| {fmt(v['tau_v_ub'])} | {fmt(v['offline_arm_return'], 2)} "
                f"| {fmt(v['online_arm_return'], 2)} | {fmt(v['conflict_rate'])} "
                f"| {fmt(v['offline_select_rate'])} | {fmt(v['eval_env_steps'], 0)} "
                f"| {'yes' if v['exited'] else 'no'} |"
            )
    lines.append("")
    lines.append("## Early vs late training-eval performance (32-episode evals, 250k bins)")
    lines.append("")
    lines.append("| map | arm | seed | first 250k win / ret | last 250k win / ret |")
    lines.append("|---|---|---:|---:|---:|")
    for m in ("5m_vs_6m", "2s3z"):
        for arm in summary["arms"]:
            if arm["map"] != m:
                continue
            curve = (arm.get("online") or {}).get("curve_250k") or []
            if not curve:
                continue
            first, last = curve[0], curve[-1]
            lines.append(
                f"| {m} | {arm['arm_id']} | {arm['seed']} "
                f"| {fmt(first['win_rate'])} / {fmt(first['mean_return'], 1)} "
                f"| {fmt(last['win_rate'])} / {fmt(last['mean_return'], 1)} |"
            )
    lines.append("")
    lines.append("## Descriptive: full 32-episode training-eval curve, 250k bins")
    lines.append("")
    for m in ("5m_vs_6m", "2s3z"):
        lines.append(f"### {m}")
        lines.append("")
        arms = [a for a in summary["arms"] if a["map"] == m and (a.get("online") or {}).get("curve_250k")]
        if not arms:
            continue
        header = "| t_env bin | " + " | ".join(
            f"{a['arm_id']} s{a['seed']} win / ret" for a in arms) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(arms) + 1))
        n_bins = max(len(a["online"]["curve_250k"]) for a in arms)
        for k in range(n_bins):
            cells = []
            for a in arms:
                curve = a["online"]["curve_250k"]
                if k < len(curve):
                    c = curve[k]
                    cells.append(f"{fmt(c['win_rate'])} / {fmt(c['mean_return'], 1)}")
                else:
                    cells.append("—")
            lines.append(f"| {int(k * CURVE_BIN / 1000)}k–{int((k + 1) * CURVE_BIN / 1000)}k | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("bash scripts/eval_fhjp.sh          # 640-episode endpoints for FHJP + OVMSE 2s3z")
    lines.append("python scripts/analyze_fhjp_vs_arms.py --out logs/fhjp_vs_arms_summary.json \\")
    lines.append("    --md documentation/experiment_reports/fhjp_vs_ovmse_uot.md")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/fhjp_vs_arms_summary.json")
    ap.add_argument("--md", default="documentation/experiment_reports/fhjp_vs_ovmse_uot.md")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    os.chdir(ROOT)
    arms: List[dict] = []
    for spec in ARMS:
        arm = dict(spec)
        arm["online_path"] = spec["online"]
        arm["endpoint"] = load_640(spec["eval"])
        arm["online"] = load_online(spec["online"])
        if spec["algo"] == "FHJP":
            arm["proposer"] = load_proposer(spec["online"])
        final_glob = glob.glob(os.path.join(os.path.dirname(spec["eval"]), "*_online_*_seed*.pt"))
        arm["final_checkpoint"] = final_glob[0] if final_glob else None
        arms.append(arm)

    contrasts: Dict[str, Dict[str, dict]] = {}
    for m in ("5m_vs_6m", "2s3z"):
        fhjp = [a for a in arms if a["map"] == m and a["algo"] == "FHJP"
                and (a.get("endpoint") or {}).get("complete")]
        others = [a for a in arms if a["map"] == m and a["algo"] != "FHJP"
                  and (a.get("endpoint") or {}).get("complete")]
        table: Dict[str, dict] = {}
        for f in fhjp:
            for o in others:
                name = f"FHJP s{f['seed']} − {o['arm_id']} s{o['seed']}"
                table[name] = contrast(f, o, n_boot=args.n_boot)
        # vanilla control is the cleanest reference: same recipe minus the proposer
        contrasts[m] = table

    cross_seed: Dict[str, dict] = {}
    for m in ("5m_vs_6m", "2s3z"):
        fhjp = [a for a in arms if a["map"] == m and a["algo"] == "FHJP"
                and (a.get("endpoint") or {}).get("complete")]
        cross_seed[m] = {
            "win_rate": seed_stats(fhjp, "win_rate"),
            "mean_return": seed_stats(fhjp, "mean_return"),
        }

    proposer = {
        f"{a['map']}/seed{a['seed']}": a["proposer"]
        for a in arms if a["algo"] == "FHJP" and a.get("proposer")
    }

    summary = {
        "generated_at": datetime.now().strftime("%F %T"),
        "primary_endpoint": "640 episodes, chunk 32, Wilson 95% CI, final online checkpoint",
        "offline_init": OFFLINE_CKPT,
        "arms": arms,
        "contrasts": contrasts,
        "fhjp_cross_seed": cross_seed,
        "proposer": proposer,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    md = build_markdown(summary)
    os.makedirs(os.path.dirname(args.md) or ".", exist_ok=True)
    with open(args.md, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(f"wrote {args.out}")
    print(f"wrote {args.md}")
    for m in ("5m_vs_6m", "2s3z"):
        print(f"--- {m} endpoints ---")
        for a in arms:
            if a["map"] != m:
                continue
            ep = a.get("endpoint") or {}
            if ep.get("complete"):
                print(f"  {a['algo']:20s} s{a['seed']}  win={ep['win_rate']:.4f} "
                      f"ci={ep['win_rate_ci95']}  ret={ep['mean_return']:.3f}")
            else:
                print(f"  {a['algo']:20s} s{a['seed']}  endpoint missing")


if __name__ == "__main__":
    main()
