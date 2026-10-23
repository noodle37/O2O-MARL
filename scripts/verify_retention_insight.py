#!/usr/bin/env python3
"""H1/H2 of the retention-insight validation: counterfactual deviations and
candidate selectability, both by branch replay against a frozen teacher.

For every probe state the script builds three kinds of joint actions without
training anything:

  * one-agent deviations from the executed (successful) joint -- primary
    (most frequent legal alternative in the offline kNN neighbourhood), hard
    (highest frozen-Q alternative) and random;
  * random recorded joint pairs, the same-state global-ranking control;
  * three behaviour-proposed candidate sets (support_dev, support_neigh,
    random) for K-way selectability.

Every distinct joint is branch-replayed once: replay the recorded prefix
bit-for-bit, force the joint at the probe step, continue greedily, and take the
discounted return. Q-bar is used only afterwards as a scorer (except `hard`,
which is the deliberate strongest-Q-competitor test).

Nothing is trained; `parameter_updates` is 0 in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SC2PATH", "/home/jxy/StarCraftII")

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.factory import build_learner, load_vault_dataset, make_env  # noqa: E402
from utils.checkpoint import load_qmix_checkpoint  # noqa: E402
from utils.paths import ensure_project_root, resolve_path  # noqa: E402
from scripts.verify_rank_reliability import (  # noqa: E402
    SupportIndex,
    _legal_support,
    _mix_joint_q,
    branch_return,
    bucket,
    clustered_rate,
    diagnostic_pairs,
    pair_outcome,
)
from scripts.verify_local_rank_inversion import (  # noqa: E402
    action_counts,
    close_env,
    joint_key,
    support_tensor,
)
from scripts.retention_insight_core import (  # noqa: E402
    RETURN_TIE,
    behaviour_candidates,
    build_deviations,
    candidate_metrics,
    cluster_bootstrap_diff,
    clustered_mean,
    discounted_from,
    effect_bin,
    means_by,
    rates_by,
    random_candidates,
    sample_random_pairs,
    support_fields,
    teacher_rollout,
)


def select_probe_steps(length: int, fracs: list[float], max_probes: int) -> list[int]:
    steps = sorted(
        {int(min(max(round(float(f) * length), 1), length - 1)) for f in fracs}
    )
    return steps[: max(1, int(max_probes))]


def _dedup_joints(joints: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    seen = set()
    for joint in joints:
        key = joint_key(joint)
        if key in seen:
            continue
        seen.add(key)
        out.append(joint)
    return out


def _outcome_value(record: dict) -> float:
    if record.get("outcome") == "tie":
        return 0.0
    return 1.0 if record.get("outcome") == "agree" else 0.0


def _effect_counts(rows: list[dict]) -> dict:
    gaps = np.asarray([r["signed_return_gap"] for r in rows], dtype=np.float64)
    if gaps.size == 0:
        return {"n": 0}
    return {
        "n": int(gaps.size),
        "dev_better": int(np.sum(gaps < -RETURN_TIE)),
        "dev_worse": int(np.sum(gaps > RETURN_TIE)),
        "same": int(np.sum(np.abs(gaps) <= RETURN_TIE)),
        "mean_signed": float(gaps.mean()),
        "median_signed": float(np.median(gaps)),
    }


def measure_state(
    env,
    base: dict,
    t: int,
    frozen,
    sidx: SupportIndex,
    support_dev: torch.Tensor,
    args,
    gamma: float,
    seed: int,
    wave: str,
) -> dict:
    rec = base["steps"][t]
    q = rec["q"].to(frozen.device)
    legal = rec["legal"].to(frozen.device)
    state = rec["state"]
    base_key = rec["action"]
    prefix_actions = [base["steps"][k]["action"] for k in range(t)]
    near = sidx.nearest(state)
    feasible = _legal_support(support_dev, legal)
    rng = np.random.default_rng(seed * 100003 + (0 if wave == "greedy" else 7919) + t)
    cluster = f"{seed}:{wave}"
    base_tensor = torch.tensor(base_key, dtype=torch.long, device=frozen.device)

    # Same-state global diagnostic control (top, long, smallest |dQ|, largest
    # |dQ|) over every recorded joint that is legal here. Q is only a selector;
    # the outcomes still come from branch replay.
    diag_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if feasible.shape[0] >= 2:
        q_all = _mix_joint_q(frozen, q, feasible, state)
        order = torch.argsort(q_all, descending=True)
        ranked = feasible[order]
        ranked_q = q_all[order]
        for name, (ia, ib) in diagnostic_pairs(ranked_q).items():
            diag_pairs[name] = (
                ranked[ia].detach().cpu(),
                ranked[ib].detach().cpu(),
            )

    devs = build_deviations(
        base_key, legal, near, q, sidx, tuple(args.dev_kinds.split(",")), rng
    )
    random_pairs = sample_random_pairs(feasible, args.random_pairs, rng)
    candidate_sets: dict[str, list[torch.Tensor]] = {}
    if "support_dev" in args.candidate_sets:
        candidate_sets["support_dev"] = _dedup_joints(
            [base_tensor]
            + [joint for (kind, _agent), (joint, _alt) in devs.items() if kind == "primary"]
        )
    if "support_neigh" in args.candidate_sets:
        candidate_sets["support_neigh"] = behaviour_candidates(
            base_key,
            legal,
            near,
            sidx,
            k=args.cand_k,
            per_agent=args.per_agent,
            max_hamming=args.max_hamming,
            n_recorded=args.n_recorded,
        )
    if "random" in args.candidate_sets:
        candidate_sets["random"] = random_candidates(
            base_key, legal, support_dev, args.cand_k, rng
        )

    needed: dict[tuple[int, ...], torch.Tensor] = {base_key: base_tensor.detach().cpu()}
    for joint, _alt in devs.values():
        needed.setdefault(joint_key(joint), joint.detach().cpu())
    for joint_a, joint_b in diag_pairs.values():
        needed.setdefault(joint_key(joint_a), joint_a.detach().cpu())
        needed.setdefault(joint_key(joint_b), joint_b.detach().cpu())
    for joint_a, joint_b in random_pairs:
        needed.setdefault(joint_key(joint_a), joint_a.detach().cpu())
        needed.setdefault(joint_key(joint_b), joint_b.detach().cpu())
    for joints in candidate_sets.values():
        for joint in joints:
            needed.setdefault(joint_key(joint), joint.detach().cpu())

    all_keys = list(needed.keys())
    if args.max_joints and len(all_keys) > args.max_joints:
        priority: list[tuple[int, ...]] = [base_key]
        for joints in candidate_sets.values():
            priority.extend(joint_key(j) for j in joints)
        priority.extend(joint_key(j) for j, _alt in devs.values())
        priority.extend(
            k for pair in diag_pairs.values() for k in (joint_key(pair[0]), joint_key(pair[1]))
        )
        priority.extend(k for pair in random_pairs for k in (joint_key(pair[0]), joint_key(pair[1])))
        keep = []
        for key in priority:
            if key in needed and key not in keep:
                keep.append(key)
        for key in all_keys:
            if len(keep) >= args.max_joints:
                break
            if key not in keep:
                keep.append(key)
        all_keys = keep[: args.max_joints]

    g_of: dict[tuple[int, ...], float] = {}
    errors: dict[tuple[int, ...], str] = {}
    for key in all_keys:
        joint = needed[key]
        g, err = branch_return(env, frozen, prefix_actions, t, joint, state, gamma)
        if err is None and g is not None:
            g_of[key] = float(g)
        else:
            errors[key] = str(err)

    if base_key not in g_of:
        return {
            "state": None,
            "deviations": [],
            "random_pairs": [],
            "global_pairs": [],
            "selectability": [],
            "candidate_pairs": [],
        }

    q_keys = [key for key in all_keys if key in g_of]
    q_tot = _mix_joint_q(
        frozen, q, torch.stack([needed[k] for k in q_keys]).to(frozen.device), state
    )
    q_of = {key: float(q_tot[i]) for i, key in enumerate(q_keys)}

    state_fields = {
        "seed": seed,
        "wave": wave,
        "t": t,
        "L": base["length"],
        "frac": t / max(base["length"] - 1, 1),
        "won": base["won"],
        "alive_allies": rec["alive_allies"],
        "alive_enemies": rec["alive_enemies"],
        "n_full_cand": int(feasible.shape[0]),
        "cluster": cluster,
        "state_id": f"{seed}:{wave}:{t}",
        "base_key": list(base_key),
        "base_q": q_of[base_key],
        "base_g": g_of[base_key],
        "n_needed": len(all_keys),
        "n_replayed": len(g_of),
        "n_dev": len(devs),
        "n_candidate_sets": len(candidate_sets),
        "errors": {str(k): v for k, v in errors.items()},
    }
    if wave == "greedy":
        episode_g = discounted_from(base, t, gamma)
        state_fields["episode_greedy_g"] = episode_g
        state_fields["base_replay_gap"] = abs(g_of[base_key] - episode_g)

    base_support = support_fields(base_key, near, sidx)
    deviation_records = []
    for (kind, agent), (joint, alt) in devs.items():
        key = joint_key(joint)
        if key not in g_of:
            continue
        dev_support = support_fields(key, near, sidx)
        record = {
            **state_fields,
            "dev_kind": kind,
            "dev_agent": int(agent),
            "dev_alt_action": int(alt),
            "dev_joint": list(key),
            "dev_q": q_of[key],
            "dev_g": g_of[key],
            "abs_q_gap": abs(q_of[base_key] - q_of[key]),
            "signed_return_gap": g_of[base_key] - g_of[key],
            "abs_return_gap": abs(g_of[base_key] - g_of[key]),
            "outcome": pair_outcome(q_of[base_key], q_of[key], g_of[base_key], g_of[key]),
            "base_support": base_support,
            "dev_support": dev_support,
        }
        deviation_records.append(record)

    random_pair_records = []
    for joint_a, joint_b in random_pairs:
        key_a, key_b = joint_key(joint_a), joint_key(joint_b)
        if key_a not in g_of or key_b not in g_of:
            continue
        record = {
            **state_fields,
            "kind": "random_pair",
            "pair_group": "random",
            "joint_a": list(key_a),
            "joint_b": list(key_b),
            "q_a": q_of[key_a],
            "q_b": q_of[key_b],
            "g_a": g_of[key_a],
            "g_b": g_of[key_b],
            "abs_q_gap": abs(q_of[key_a] - q_of[key_b]),
            "abs_return_gap": abs(g_of[key_a] - g_of[key_b]),
            "outcome": pair_outcome(q_of[key_a], q_of[key_b], g_of[key_a], g_of[key_b]),
        }
        random_pair_records.append(record)

    global_pair_records = []
    for name, (joint_a, joint_b) in diag_pairs.items():
        key_a, key_b = joint_key(joint_a), joint_key(joint_b)
        if key_a not in g_of or key_b not in g_of:
            continue
        record = {
            **state_fields,
            "kind": f"global_{name}",
            "pair_group": "global_diag",
            "joint_a": list(key_a),
            "joint_b": list(key_b),
            "q_a": q_of[key_a],
            "q_b": q_of[key_b],
            "g_a": g_of[key_a],
            "g_b": g_of[key_b],
            "abs_q_gap": abs(q_of[key_a] - q_of[key_b]),
            "abs_return_gap": abs(g_of[key_a] - g_of[key_b]),
            "outcome": pair_outcome(q_of[key_a], q_of[key_b], g_of[key_a], g_of[key_b]),
        }
        global_pair_records.append(record)

    selectability_records = []
    candidate_pair_records = []
    for set_name, joints in candidate_sets.items():
        joints = _dedup_joints(joints)
        keys = [joint_key(j) for j in joints]
        missing = [k for k in keys if k not in g_of]
        if len(keys) < 2 or missing:
            selectability_records.append(
                {
                    **state_fields,
                    "set": set_name,
                    "k": len(keys),
                    "skipped": True,
                    "n_missing": len(missing),
                }
            )
            continue
        qs = [q_of[k] for k in keys]
        gs = [g_of[k] for k in keys]
        metrics = candidate_metrics(keys, qs, gs)
        order = np.argsort(-np.asarray(qs))
        candidates = []
        for qrank, idx in enumerate(order):
            candidates.append(
                {
                    "joint": list(keys[int(idx)]),
                    "q": qs[int(idx)],
                    "g": gs[int(idx)],
                    "qrank": int(qrank),
                    "grank": int(np.sum(np.asarray(gs) > gs[int(idx)] + RETURN_TIE)),
                    "support": support_fields(keys[int(idx)], near, sidx),
                }
            )
        selectability_records.append(
            {
                **state_fields,
                "set": set_name,
                "k": len(keys),
                "skipped": False,
                "base_in_set": base_key in keys,
                "candidates": candidates,
                **metrics,
            }
        )
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                qa, qb = qs[a], qs[b]
                ga, gb = gs[a], gs[b]
                candidate_pair_records.append(
                    {
                        **state_fields,
                        "set": set_name,
                        "kind": f"{set_name}_pair",
                        "joint_a": list(keys[a]),
                        "joint_b": list(keys[b]),
                        "q_a": qa,
                        "q_b": qb,
                        "g_a": ga,
                        "g_b": gb,
                        "abs_q_gap": abs(qa - qb),
                        "abs_return_gap": abs(ga - gb),
                        "outcome": pair_outcome(qa, qb, ga, gb),
                    }
                )

    return {
        "state": state_fields,
        "deviations": deviation_records,
        "random_pairs": random_pair_records,
        "global_pairs": global_pair_records,
        "selectability": selectability_records,
        "candidate_pairs": candidate_pair_records,
    }


def summarize(records: dict, args) -> dict:
    dev = records["deviations"]
    rnd = records["random_pairs"]
    gdiag = records["global_pairs"]
    sel = records["selectability"]
    cpairs = records["candidate_pairs"]
    states = records["states"]
    episodes = records["episodes"]

    summary: dict = {
        "n_episodes": len(episodes),
        "n_wins": int(sum(1 for e in episodes if e.get("won"))),
        "n_states": len(states),
        "n_deviations": len(dev),
        "n_random_pairs": len(rnd),
        "n_global_pairs": len(gdiag),
        "n_selectability_sets": len(sel),
        "n_candidate_pairs": len(cpairs),
        "base_replay_gap_max": max(
            (s.get("base_replay_gap", 0.0) for s in states), default=None
        ),
        "deviation_by_kind": rates_by(dev, lambda r: r["dev_kind"]),
        "deviation_by_agent": rates_by(dev, lambda r: f"{r['dev_kind']}|agent{r['dev_agent']}"),
        "deviation_by_kind_effect": rates_by(
            dev, lambda r: f"{r['dev_kind']}|{effect_bin(r['abs_return_gap'])}"
        ),
        "deviation_by_kind_qgap": rates_by(
            dev, lambda r: f"{r['dev_kind']}|{bucket(r['abs_q_gap'], [0, 1, 5, 20, 1e9])}"
        ),
        "random_pairs": clustered_rate(rnd),
        "random_pairs_by_effect": rates_by(rnd, lambda r: effect_bin(r["abs_return_gap"])),
        "global_diag": clustered_rate(gdiag),
        "global_diag_by_kind": rates_by(gdiag, lambda r: r["kind"]),
        "global_diag_by_effect": rates_by(gdiag, lambda r: effect_bin(r["abs_return_gap"])),
        "deviation_minus_random": cluster_bootstrap_diff(dev, rnd, _outcome_value),
        "deviation_minus_global_diag": cluster_bootstrap_diff(dev, gdiag, _outcome_value),
        "deviation_minus_random_by_kind": {
            kind: cluster_bootstrap_diff(
                [r for r in dev if r["dev_kind"] == kind], rnd, _outcome_value
            )
            for kind in sorted({r["dev_kind"] for r in dev})
        },
        "negative_knowledge": clustered_rate(
            [r for r in dev if r["signed_return_gap"] > RETURN_TIE]
        ),
        "positive_knowledge": clustered_rate(
            [r for r in dev if r["signed_return_gap"] < -RETURN_TIE]
        ),
        "deviation_signed_effect": clustered_mean(
            [r["signed_return_gap"] for r in dev], [r["cluster"] for r in dev]
        ),
        "deviation_abs_effect": clustered_mean(
            [r["abs_return_gap"] for r in dev], [r["cluster"] for r in dev]
        ),
        "deviation_signed_effect_by_kind": means_by(
            dev, lambda r: r["signed_return_gap"], lambda r: r["dev_kind"], label="signed"
        ),
        "deviation_abs_effect_by_kind": means_by(
            dev, lambda r: r["abs_return_gap"], lambda r: r["dev_kind"], label="abs"
        ),
        "deviation_effect_counts": {
            kind: _effect_counts([r for r in dev if r["dev_kind"] == kind])
            for kind in sorted({r["dev_kind"] for r in dev})
        },
        "selectability_by_set": {},
        "candidate_pairs_by_set": rates_by(cpairs, lambda r: r["set"]),
        "selectability_minus_global": {},
        "selectability_minus_random": {},
    }

    for set_name in sorted({r["set"] for r in sel if not r.get("skipped")}):
        rows = [r for r in sel if r["set"] == set_name and not r.get("skipped")]
        pair_rows = [r for r in cpairs if r["set"] == set_name]
        top1 = [
            {**r, "outcome": "agree" if r["top1_correct"] else "invert", "cluster": r["cluster"]}
            for r in rows
        ]
        top2 = [
            {**r, "outcome": "agree" if r["top2_correct"] else "invert", "cluster": r["cluster"]}
            for r in rows
        ]
        summary["selectability_by_set"][set_name] = {
            "n_sets": len(rows),
            "mean_k": float(np.mean([r["k"] for r in rows])) if rows else None,
            "top1_accuracy": clustered_rate(top1),
            "top1_chance_mean": float(np.mean([1.0 / r["k"] for r in rows])) if rows else None,
            "top1_minus_chance": clustered_mean(
                [float(r["top1_correct"]) - 1.0 / r["k"] for r in rows],
                [r["cluster"] for r in rows],
                label="top1_minus_chance",
            ),
            "top2_accuracy": clustered_rate(top2),
            "regret": clustered_mean([r["regret"] for r in rows], [r["cluster"] for r in rows]),
            "regret_p50": float(np.median([r["regret"] for r in rows])) if rows else None,
            "regret_p90": float(np.percentile([r["regret"] for r in rows], 90)) if rows else None,
            "pair_concordance": clustered_rate(pair_rows),
            "true_best_qrank": clustered_mean(
                [r["true_best_qrank"] for r in rows], [r["cluster"] for r in rows]
            ),
            "q_top1_truerank": clustered_mean(
                [r["q_top1_truerank"] for r in rows], [r["cluster"] for r in rows]
            ),
        }
        summary["selectability_minus_global"][set_name] = cluster_bootstrap_diff(
            pair_rows, gdiag, _outcome_value
        )
        summary["selectability_minus_random"][set_name] = cluster_bootstrap_diff(
            pair_rows, rnd, _outcome_value
        )

    summary["candidate_pairs_by_set_effect"] = rates_by(
        cpairs, lambda r: f"{r['set']}|{effect_bin(r['abs_return_gap'])}"
    )
    return summary


RECORD_KEYS = (
    "episodes",
    "states",
    "deviations",
    "random_pairs",
    "global_pairs",
    "selectability",
    "candidate_pairs",
)
META_KEYS = (
    "config",
    "frozen_checkpoint",
    "population",
    "waves",
    "eps",
    "probe_fracs",
    "dev_kinds",
    "candidate_sets",
    "cand_k",
    "k_near",
)


def merge_shards(shard_dir: Path) -> tuple[dict, dict]:
    records: dict = {key: [] for key in RECORD_KEYS}
    meta: dict = {}
    paths = sorted(shard_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no shard JSON files under {shard_dir}")
    n_loaded = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "states" not in payload or "episodes" not in payload:
            continue
        n_loaded += 1
        if not meta:
            meta = {key: payload.get(key) for key in META_KEYS}
        for key in RECORD_KEYS:
            records[key].extend(payload.get(key, []))
    if n_loaded == 0:
        raise RuntimeError(f"no valid shard payloads under {shard_dir}")
    return records, meta


def _write(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/5m_vs_6m.yaml")
    ap.add_argument("--frozen-ckpt", default=None)
    ap.add_argument("--n-seeds", type=int, default=80)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--waves", default="greedy")
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--population", choices=("win", "all"), default="win")
    ap.add_argument("--probe-fracs", default="0.3,0.7")
    ap.add_argument("--max-probes", type=int, default=2)
    ap.add_argument("--dev-kinds", default="primary,hard,random")
    ap.add_argument("--random-pairs", type=int, default=6)
    ap.add_argument("--candidate-sets", default="support_dev,support_neigh,random")
    ap.add_argument("--cand-k", type=int, default=8)
    ap.add_argument("--per-agent", type=int, default=2)
    ap.add_argument("--max-hamming", type=int, default=2)
    ap.add_argument("--n-recorded", type=int, default=32)
    ap.add_argument("--max-joints", type=int, default=0, help="0 = no cap (smoke only)")
    ap.add_argument("--k-near", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="logs/retention_insight.json")
    ap.add_argument(
        "--merge-dir",
        default=None,
        help="merge per-seed shard JSONs from this directory instead of measuring",
    )
    args = ap.parse_args()

    ensure_project_root()
    if args.merge_dir:
        records, meta = merge_shards(Path(resolve_path(args.merge_dir)))
        payload = {
            **meta,
            "parameter_updates": 0,
            **records,
            "summary": summarize(records, args),
        }
        out = Path(resolve_path(args.out))
        _write(out, payload)
        print(
            f"merged {len(records['episodes'])} episodes / {len(records['states'])} states "
            f"from {args.merge_dir} -> {out}",
            flush=True,
        )
        print("SUMMARY\n" + json.dumps(payload["summary"], indent=2), flush=True)
        return

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg["system"]["data_device"] = "cpu"
    cfg["system"]["device"] = args.device
    frozen_ckpt = resolve_path(args.frozen_ckpt or cfg["checkpoint_path"])
    gamma = float(cfg["system"]["discount"])
    fracs = [float(x) for x in args.probe_fracs.split(",") if x.strip()]
    waves = [x.strip() for x in args.waves.split(",") if x.strip()]
    print(
        f"frozen={frozen_ckpt} population={args.population} waves={waves} "
        f"dev={args.dev_kinds} sets={args.candidate_sets} cand_k={args.cand_k} "
        f"random_pairs={args.random_pairs} device={args.device} gamma={gamma} updates=0",
        flush=True,
    )

    dataset = load_vault_dataset(cfg)
    counts = action_counts(dataset)
    support_cpu = support_tensor(counts)
    sidx = SupportIndex(dataset, k_near=args.k_near)
    print(
        f"unique_joints={support_cpu.shape[0]} offline_states={sidx.states_norm.shape[0]}",
        flush=True,
    )
    frozen = build_learner(cfg, dataset)
    load_qmix_checkpoint(frozen_ckpt, frozen, load_optimizer=False)
    frozen.q_network.eval()
    frozen.mixer.eval()
    del dataset
    support_dev = support_cpu.to(frozen.device)

    records: dict = {
        "episodes": [],
        "states": [],
        "deviations": [],
        "random_pairs": [],
        "global_pairs": [],
        "selectability": [],
        "candidate_pairs": [],
    }
    out = Path(resolve_path(args.out))
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    for seed in seeds:
        for wave in waves:
            env = make_env(cfg, seed=seed)
            try:
                base = teacher_rollout(
                    env,
                    frozen,
                    wave=wave,
                    eps=args.eps,
                    rng=np.random.default_rng(2000 + seed),
                    store_obs=False,
                )
                if args.population == "win" and not base["won"]:
                    print(
                        f"[seed {seed} {wave}] not a win (return={base['return']:.2f}, "
                        f"L={base['length']}); skip",
                        flush=True,
                    )
                    records["episodes"].append(
                        {
                            "seed": seed,
                            "wave": wave,
                            "won": bool(base["won"]),
                            "length": base["length"],
                            "return": base["return"],
                            "kept": False,
                            "n_probes": 0,
                        }
                    )
                    continue
                probe_steps = select_probe_steps(base["length"], fracs, args.max_probes)
                print(
                    f"[seed {seed} {wave}] win={base['won']} L={base['length']} "
                    f"return={base['return']:.2f} probes={probe_steps}",
                    flush=True,
                )
                for t in probe_steps:
                    row = measure_state(
                        env, base, t, frozen, sidx, support_dev, args, gamma, seed, wave
                    )
                    if row["state"] is not None:
                        records["states"].append(row["state"])
                    records["deviations"].extend(row["deviations"])
                    records["random_pairs"].extend(row["random_pairs"])
                    records["global_pairs"].extend(row["global_pairs"])
                    records["selectability"].extend(row["selectability"])
                    records["candidate_pairs"].extend(row["candidate_pairs"])
                records["episodes"].append(
                    {
                        "seed": seed,
                        "wave": wave,
                        "won": bool(base["won"]),
                        "length": base["length"],
                        "return": base["return"],
                        "kept": True,
                        "n_probes": len(probe_steps),
                        "probe_steps": probe_steps,
                    }
                )
            finally:
                close_env(env)

        payload = {
            "config": args.config,
            "frozen_checkpoint": frozen_ckpt,
            "population": args.population,
            "seed0": args.seed0,
            "n_seeds": args.n_seeds,
            "waves": waves,
            "eps": args.eps,
            "probe_fracs": fracs,
            "dev_kinds": args.dev_kinds,
            "candidate_sets": args.candidate_sets,
            "cand_k": args.cand_k,
            "k_near": args.k_near,
            "parameter_updates": 0,
            "episodes": records["episodes"],
            "states": records["states"],
            "deviations": records["deviations"],
            "random_pairs": records["random_pairs"],
            "global_pairs": records["global_pairs"],
            "selectability": records["selectability"],
            "candidate_pairs": records["candidate_pairs"],
            "summary": summarize(records, args),
        }
        _write(out, payload)
        print(f"[seed {seed}] wrote {out}", flush=True)

    print("SUMMARY\n" + json.dumps(summarize(records, args), indent=2), flush=True)


if __name__ == "__main__":
    main()
