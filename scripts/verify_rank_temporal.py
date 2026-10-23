#!/usr/bin/env python3
"""Temporal evolution of offline-teacher ranking reliability (task 2 supplement).

The frozen offline teacher Q-bar is fixed (the same checkpoint task 2 measured).
What changes across arms is the *continuation policy* used in branch replay: a
series of online checkpoints along one UOT training trajectory, plus the
teacher itself and the seed-2 offline critic as reference arms.

For every probe state s (same states, same action pairs, same Q-bar as task 2),
each continuation policy C branch-replays the candidate joint actions: replay
the recorded prefix bit-for-bit, force joint u at the probe step, then finish
the episode greedily under C. The discounted return from the probe step is the
ground truth against which Q-bar's order is scored, so the concordance reads
"how reliable is the teacher's preference relative to the current continuation
policy".

Two measurement modes:
  fixed  -- probe states come from the teacher's own greedy / eps-greedy
            rollouts (identical to task 2). Across arms only the continuation
            policy differs: teacher, states, pairs and Q values are the same,
            so any change in concordance is attributable to the continuation.
  own    -- each continuation policy produces its own rollouts (its own visited
            states and executed anchors) while Q-bar still scores the states.
            Measures reliability on the states the current policy actually
            visits; arms are compared at the distribution level only.

Continuation specs are `tag=path[:t_env]` (t_env is inferred from the
checkpoint's `extra` when not given). Records carry `mode`, `cont` and
`cont_t` so the analysis can plot concordance against online training step.

Statistics: the effective sample is independent trajectories; every rate is
clustered by trajectory (seed, wave) with a cluster-robust SE and 95% CI.
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
from uot.support import RecordedJoints  # noqa: E402

# The proven branch-replay / pair-selection machinery, reused unchanged.
from scripts.verify_local_rank_inversion import (  # noqa: E402
    _legal_support,
    _mix_joint_q,
    action_counts,
    close_env,
    joint_key,
    support_tensor,
)
from scripts.verify_rank_reliability import (  # noqa: E402
    FRACTIONS,
    SupportIndex,
    _self_check,
    algorithm_pairs,
    base_rollout,
    branch_return,
    bucket,
    clustered_rate,
    diagnostic_pairs,
)


# --------------------------------------------------------------------------- #
# Continuation specs.
# --------------------------------------------------------------------------- #
def parse_cont_spec(spec: str) -> tuple[str, str, int | None]:
    """`tag=path[:t_env]` -> (tag, path, forced t_env or None)."""
    tag, _, rest = spec.partition("=")
    tag = tag.strip()
    if not tag or not rest.strip():
        raise ValueError(f"bad cont-spec '{spec}'; expected tag=path[:t_env]")
    rest = rest.strip()
    path_part, sep, t_part = rest.rpartition(":")
    if not sep or not path_part:
        return tag, rest, None
    t_force = int(t_part) if t_part.strip() else None
    return tag, path_part, t_force


def ckpt_t_env(path: str) -> int | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    extra = payload.get("extra") or {}
    return int(extra["t_env"]) if "t_env" in extra else None


def load_continuation(cfg, dataset, tag: str, path: str, t_force: int | None) -> dict:
    learner = build_learner(cfg, dataset)
    load_qmix_checkpoint(resolve_path(path), learner, load_optimizer=False)
    learner.q_network.eval()
    learner.mixer.eval()
    t_env = t_force if t_force is not None else ckpt_t_env(resolve_path(path))
    print(f"continuation {tag}: {path} (t_env={t_env})", flush=True)
    return {"tag": tag, "path": resolve_path(path), "t": t_env, "learner": learner}


# --------------------------------------------------------------------------- #
# Per-probe measurement, factored out of the task-2 main loop.
# --------------------------------------------------------------------------- #
def measure_probes(
    env, base, frozen, conts, mode, records, args, sidx, support_obj, support_dev,
    gamma, seed, wave,
) -> None:
    L = base["length"]
    if L < 3:
        print(f"[seed {seed} {wave} {mode}] too short (L={L}); skip", flush=True)
        return
    probe_steps = sorted(
        {int(min(max(round(f * L), 1), L - 1)) for f in FRACTIONS}
    )[: args.max_probes]
    print(f"[seed {seed} {wave} {mode}] L={L} win={base['won']} probes={probe_steps}", flush=True)
    for t in probe_steps:
        rec = base["steps"][t]
        q = rec["q"].to(frozen.device)
        legal = rec["legal"].to(frozen.device)
        state = rec["state"]
        prefix_actions = [base["steps"][k]["action"] for k in range(t)]
        executed_key = rec["action"]

        full_cand = _legal_support(support_dev, legal)
        if full_cand.shape[0] >= 2:
            q_tot = _mix_joint_q(frozen, q, full_cand, state)
            order = torch.argsort(q_tot, descending=True)
            ranked = full_cand[order]
            ranked_q = q_tot[order]
            diag = diagnostic_pairs(ranked_q)
        else:
            ranked, ranked_q, diag = full_cand, q.new_empty(full_cand.shape[0]), {}

        cand_joints, cand_q, algo = algorithm_pairs(
            frozen, q, legal, state, executed_key, support_obj, args.k_cand, args.margin
        )

        # union of distinct joints to branch-replay (each joint once per cont)
        needed: dict[tuple[int, ...], torch.Tensor] = {}

        def add(ia, ib, src_joints, src_q):
            for j in (ia, ib):
                key = joint_key(src_joints[j])
                needed.setdefault(key, src_joints[j].detach().cpu())

        for kind, (ia, ib) in diag.items():
            add(ia, ib, ranked, ranked_q)
        for ia, ib in algo:
            add(ia, ib, cand_joints, cand_q)

        needed_keys = list(needed.keys())
        if args.max_joints and len(needed_keys) > args.max_joints:
            needed_keys = needed_keys[: args.max_joints]

        near = sidx.nearest(state)
        g_of: dict[tuple[str, tuple[int, ...]], tuple] = {}
        for key in needed_keys:
            u = needed[key]
            for c in conts:
                g, err = branch_return(env, c["learner"], prefix_actions, t, u, state, gamma)
                g_of[(c["tag"], key)] = (g, err)
        base_feat = {
            "seed": seed, "wave": wave, "t": t, "L": L,
            "frac": t / max(L - 1, 1),
            "alive_allies": rec["alive_allies"], "alive_enemies": rec["alive_enemies"],
            "won": base["won"], "n_full_cand": int(ranked.shape[0]), "mode": mode,
        }
        t_of = {c["tag"]: c["t"] for c in conts}
        start = len(records)
        for c in conts:
            for kind, (ia, ib) in diag.items():
                if ia >= ranked.shape[0] or ib >= ranked.shape[0]:
                    continue
                ka, kb = joint_key(ranked[ia]), joint_key(ranked[ib])
                _emit(records, base_feat, c["tag"], kind, ka, kb,
                      float(ranked_q[ia]), float(ranked_q[ib]),
                      g_of.get((c["tag"], ka)), g_of.get((c["tag"], kb)),
                      needed, near, sidx)
            for ia, ib in algo:
                ka, kb = joint_key(cand_joints[ia]), joint_key(cand_joints[ib])
                _emit(records, base_feat, c["tag"], "algorithm", ka, kb,
                      float(cand_q[ia]), float(cand_q[ib]),
                      g_of.get((c["tag"], ka)), g_of.get((c["tag"], kb)),
                      needed, near, sidx)
        for r in records[start:]:
            r["cont_t"] = t_of.get(r["cont"])
            if mode == "own":
                r["cluster"] = f"{seed}:{wave}:{r['cont']}"


def _emit(records, base_feat, cname, kind, ka, kb, q_a, q_b, ga, gb, needed, near, sidx):
    """Thin wrapper over the task-2 emitter (same record schema)."""
    from scripts.verify_rank_reliability import _emit as _task2_emit

    return _task2_emit(records, base_feat, cname, kind, ka, kb, q_a, q_b, ga, gb,
                       needed, near, sidx)


# --------------------------------------------------------------------------- #
# Statistics.
# --------------------------------------------------------------------------- #
def _temporal_summary(records: list[dict]) -> dict:
    def by(key_fn):
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(key_fn(r), []).append(r)
        return {k: clustered_rate(v) for k, v in groups.items()}

    def mode_cont(r):
        return f"{r.get('mode')}|{r['cont']}|t{r.get('cont_t')}"

    def phase(r):
        return "early" if r["frac"] < 0.34 else ("mid" if r["frac"] < 0.67 else "late")

    return {
        "n_records": len(records),
        "by_mode_cont": by(mode_cont),
        "by_mode_cont_kind": by(lambda r: f"{r.get('mode')}|{r['cont']}|{r['kind']}"),
        "by_mode_cont_qgap": by(
            lambda r: f"{r.get('mode')}|{r['cont']}|{bucket(r['abs_q_gap'], [0, 1, 5, 20, 1e9])}"
        ),
        "by_mode_cont_phase": by(lambda r: f"{r.get('mode')}|{r['cont']}|{phase(r)}"),
        "by_mode_cont_win": by(
            lambda r: f"{r.get('mode')}|{r['cont']}|" + ("win" if r["won"] else "loss")
        ),
    }


def _write(out: Path, records, args, frozen_ckpt, conts):
    payload = {
        "config": args.config,
        "frozen_checkpoint": frozen_ckpt,
        "conts": [{"tag": c["tag"], "path": str(c["path"]), "t": c["t"]} for c in conts],
        "mode": args.mode,
        "waves": args.waves, "eps": args.eps, "k_cand": args.k_cand,
        "margin": args.margin, "k_near": args.k_near,
        "parameter_updates": 0,
        "records": records,
        "summary": _temporal_summary(records),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> None:
    _self_check()
    if parse_cont_spec("a=b.pt:5") != ("a", "b.pt", 5) or parse_cont_spec("a=b.pt") != ("a", "b.pt", None):
        raise RuntimeError("cont-spec self-check failed")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/5m_vs_6m.yaml")
    ap.add_argument("--frozen-ckpt", default=None)
    ap.add_argument("--cont-spec", action="append", required=True,
                    help="tag=path[:t_env]; repeatable")
    ap.add_argument("--mode", choices=("fixed", "own", "both"), default="fixed")
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--waves", default="greedy,epsgreedy")
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--k-cand", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--max-probes", type=int, default=4)
    ap.add_argument("--max-joints", type=int, default=0, help="0 = no cap (smoke only)")
    ap.add_argument("--k-near", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="logs/rank_temporal.json")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg["system"]["data_device"] = "cpu"
    cfg["system"]["device"] = args.device
    frozen_ckpt = resolve_path(args.frozen_ckpt or cfg["checkpoint_path"])
    gamma = float(cfg["system"]["discount"])
    waves = [w.strip() for w in args.waves.split(",") if w.strip()]
    print(f"frozen={frozen_ckpt} mode={args.mode} waves={waves} K={args.k_cand} "
          f"margin={args.margin} device={args.device} gamma={gamma} updates=0", flush=True)

    dataset = load_vault_dataset(cfg)
    counts = action_counts(dataset)
    support_cpu = support_tensor(counts)
    support_obj = RecordedJoints.from_episode_actions(dataset.actions, dataset.filled, dataset.stats.n_actions)
    sidx = SupportIndex(dataset, k_near=args.k_near)
    print(f"unique_joints={support_cpu.shape[0]} offline_states={sidx.states_norm.shape[0]}", flush=True)

    frozen = build_learner(cfg, dataset)
    load_qmix_checkpoint(frozen_ckpt, frozen, load_optimizer=False)
    frozen.q_network.eval(); frozen.mixer.eval()

    conts = [load_continuation(cfg, dataset, *parse_cont_spec(spec)) for spec in args.cont_spec]
    del dataset  # support tables already extracted to numpy/tensors

    device = frozen.device
    support_dev = support_cpu.to(device)
    out = Path(resolve_path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    modes = ("fixed", "own") if args.mode == "both" else (args.mode,)

    for seed in seeds:
        for wave in waves:
            if "fixed" in modes:
                env = make_env(cfg, seed=seed)
                try:
                    base = base_rollout(env, frozen, wave, args.eps,
                                        np.random.default_rng(1000 + seed))
                    measure_probes(env, base, frozen, conts, "fixed", records, args,
                                   sidx, support_obj, support_dev, gamma, seed, wave)
                finally:
                    close_env(env)
            if "own" in modes:
                for c in conts:
                    env = make_env(cfg, seed=seed)
                    try:
                        base = base_rollout(env, c["learner"], wave, args.eps,
                                            np.random.default_rng(1000 + seed),
                                            q_evaluator=frozen)
                        measure_probes(env, base, frozen, [c], "own", records, args,
                                       sidx, support_obj, support_dev, gamma, seed, wave)
                    finally:
                        close_env(env)
            _write(out, records, args, frozen_ckpt, conts)

    _write(out, records, args, frozen_ckpt, conts)
    print("SUMMARY\n" + json.dumps(_temporal_summary(records), indent=2), flush=True)


if __name__ == "__main__":
    main()
