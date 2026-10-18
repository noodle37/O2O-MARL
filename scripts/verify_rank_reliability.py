#!/usr/bin/env python3
"""Offline joint-action ranking reliability via branch replay (task 2).

For each probe state s (sampled from frozen-greedy AND eps-greedy rollouts, at
early/mid/late steps), rank legal joint actions by the frozen offline critic
Q_bar, then branch-replay selected action pairs to test whether the Q_bar order
matches the true discounted-return order under a fixed continuation policy.

Continuation policies (both greedy, eps=0):
  A = frozen critic greedy (Q_bar's own implicit continuation)
  B = a pure-online policy (optional --cont-ckpt); if absent, only A runs.

Branch replay re-runs the episode from the same seed and replays the recorded
prefix bit-for-bit (SMAC v1 deterministic), so environment state, observation
history (hence the GRU state) and RNG are identical at the probe step. The
current state at the probe step is checked bit-exact against the base rollout;
a mismatch drops that branch.

Pair populations per state:
  diagnostic (aligned with the existing 8-seed report, over legal x recorded
    support): top(1v2), long(1vlast), smallest |dQ| adjacent pair, largest
    |dQ| pair.
  algorithm (what UOT actually consumes): candidates from
    uot.neighborhood.build_recorded_neighborhood (S_off, Hamming<=2, K), then
    all C(K,2) pairs with |dQ_bar| >= margin.

Supports per candidate joint u (exploratory; no estimator trained): global
count, per-agent marginal, state-conditional joint count/freq via k-NN on the
normalized global state, per-agent marginal within nearby states.

Statistics: the effective sample is independent trajectories, not pairs; every
rate is clustered by trajectory (seed, wave) with a cluster-robust SE and 95%
CI. Do not read a small-sample 50%/80% as the population accuracy.
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

from training.factory import build_learner, load_vault_dataset, make_env
from utils.checkpoint import load_qmix_checkpoint
from utils.paths import ensure_project_root, resolve_path
from uot.neighborhood import build_recorded_neighborhood
from uot.support import RecordedJoints

# Reuse the proven branch-replay / Q-eval helpers from the existing protocol.
from scripts.verify_local_rank_inversion import (
    MIXER_CHUNK,
    RETURN_TIE,
    _agent_names,
    _forward_qs,
    _legal_support,
    _legal_tensor,
    _mix_joint_q,
    _new_hidden,
    action_counts,
    close_env,
    discounted_return,
    joint_key,
    reset_env,
    support_tensor,
)

FRACTIONS = (0.2, 0.4, 0.6, 0.8)  # early / mid / late probe points per episode


# --------------------------------------------------------------------------- #
# Data support index (exploratory; global / marginal / state-conditional).
# --------------------------------------------------------------------------- #
class SupportIndex:
    def __init__(self, dataset, k_near: int = 64):
        filled = dataset.filled.squeeze(-1) > 0.5  # [E,T]
        states = dataset.state[filled].detach().cpu().numpy().astype(np.float64)
        actions = dataset.actions.squeeze(-1)[filled].detach().cpu().numpy().astype(np.int64)
        self.N = int(actions.shape[1])
        self.A = int(dataset.avail_actions.shape[-1])
        self.k = int(k_near)
        self.mean = states.mean(0)
        self.std = states.std(0) + 1e-6
        self.states_norm = (states - self.mean) / self.std  # [M,D]
        self.sq = (self.states_norm ** 2).sum(1)  # [M]
        self.actions = actions  # [M,N]
        uniq, cnt = np.unique(actions, axis=0, return_counts=True)
        self.counts = {tuple(int(x) for x in uniq[i]): int(cnt[i]) for i in range(uniq.shape[0])}
        self.marginal = np.stack(
            [np.bincount(actions[:, i], minlength=self.A) for i in range(self.N)]
        )  # [N,A]

    def nearest(self, state_raw) -> np.ndarray:
        s = (np.asarray(state_raw, dtype=np.float64) - self.mean) / self.std
        d = self.sq - 2.0 * (self.states_norm @ s) + float(s @ s)
        k = min(self.k, d.size)
        idx = np.argpartition(d, k - 1)[:k]
        return self.actions[idx]  # [k,N]

    def support(self, near: np.ndarray, u_key: tuple[int, ...]) -> dict:
        u = np.asarray(u_key, dtype=np.int64)
        match = np.all(near == u[None, :], axis=1)
        state_cond = int(match.sum())
        k = int(near.shape[0])
        marg_near = [int((near[:, i] == u[i]).sum()) for i in range(self.N)]
        return {
            "global_count": int(self.counts.get(tuple(int(x) for x in u), 0)),
            "marginal_global": [int(self.marginal[i, u[i]]) for i in range(self.N)],
            "state_cond_joint_count": state_cond,
            "cond_freq": state_cond / max(k, 1),
            "marginal_near": marg_near,
            "k_near": k,
        }


def minmax_support(sup: dict) -> tuple[int, int]:
    g = sup["global_count"]
    m = min(sup["marginal_global"]) if sup["marginal_global"] else 0
    return g, m


# --------------------------------------------------------------------------- #
# Environment helpers.
# --------------------------------------------------------------------------- #
def unit_counts(env):
    try:
        inner = env._environment

        def alive(coll):
            vals = coll.values() if isinstance(coll, dict) else coll
            return sum(1 for u in vals if getattr(u, "health", 0.0) > 0)

        return alive(inner.agents), alive(inner.enemies)
    except Exception:
        return None, None


def random_legal(legal: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    legal_np = legal.detach().cpu().numpy()
    out = np.zeros(legal_np.shape[0], dtype=np.int64)
    for i in range(legal_np.shape[0]):
        options = np.flatnonzero(legal_np[i])
        out[i] = int(rng.choice(options)) if options.size else 0
    return torch.from_numpy(out).to(legal.device)


def episode_done(terminal, trunc) -> bool:
    return all(bool(np.asarray(v)) for v in terminal.values()) or all(
        bool(np.asarray(v)) for v in trunc.values()
    )


# --------------------------------------------------------------------------- #
# Base rollout: record per-step frozen Q, legal, state, action, unit counts.
# --------------------------------------------------------------------------- #
def base_rollout(env, learner, wave: str, eps: float, rng: np.random.Generator) -> dict:
    names = _agent_names(learner.n_agents)
    n_actions = learner.n_actions
    rnn = _new_hidden(learner)
    last: dict[str, int] = {}
    obs, info = reset_env(env)
    steps = []
    won = None
    t = 0
    while True:
        state = np.asarray(info["state"], dtype=np.float32).copy()
        legal = _legal_tensor(info["legals"], names, n_actions, learner.device)
        with torch.inference_mode():
            q = _forward_qs(learner, obs, rnn, last)  # [N,A]; updates rnn
        greedy = q.masked_fill(~legal, -1e9).argmax(dim=-1)
        if wave == "epsgreedy" and eps > 0 and rng.random() < eps:
            chosen = random_legal(legal, rng)
        else:
            chosen = greedy
        alive_a, alive_e = unit_counts(env)
        steps.append(
            {
                "t": t,
                "state": state,
                "q": q.detach().cpu(),
                "legal": legal.detach().cpu(),
                "action": tuple(int(chosen[i]) for i in range(learner.n_agents)),
                "alive_allies": alive_a,
                "alive_enemies": alive_e,
            }
        )
        act = {names[i]: int(chosen[i].item()) for i in range(learner.n_agents)}
        for agent in names:
            last[agent] = act[agent]
        obs, rew, terminal, trunc, info = env.step(act)
        if "battle_won" in info:
            won = bool(info["battle_won"])
        t += 1
        if episode_done(terminal, trunc) or t > 400:
            break
    return {"steps": steps, "won": won, "length": t}


# --------------------------------------------------------------------------- #
# Branch replay: replay prefix bit-for-bit, force u at probe_t, continue greedy.
# --------------------------------------------------------------------------- #
def branch_return(env, cont, prefix_actions, probe_t, forced, base_state_t, gamma):
    names = _agent_names(cont.n_agents)
    n_actions = cont.n_actions
    rnn = _new_hidden(cont)
    last: dict[str, int] = {}
    obs, info = reset_env(env)
    rewards = []
    t = 0
    while True:
        state = np.asarray(info["state"], dtype=np.float32)
        if t == probe_t and not np.array_equal(state, base_state_t):
            return None, "state_mismatch"
        with torch.inference_mode():
            q = _forward_qs(cont, obs, rnn, last)
        legal = _legal_tensor(info["legals"], names, n_actions, cont.device)
        if t < probe_t:
            chosen = torch.as_tensor(prefix_actions[t], dtype=torch.long, device=cont.device)
        elif t == probe_t:
            chosen = forced.to(cont.device)
        else:
            chosen = q.masked_fill(~legal, -1e9).argmax(dim=-1)
        act = {names[i]: int(chosen[i].item()) for i in range(cont.n_agents)}
        for agent in names:
            last[agent] = act[agent]
        obs, rew, terminal, trunc, info = env.step(act)
        rewards.append(float(np.mean([float(np.asarray(r)) for r in rew.values()])))
        t += 1
        if episode_done(terminal, trunc) or t > 400:
            break
    return discounted_return(rewards, probe_t, gamma), None


# --------------------------------------------------------------------------- #
# Pair selection at one probe state.
# --------------------------------------------------------------------------- #
def diagnostic_pairs(ranked_q: torch.Tensor) -> dict[str, tuple[int, int]]:
    n = ranked_q.shape[0]
    pairs: dict[str, tuple[int, int]] = {}
    if n >= 2:
        pairs["top"] = (0, 1)
    if n >= 2:
        pairs["long"] = (0, n - 1)
    if n >= 3:
        gaps = (ranked_q[:-1] - ranked_q[1:]).abs()
        i_small = int(torch.argmin(gaps).item())
        pairs["small_gap"] = (i_small, i_small + 1)
        i_big = int(torch.argmax(gaps).item())
        if (i_big, i_big + 1) != pairs["small_gap"]:
            pairs["large_gap"] = (i_big, i_big + 1)
    return pairs


def algorithm_pairs(frozen, q, legal, state, executed_key, support_obj, k, margin):
    N, A = q.shape
    live_q = q.view(1, 1, N, A)
    avail = legal.view(1, 1, N, A)
    executed = torch.as_tensor(executed_key, dtype=torch.long, device=q.device).view(1, 1, N)
    cand, valid = build_recorded_neighborhood(
        live_q, live_q, avail, executed, support_obj, k=k, max_hamming=2
    )
    cand_joints = cand[:, 0, 0, :]  # [K,N]
    valid_flat = valid[:, 0, 0]  # [K]
    q_tot = _mix_joint_q(frozen, q, cand_joints, state)  # [K]
    idx = [i for i in range(cand_joints.shape[0]) if bool(valid_flat[i])]
    pairs = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            ia, ib = idx[a], idx[b]
            if torch.all(cand_joints[ia] == cand_joints[ib]):
                continue
            if abs(float(q_tot[ia]) - float(q_tot[ib])) < margin:
                continue
            pairs.append((int(ia), int(ib)))
    return cand_joints, q_tot, pairs


def pair_outcome(q_a, q_b, g_a, g_b) -> str:
    dq = float(q_a) - float(q_b)
    dg = float(g_a) - float(g_b)
    if abs(dq) <= 1e-8 or abs(dg) <= RETURN_TIE:
        return "tie"
    return "agree" if (dq > 0) == (dg > 0) else "invert"


# --------------------------------------------------------------------------- #
# Statistics: cluster by trajectory.
# --------------------------------------------------------------------------- #
def clustered_rate(records: list[dict]) -> dict:
    decided = [r for r in records if r["outcome"] in ("agree", "invert")]
    if not decided:
        return {"n_pairs": len(records), "n_tie": sum(r["outcome"] == "tie" for r in records),
                "n_decided": 0, "concordance": None, "n_clusters": 0, "se": None, "ci95": None}
    clusters: dict[str, list[bool]] = {}
    for r in decided:
        clusters.setdefault(r["cluster"], []).append(r["outcome"] == "agree")
    per = np.array([np.mean(v) for v in clusters.values() if v], dtype=np.float64)
    mean = float(per.mean())
    out = {"n_pairs": len(records), "n_tie": sum(r["outcome"] == "tie" for r in records),
           "n_decided": len(decided), "concordance": mean, "n_clusters": int(per.size)}
    if per.size >= 2:
        se = float(per.std(ddof=1) / np.sqrt(per.size))
        out["se"] = se
        out["ci95"] = [mean - 1.96 * se, mean + 1.96 * se]
    else:
        out["se"] = None
        out["ci95"] = None
    return out


def bucket(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return f"[{edges[i]},{edges[i+1]})"
    return f">={edges[-1]}"


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> None:
    _self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/5m_vs_6m.yaml")
    ap.add_argument("--frozen-ckpt", default=None)
    ap.add_argument("--cont-ckpt", default=None, help="pure-online continuation B; omit to run A only")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--waves", default="greedy,epsgreedy")
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--k-cand", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--max-probes", type=int, default=4)
    ap.add_argument("--max-joints", type=int, default=0, help="0 = no cap; cap distinct joints/probe (smoke)")
    ap.add_argument("--k-near", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="logs/rank_reliability.json")
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg["system"]["data_device"] = "cpu"
    cfg["system"]["device"] = args.device
    frozen_ckpt = resolve_path(args.frozen_ckpt or cfg["checkpoint_path"])
    gamma = float(cfg["system"]["discount"])
    n_actions = None
    waves = [w.strip() for w in args.waves.split(",") if w.strip()]
    print(f"frozen={frozen_ckpt} cont={args.cont_ckpt} waves={waves} K={args.k_cand} "
          f"margin={args.margin} device={args.device} gamma={gamma} updates=0", flush=True)

    dataset = load_vault_dataset(cfg)
    n_agents = dataset.stats.n_agents
    n_actions = dataset.stats.n_actions
    counts = action_counts(dataset)
    support_cpu = support_tensor(counts)
    support_obj = RecordedJoints.from_episode_actions(dataset.actions, dataset.filled, n_actions)
    sidx = SupportIndex(dataset, k_near=args.k_near)
    print(f"unique_joints={support_cpu.shape[0]} offline_states={sidx.states_norm.shape[0]}", flush=True)

    frozen = build_learner(cfg, dataset)
    load_qmix_checkpoint(frozen_ckpt, frozen, load_optimizer=False)
    frozen.q_network.eval(); frozen.mixer.eval()

    cont_b = None
    if args.cont_ckpt:
        cb = resolve_path(args.cont_ckpt)
        if os.path.isfile(cb):
            cont_b = build_learner(cfg, dataset)
            load_qmix_checkpoint(cb, cont_b, load_optimizer=False)
            cont_b.q_network.eval(); cont_b.mixer.eval()
            print(f"loaded continuation B from {cb}", flush=True)
        else:
            print(f"WARNING: cont-ckpt not found ({cb}); running continuation A only", flush=True)
    del dataset  # support tables already extracted to numpy/tensors

    device = frozen.device
    support_dev = support_cpu.to(device)
    conts = [("A", frozen)] + ([("B", cont_b)] if cont_b is not None else [])

    out = Path(resolve_path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))

    for seed in seeds:
        for wave in waves:
            env = make_env(cfg, seed=seed)
            try:
                rng = np.random.default_rng(1000 + seed)
                base = base_rollout(env, frozen, wave, args.eps, rng)
                L = base["length"]
                if L < 3:
                    print(f"[seed {seed} {wave}] too short (L={L}); skip", flush=True)
                    continue
                probe_steps = sorted({
                    int(min(max(round(f * L), 1), L - 1)) for f in FRACTIONS
                })[: args.max_probes]
                print(f"[seed {seed} {wave}] L={L} win={base['won']} probes={probe_steps}", flush=True)
                for t in probe_steps:
                    rec = base["steps"][t]
                    q = rec["q"].to(device)
                    legal = rec["legal"].to(device)
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

                    # union of distinct joints to branch-replay
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
                        for cname, cont in conts:
                            g, err = branch_return(env, cont, prefix_actions, t, u, state, gamma)
                            g_of[(cname, key)] = (g, err)
                    base_feat = {
                        "seed": seed, "wave": wave, "t": t, "L": L,
                        "frac": t / max(L - 1, 1),
                        "alive_allies": rec["alive_allies"], "alive_enemies": rec["alive_enemies"],
                        "won": base["won"], "n_full_cand": int(ranked.shape[0]),
                    }
                    for cname, _ in conts:
                        for kind, (ia, ib) in diag.items():
                            if ia >= ranked.shape[0] or ib >= ranked.shape[0]:
                                continue
                            ka, kb = joint_key(ranked[ia]), joint_key(ranked[ib])
                            _emit(records, base_feat, cname, kind, ka, kb,
                                  float(ranked_q[ia]), float(ranked_q[ib]),
                                  g_of.get((cname, ka)), g_of.get((cname, kb)),
                                  needed, near, sidx)
                        for ia, ib in algo:
                            ka, kb = joint_key(cand_joints[ia]), joint_key(cand_joints[ib])
                            _emit(records, base_feat, cname, "algorithm", ka, kb,
                                  float(cand_q[ia]), float(cand_q[ib]),
                                  g_of.get((cname, ka)), g_of.get((cname, kb)),
                                  needed, near, sidx)
                _write(out, records, args, frozen_ckpt)
            finally:
                close_env(env)

    _write(out, records, args, frozen_ckpt)
    print("SUMMARY\n" + json.dumps(_summary(records), indent=2), flush=True)


def _emit(records, base_feat, cname, kind, ka, kb, q_a, q_b, ga, gb, needed, near, sidx):
    if ga is None or gb is None:
        return
    g_a, err_a = ga
    g_b, err_b = gb
    if g_a is None or g_b is None:
        return
    sup_a = sidx.support(near, ka)
    sup_b = sidx.support(near, kb)
    rec = dict(base_feat)
    rec.update({
        "cluster": f"{base_feat['seed']}:{base_feat['wave']}",
        "cont": cname, "kind": kind,
        "joint_a": list(ka), "joint_b": list(kb),
        "q_a": q_a, "q_b": q_b, "abs_q_gap": abs(q_a - q_b),
        "g_a": g_a, "g_b": g_b, "abs_return_gap": abs(g_a - g_b),
        "outcome": pair_outcome(q_a, q_b, g_a, g_b),
        "min_global_count": min(sup_a["global_count"], sup_b["global_count"]),
        "min_marginal_global": min(min(sup_a["marginal_global"]), min(sup_b["marginal_global"])),
        "min_state_cond": min(sup_a["state_cond_joint_count"], sup_b["state_cond_joint_count"]),
        "min_cond_freq": min(sup_a["cond_freq"], sup_b["cond_freq"]),
        "min_marginal_near": min(min(sup_a["marginal_near"]), min(sup_b["marginal_near"])),
        "sup_a": sup_a, "sup_b": sup_b,
    })
    records.append(rec)


def _summary(records: list[dict]) -> dict:
    def by(key_fn):
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(key_fn(r), []).append(r)
        return {k: clustered_rate(v) for k, v in groups.items()}

    return {
        "n_records": len(records),
        "pooled": clustered_rate(records),
        "by_cont": by(lambda r: r["cont"]),
        "by_kind": by(lambda r: r["kind"]),
        "by_phase": by(lambda r: "early" if r["frac"] < 0.34 else ("mid" if r["frac"] < 0.67 else "late")),
        "by_win": by(lambda r: "win" if r["won"] else "loss"),
        "by_wave": by(lambda r: r["wave"]),
        "by_qgap": by(lambda r: bucket(r["abs_q_gap"], [0, 1, 5, 20, 1e9])),
        "by_state_cond": by(lambda r: bucket(r["min_state_cond"], [0, 1, 3, 10, 1e9])),
        "by_global_count": by(lambda r: bucket(r["min_global_count"], [0, 1, 10, 50, 1e9])),
    }


def _write(out: Path, records, args, frozen_ckpt):
    payload = {
        "config": args.config,
        "frozen_checkpoint": frozen_ckpt,
        "cont_checkpoint": args.cont_ckpt,
        "waves": args.waves, "eps": args.eps, "k_cand": args.k_cand,
        "margin": args.margin, "k_near": args.k_near,
        "parameter_updates": 0,
        "records": records,
        "summary": _summary(records),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _self_check() -> None:
    if pair_outcome(1.0, 0.0, 2.0, 1.0) != "agree":
        raise RuntimeError("agree self-check failed")
    if pair_outcome(1.0, 0.0, 1.0, 2.0) != "invert":
        raise RuntimeError("invert self-check failed")
    if pair_outcome(1.0, 0.0, 1.0, 1.0) != "tie":
        raise RuntimeError("tie self-check failed")
    if diagnostic_pairs(torch.tensor([5.0, 4.9, 1.0, -3.0]))["top"] != (0, 1):
        raise RuntimeError("diag top self-check failed")


if __name__ == "__main__":
    main()
