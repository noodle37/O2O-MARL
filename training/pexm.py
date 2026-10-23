#!/usr/bin/env python3
"""Episode-level PEX-M online pilot for 5m_vs_6m.

Two QMIX learners start from the same offline checkpoint:

  offline : frozen teacher, used as one behaviour arm;
  online  : trained with pure online TD(lambda), no OVM / CQL / UOT / offline
            data mixing. This isolates "keep the behaviour policy, release the
            critic" from the ranking losses tested elsewhere.

Every episode is assigned to an arm by a Normal-Inverse-Gamma Thompson
selector over episode returns. A min_online floor forces some online episodes
so the trainable arm cannot be starved. Both arms explore with the same
epsilon schedule during collection; evaluation is greedy.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from training.evaluation import evaluate
from training.factory import build_learner, load_vault_dataset, make_env, make_logger
from training.online import _mean_stats
from training.rollout import stack_episodes, td_terminated_flag
from training.rollout_helpers import actions_to_step_array, legals_from_info
from utils.checkpoint import load_qmix_checkpoint, save_qmix_checkpoint
from utils.paths import resolve_path

ARMS = ("offline", "online")


def _new_policy_state(net, learner) -> Dict[str, Any]:
    return {
        "rnn": {
            f"agent_{i}": net.initial_state(1, learner.device)
            for i in range(learner.n_agents)
        },
        "last": {},
    }


def select_actions_with_net(net, learner, observations, legals, eps, state, rng) -> Dict[str, int]:
    """epsilon-greedy per-agent actions from an explicit Q network.

    Mirrors qmix.policy.EvalPolicyMixin.select_actions_eval but swaps the network
    (frozen offline or online) and takes a per-environment RNG.
    """
    epsilon_dec = 0.0 if eps <= 0.0 else eps / learner.n_agents
    actions: Dict[str, int] = {}
    for i, agent in enumerate(sorted(observations.keys())):
        obs = torch.from_numpy(observations[agent]).float().to(learner.device)
        last_idx = state["last"].get(agent)
        if last_idx is None:
            last_oh = torch.zeros(learner.n_actions, device=learner.device)
        else:
            last_oh = torch.nn.functional.one_hot(
                torch.tensor(int(last_idx), device=learner.device), learner.n_actions
            ).float()
        pieces = [obs, last_oh]
        if learner.add_agent_id_to_obs:
            pieces.append(
                torch.nn.functional.one_hot(
                    torch.tensor(i, device=learner.device), learner.n_agents
                ).float()
            )
        obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
        with torch.inference_mode():
            q, state["rnn"][agent] = net(obs_in, state["rnn"].get(agent))
            q = q[0]
            if legals is None:
                legal = torch.ones(learner.n_actions, dtype=torch.bool, device=learner.device)
            else:
                legal = torch.from_numpy(legals[agent]).bool().to(learner.device)
            q = q.masked_fill(~legal, -1e9)
        if eps > 0.0 and rng.random() < epsilon_dec:
            legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1).cpu().numpy()
            action = int(legal_idx[int(rng.integers(legal_idx.size))])
        else:
            action = int(q.argmax().item())
        state["last"][agent] = action
        actions[agent] = action
    return actions


class ThompsonReturnSelector:
    """Normal-Inverse-Gamma posterior over each arm's mean episode return."""

    def __init__(
        self,
        *,
        min_online: float = 0.25,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 2.0,
        beta0: float = 25.0,
    ):
        if not 0.0 <= float(min_online) <= 1.0:
            raise ValueError(f"min_online must be in [0, 1], got {min_online}")
        self.min_online = float(min_online)
        self.prior = {
            "mu0": float(mu0),
            "kappa0": float(kappa0),
            "alpha0": float(alpha0),
            "beta0": float(beta0),
        }
        self._obs = {
            arm: {"n": 0, "mean": 0.0, "m2": 0.0} for arm in ARMS
        }
        self.counts = Counter()
        self.forced_online = 0

    # -- posterior -------------------------------------------------------- #
    def params(self, arm: str) -> Dict[str, float]:
        obs = self._obs[arm]
        p = self.prior
        n = float(obs["n"])
        kappa = p["kappa0"] + n
        mu = (p["kappa0"] * p["mu0"] + n * obs["mean"]) / kappa
        alpha = p["alpha0"] + 0.5 * n
        beta = (
            p["beta0"]
            + 0.5 * obs["m2"]
            + 0.5 * p["kappa0"] * n * (obs["mean"] - p["mu0"]) ** 2 / kappa
        )
        return {
            "n": int(obs["n"]),
            "mu": float(mu),
            "kappa": float(kappa),
            "alpha": float(alpha),
            "beta": float(beta),
            "mean": float(obs["mean"]) if obs["n"] else None,
            "sd": float(math.sqrt(beta / ((alpha - 1.0) * kappa))) if alpha > 1.0 else None,
        }

    def sample_mean(self, arm: str, rng: np.random.Generator) -> float:
        p = self.params(arm)
        scale = math.sqrt(p["beta"] / (p["alpha"] * p["kappa"]))
        return float(p["mu"] + scale * rng.standard_t(2.0 * p["alpha"]))

    def choose(self, rng: np.random.Generator) -> Tuple[str, bool]:
        if rng.random() < self.min_online:
            self.forced_online += 1
            self.counts["online"] += 1
            return "online", True
        if self.sample_mean("online", rng) >= self.sample_mean("offline", rng):
            self.counts["online"] += 1
            return "online", False
        self.counts["offline"] += 1
        return "offline", False

    def observe(self, arm: str, value: float) -> None:
        obs = self._obs[arm]
        obs["n"] += 1
        delta = float(value) - obs["mean"]
        obs["mean"] += delta / obs["n"]
        obs["m2"] += delta * (float(value) - obs["mean"])

    # -- persistence ------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        total = int(sum(self.counts.values()))
        return {
            "min_online": self.min_online,
            "prior": dict(self.prior),
            "params": {arm: self.params(arm) for arm in ARMS},
            "counts": {arm: int(self.counts[arm]) for arm in ARMS},
            "forced_online": int(self.forced_online),
            "online_share": (self.counts["online"] / total) if total else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.stats(), "obs": {arm: dict(v) for arm, v in self._obs.items()}}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ThompsonReturnSelector":
        prior = payload.get("prior") or {}
        sel = cls(
            min_online=float(payload.get("min_online", 0.0)),
            mu0=float(prior.get("mu0", 0.0)),
            kappa0=float(prior.get("kappa0", 1.0)),
            alpha0=float(prior.get("alpha0", 2.0)),
            beta0=float(prior.get("beta0", 25.0)),
        )
        for arm in ARMS:
            obs = (payload.get("obs") or {}).get(arm)
            if obs:
                sel._obs[arm] = {
                    "n": int(obs.get("n", 0)),
                    "mean": float(obs.get("mean", 0.0)),
                    "m2": float(obs.get("m2", 0.0)),
                }
        for arm in ARMS:
            sel.counts[arm] = int((payload.get("counts") or {}).get(arm, 0))
        sel.forced_online = int(payload.get("forced_online", 0))
        return sel


def collect_pexm_round(
    envs,
    online,
    offline,
    selector: ThompsonReturnSelector,
    eps: float,
    act_rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """One episode per env, each on its own selector-chosen arm."""
    n = len(envs)
    choices = [selector.choose(act_rng) for _ in range(n)]
    learners = [online if arm == "online" else offline for arm, _ in choices]
    nets = [learner.q_network for learner in learners]
    states = [_new_policy_state(net, learner) for net, learner in zip(nets, learners)]
    obs_info = [env.reset() for env in envs]
    episodes = [
        {"steps": [], "length": 0, "return": 0.0, "won": None,
         "arm": arm, "forced": bool(forced)}
        for arm, forced in choices
    ]
    done = [False] * n
    pool = ThreadPoolExecutor(max_workers=n)

    def step_one(i: int, actions):
        if done[i]:
            return None
        return envs[i].step(actions)

    try:
        while not all(done):
            actions = [None] * n
            for i in range(n):
                if done[i]:
                    continue
                obs, info = obs_info[i]
                agents = sorted(obs.keys())
                actions[i] = select_actions_with_net(
                    nets[i],
                    learners[i],
                    obs,
                    legals_from_info(info, agents),
                    eps,
                    states[i],
                    act_rng,
                )
            results = list(pool.map(lambda i: step_one(i, actions[i]), range(n)))
            for i, result in enumerate(results):
                if result is None:
                    continue
                obs, info = obs_info[i]
                next_obs, rewards, terminal, trunc, next_info = result
                agents = sorted(obs.keys())
                step = {
                    "obs": np.stack([obs[a] for a in agents]),
                    "actions": actions_to_step_array(actions[i], agents),
                    "state": info["state"],
                    "reward": float(np.mean(list(rewards.values()))),
                    "terminated": td_terminated_flag(terminal, next_info),
                }
                legals = legals_from_info(info, agents)
                if legals is not None:
                    step["legals"] = np.stack([legals[a] for a in agents])
                episodes[i]["steps"].append(step)
                episodes[i]["return"] += step["reward"]
                obs_info[i] = (next_obs, next_info)
                if all(terminal.values()) or all(trunc.values()):
                    done[i] = True
                    episodes[i]["length"] = len(episodes[i]["steps"])
                    if "battle_won" in next_info:
                        episodes[i]["won"] = bool(next_info["battle_won"])
    finally:
        pool.shutdown(wait=True)
    return episodes


def _prefix(prefix: str, logs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        f"{prefix}{key.split('/', 1)[1]}": value
        for key, value in logs.items()
        if "/" in key
    }


def _write_selector(path: str, selector: ThompsonReturnSelector, cfg: Dict[str, Any],
                    history: List[Dict[str, Any]], extra: Dict[str, Any]) -> None:
    payload = selector.to_dict()
    payload.update(
        {
            "history": history,
            "config": cfg.get("pexm", {}),
            "seed": int(cfg["seed"]),
            **extra,
        }
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def run_pexm_online(cfg: Dict[str, Any], offline_ckpt: Optional[str] = None) -> None:
    pexm = cfg.get("pexm") or {}
    seed = int(cfg["seed"])
    dataset = load_vault_dataset(cfg)
    online = build_learner(cfg, dataset)
    offline = build_learner(cfg, dataset)
    max_seq = int(dataset.stats.max_seq_length)
    del dataset

    ckpt = resolve_path(offline_ckpt or pexm.get("offline_checkpoint") or cfg.get("checkpoint_path"))
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError(f"PEX-M offline checkpoint missing: {ckpt}")
    load_qmix_checkpoint(ckpt, online, load_optimizer=True, map_location=online.device)
    load_qmix_checkpoint(ckpt, offline, load_optimizer=False, map_location=offline.device)
    for param in offline.q_network.parameters():
        param.requires_grad_(False)
    offline.q_network.eval()
    offline.mixer.eval()

    selector = ThompsonReturnSelector(
        min_online=float(pexm.get("min_online", 0.25)),
        mu0=float(pexm.get("prior_mu", 0.0)),
        kappa0=float(pexm.get("prior_kappa", 1.0)),
        alpha0=float(pexm.get("prior_alpha", 2.0)),
        beta0=float(pexm.get("prior_beta", 25.0)),
    )

    logger = make_logger(cfg)
    envs = [make_env(cfg, seed=seed + i) for i in range(int(cfg.get("batch_size_run", 8)))]
    eval_env = envs[0]
    num_eval = int(cfg.get("num_eval_episodes", 32))
    test_interval = int(cfg.get("test_interval", 10_000))
    t_max = int(cfg["t_max"])
    batch_size = int(cfg.get("batch_size", 128))
    buffer_size = int(cfg.get("buffer_size", 5000))
    ckpt_every = int(pexm.get("ckpt_every", 100_000))
    ckpt_dir = resolve_path(cfg.get("checkpoint_dir") or ".")
    os.makedirs(ckpt_dir, exist_ok=True)
    selector_path = os.path.join(ckpt_dir, f"pexm_selector_seed{seed}.json")

    eps_start = float(cfg["system"].get("eps_start", 1.0))
    eps_min = float(cfg["system"].get("eps_min", 0.05))
    eps_decay = int(cfg["system"].get("eps_decay_steps", 100_000))

    buffer: deque = deque(maxlen=buffer_size)
    t_env = 0
    online.t_env = 0
    last_test_t = -test_interval
    last_ckpt_t = 0
    history: List[Dict[str, Any]] = []
    train_stats: List[Dict[str, float]] = []
    act_rng = np.random.default_rng(seed + 1)
    batch_rng = np.random.default_rng(seed + 2)
    interval_choices: Counter = Counter()
    interval_forced = 0

    def eps_at(t: int) -> float:
        return online.linear_epsilon(t, eps_start, eps_min, eps_decay)

    print(
        f"PEX-M online learner={type(online).__name__} offline_ckpt={ckpt} "
        f"t_max={t_max} envs={len(envs)} min_online={selector.min_online}",
        flush=True,
    )

    while t_env < t_max:
        if test_interval > 0 and t_env - last_test_t >= test_interval:
            last_test_t = t_env
            online_logs = evaluate(eval_env, online, num_eval, t_env, cfg)
            offline_logs = evaluate(eval_env, offline, num_eval, t_env, cfg)
            row: Dict[str, Any] = {
                **_prefix("eval_online/", online_logs),
                **_prefix("eval_offline/", offline_logs),
                **_mean_stats(train_stats),
                "evaluation/training_steps": t_env,
                "t_env": t_env,
                "transfer/mode": "pexm",
                "pexm/interval_offline": int(interval_choices["offline"]),
                "pexm/interval_online": int(interval_choices["online"]),
                "pexm/interval_forced_online": int(interval_forced),
                "pexm/eps": eps_at(t_env),
            }
            stats = selector.stats()
            row["pexm/online_share_cum"] = stats["online_share"]
            for arm in ARMS:
                p = stats["params"][arm]
                row[f"pexm/{arm}_mu"] = p["mu"]
                row[f"pexm/{arm}_sd"] = p["sd"]
                row[f"pexm/{arm}_n"] = float(p["n"])
            logger.write(row, force=True)
            history.append(row)
            train_stats = []
            interval_choices = Counter()
            interval_forced = 0
            _write_selector(selector_path, selector, cfg, history, {"t_env": t_env})
            if ckpt_every > 0 and t_env - last_ckpt_t >= ckpt_every:
                last_ckpt_t = t_env
                periodic = os.path.join(ckpt_dir, "periodic", f"pexm_ckpt_t{t_env}.pt")
                save_qmix_checkpoint(
                    periodic,
                    online,
                    include_optimizer=True,
                    extra={"seed": seed, "t_env": t_env, "phase": "online",
                           "learner": "pexm"},
                )
                _write_selector(
                    periodic.replace(".pt", "_selector.json"), selector, cfg, history,
                    {"t_env": t_env},
                )

        episodes = collect_pexm_round(envs, online, offline, selector, eps_at(t_env), act_rng)
        for ep in episodes:
            buffer.append(ep)
            selector.observe(ep["arm"], float(ep["return"]))
            interval_choices[ep["arm"]] += 1
            interval_forced += int(ep["forced"])
        t_env += sum(int(ep["length"]) for ep in episodes)
        online.t_env = t_env

        if len(buffer) < batch_size:
            continue

        picked = batch_rng.choice(len(buffer), size=batch_size, replace=False)
        train_batch = stack_episodes(
            [buffer[int(i)] for i in picked], max_seq, online.device
        )
        logs = online.train_on_batch(
            train_batch,
            cql_weight=0.0,
            use_ovm=False,
            t_env=t_env,
            episode_delta=len(envs),
        )
        train_stats.append(logs)

    final_path = os.path.join(ckpt_dir, f"pexm_online_seed{seed}.pt")
    save_qmix_checkpoint(
        final_path,
        online,
        include_optimizer=True,
        extra={"seed": seed, "t_env": t_env, "phase": "online", "learner": "pexm"},
    )
    _write_selector(selector_path, selector, cfg, history,
                    {"t_env": t_env, "final_checkpoint": final_path})
    print(
        f"PEX-M done t_env={t_env} online_share={selector.stats()['online_share']} "
        f"final={final_path} selector={selector_path}",
        flush=True,
    )
