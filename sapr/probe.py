# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Random local-intervention probe for the frozen prior's average team value.

One probe samples an online episode, picks a decision time uniformly from it,
picks one agent uniformly, flips a fair coin ``Z``, and replaces that agent's
action with a sample from

    Z = 1: pi_off^i(. | h^i)   (frozen offline prior)
    Z = 0: pi_on^i(. | h^i)    (current online policy)

while every other agent follows the current online policy.  All agents then
resume the online policy and the discounted team return from the intervention
step is recorded.  Because ``Z`` is independent of the state, the difference of
group means estimates the average local value of the prior under the current
teammates:

    mu_hat = mean(G | Z=1) - mean(G | Z=0).

The online network is deep-copied for the whole probe and the offline network
is frozen, so probing never mutates training state or consumes its RNG streams.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sapr.controller import SAPRSettings
from sapr.regularizer import masked_policy
from training.rollout_helpers import legals_from_info


@dataclass
class ProbeRecord:
    episode: int
    z: int
    agent: int
    intervention_step: int
    ret: float
    base_steps: int
    replay_steps: int


def _close_env(env) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def _fresh_state(net: torch.nn.Module, learner) -> Dict[str, Any]:
    return {
        "rnn": {
            f"agent_{i}": net.initial_state(1, learner.device)
            for i in range(learner.n_agents)
        },
        "last": {},
    }


def _agent_input(learner, observation, last_action: Optional[int], agent_index: int):
    pieces = [torch.from_numpy(np.asarray(observation)).float().to(learner.device)]
    if last_action is None:
        pieces.append(torch.zeros(learner.n_actions, device=learner.device))
    else:
        pieces.append(
            F.one_hot(
                torch.tensor(int(last_action), device=learner.device),
                learner.n_actions,
            ).float()
        )
    if getattr(learner, "add_agent_id_to_obs", False):
        pieces.append(
            F.one_hot(
                torch.tensor(agent_index, device=learner.device), learner.n_agents
            ).float()
        )
    return torch.cat(pieces, dim=-1).unsqueeze(0)


def _legal_tensor(learner, legals, agent: str) -> torch.Tensor:
    if legals is None or agent not in legals:
        return torch.ones(learner.n_actions, dtype=torch.bool, device=learner.device)
    return torch.from_numpy(np.asarray(legals[agent])).bool().to(learner.device)


def _agent_qs(net, learner, observations, legals, state) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Per-agent masked Q values; advances the net's hidden state in place."""
    qs: Dict[str, torch.Tensor] = {}
    usable_legals: Dict[str, torch.Tensor] = {}
    for i, agent in enumerate(sorted(observations.keys())):
        legal = _legal_tensor(learner, legals, agent)
        usable = legal if bool(legal.any()) else torch.ones_like(legal)
        q, state["rnn"][agent] = net(
            _agent_input(learner, observations[agent], state["last"].get(agent), i),
            state["rnn"].get(agent),
        )
        q = q.reshape(-1).masked_fill(~usable, -1e9)
        qs[agent] = q
        usable_legals[agent] = usable
    return qs, usable_legals


def _greedy_action(q: torch.Tensor) -> int:
    return int(torch.argmax(q).item())


def _sample_action(
    q: torch.Tensor,
    legal: torch.Tensor,
    *,
    temperature: float,
    support_floor: float,
    rng: np.random.Generator,
) -> int:
    probs = (
        masked_policy(q, legal, temperature, support_floor)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return _greedy_action(q)
    probs = probs / total
    return int(rng.choice(probs.size, p=probs))


def _select_actions(
    qs: Mapping[str, torch.Tensor],
    legals: Mapping[str, torch.Tensor],
    *,
    behaviour: str,
    temperature: float,
    support_floor: float,
    rng: np.random.Generator,
) -> Dict[str, int]:
    if behaviour == "greedy":
        return {agent: _greedy_action(q) for agent, q in qs.items()}
    return {
        agent: _sample_action(
            q,
            legals[agent],
            temperature=temperature,
            support_floor=support_floor,
            rng=rng,
        )
        for agent, q in qs.items()
    }


def _episode_over(terminal, trunc) -> bool:
    return all(bool(v) for v in terminal.values()) or all(
        bool(v) for v in trunc.values()
    )


def _base_length(
    env,
    net,
    learner,
    *,
    behaviour: str,
    temperature: float,
    support_floor: float,
    rng: np.random.Generator,
) -> int:
    """Run one online-policy episode and return its number of decisions."""
    obs, info = env.reset()
    agents = sorted(obs.keys())
    state = _fresh_state(net, learner)
    steps = 0
    while True:
        legals = legals_from_info(info, agents)
        qs, usable = _agent_qs(net, learner, obs, legals, state)
        actions = _select_actions(
            qs,
            usable,
            behaviour=behaviour,
            temperature=temperature,
            support_floor=support_floor,
            rng=rng,
        )
        obs, rewards, terminal, trunc, info = env.step(actions)
        state["last"].update(actions)
        steps += 1
        if _episode_over(terminal, trunc):
            return steps


def _replay_probe(
    env,
    online_net,
    offline_net,
    learner,
    *,
    t_int: int,
    actor: str,
    z: int,
    gamma: float,
    behaviour: str,
    temperature: float,
    support_floor: float,
    rng: np.random.Generator,
) -> Tuple[Optional[float], int]:
    """Replay to ``t_int``, intervene on one agent, and return its discounted return.

    ``steps`` counts every environment step consumed, including aborted replays.
    """
    obs, info = env.reset()
    agents = sorted(obs.keys())
    online = _fresh_state(online_net, learner)
    offline = _fresh_state(offline_net, learner)
    steps = 0

    for _ in range(int(t_int)):
        legals = legals_from_info(info, agents)
        q_on, usable_on = _agent_qs(online_net, learner, obs, legals, online)
        if z:
            _agent_qs(offline_net, learner, obs, legals, offline)
        actions = _select_actions(
            q_on,
            usable_on,
            behaviour=behaviour,
            temperature=temperature,
            support_floor=support_floor,
            rng=rng,
        )
        obs, rewards, terminal, trunc, info = env.step(actions)
        online["last"].update(actions)
        if z:
            offline["last"].update(actions)
        steps += 1
        if _episode_over(terminal, trunc):
            return None, steps

    legals = legals_from_info(info, agents)
    q_on, usable_on = _agent_qs(online_net, learner, obs, legals, online)
    q_off = usable_off = None
    if z:
        q_off, usable_off = _agent_qs(offline_net, learner, obs, legals, offline)
    # Dead units have an all-false legal mask (the training rollout then forces
    # action 0, which the environment ignores). Intervening on such a unit has
    # no effect, so the probe is undefined and must be resampled.
    raw_legal = legals.get(actor) if legals is not None else None
    if raw_legal is not None and not bool(np.any(raw_legal)):
        return None, steps
    actions = _select_actions(
        q_on,
        usable_on,
        behaviour=behaviour,
        temperature=temperature,
        support_floor=support_floor,
        rng=rng,
    )
    source_q, source_legal = (q_off, usable_off) if z else (q_on, usable_on)
    actions[actor] = _sample_action(
        source_q[actor],
        source_legal[actor],
        temperature=temperature,
        support_floor=support_floor,
        rng=rng,
    )
    obs, rewards, terminal, trunc, info = env.step(actions)
    online["last"].update(actions)
    if z:
        offline["last"].update(actions)
    steps += 1
    ret = float(np.mean(list(rewards.values())))
    power = 1
    while not _episode_over(terminal, trunc):
        legals = legals_from_info(info, agents)
        q_on, usable_on = _agent_qs(online_net, learner, obs, legals, online)
        actions = _select_actions(
            q_on,
            usable_on,
            behaviour=behaviour,
            temperature=temperature,
            support_floor=support_floor,
            rng=rng,
        )
        obs, rewards, terminal, trunc, info = env.step(actions)
        online["last"].update(actions)
        ret += (gamma**power) * float(np.mean(list(rewards.values())))
        power += 1
        steps += 1
    return ret, steps


def _cluster_bootstrap(
    records: Sequence[ProbeRecord],
    *,
    samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Episode-clustered bootstrap of the difference in group means.

    Clusters are resampled independently within each treatment group so the
    two group sizes are preserved. Returns ``(se, ci_low, ci_high)``; any
    undefined quantity is reported as ``inf``/``nan``.
    """
    by_group: Dict[int, Dict[int, List[float]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    for record in records:
        by_group[record.z][record.episode].append(float(record.ret))
    if len(by_group[0]) < 2 or len(by_group[1]) < 2 or samples < 2:
        return float("inf"), float("nan"), float("nan")

    off_clusters = [np.asarray(v, dtype=np.float64) for v in by_group[1].values()]
    on_clusters = [np.asarray(v, dtype=np.float64) for v in by_group[0].values()]
    draws: List[float] = []
    for _ in range(int(samples)):
        picked_off = rng.integers(0, len(off_clusters), size=len(off_clusters))
        picked_on = rng.integers(0, len(on_clusters), size=len(on_clusters))
        off = np.concatenate([off_clusters[i] for i in picked_off])
        on = np.concatenate([on_clusters[i] for i in picked_on])
        draws.append(float(off.mean() - on.mean()))
    arr = np.asarray(draws, dtype=np.float64)
    se = float(arr.std(ddof=1))
    low, high = np.percentile(arr, [2.5, 97.5])
    return se, float(low), float(high)


def estimate_prior_utility(
    records: Sequence[ProbeRecord],
    settings: SAPRSettings,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Difference of group means and its standard error.

    The default analytic standard error is the one in the algorithm
    specification. ``se_mode="cluster_bootstrap"`` replaces it with an
    episode-clustered bootstrap, which is required when one trajectory
    contributes several probe records.
    """
    off = np.asarray([r.ret for r in records if r.z == 1], dtype=np.float64)
    on = np.asarray([r.ret for r in records if r.z == 0], dtype=np.float64)
    stats: Dict[str, float] = {
        "n_off": float(off.size),
        "n_on": float(on.size),
        "valid": 0.0,
        "mu_hat": float("nan"),
        "s_hat": float("inf"),
        "s_analytic": float("inf"),
        "mu_ci_low": float("nan"),
        "mu_ci_high": float("nan"),
    }
    if off.size and on.size:
        stats["mean_g_off"] = float(off.mean())
        stats["mean_g_on"] = float(on.mean())
        stats["mu_hat"] = float(off.mean() - on.mean())
    valid_records = [r for r in records]
    if valid_records:
        stats["mean_intervention_step"] = float(
            np.mean([r.intervention_step for r in valid_records])
        )
        stats["mean_replay_steps"] = float(
            np.mean([r.replay_steps for r in valid_records])
        )

    analytic_ok = off.size >= 2 and on.size >= 2
    if analytic_ok:
        s_analytic = math.sqrt(
            float(off.var(ddof=1)) / off.size + float(on.var(ddof=1)) / on.size
        )
        stats["s_analytic"] = s_analytic
        stats["s_hat"] = s_analytic
        stats["valid"] = float(
            math.isfinite(stats["mu_hat"]) and math.isfinite(s_analytic)
        )

    if settings.se_mode == "cluster_bootstrap":
        bootstrap_rng = rng if rng is not None else np.random.default_rng(0)
        se, low, high = _cluster_bootstrap(
            records, samples=settings.bootstrap_samples, rng=bootstrap_rng
        )
        stats["s_hat"] = se
        stats["mu_ci_low"] = low
        stats["mu_ci_high"] = high
        stats["valid"] = float(
            math.isfinite(stats["mu_hat"]) and math.isfinite(se) and se >= 0.0
        )
    return stats


def probe_prior_utility(
    env_factory: Callable[[int], Any],
    learner,
    settings: SAPRSettings,
    *,
    n_probes: Optional[int] = None,
    seed_base: Optional[int] = None,
    gamma: Optional[float] = None,
) -> Dict[str, Any]:
    """Run ``n_probes`` valid random-intervention probes on fresh environments."""
    if getattr(learner, "offline_q_network", None) is None:
        raise RuntimeError(
            "prior utility probe needs a frozen offline_q_network on the learner"
        )
    if int(learner.n_actions) <= 1:
        raise ValueError("prior utility probe requires discrete actions")

    target = int(settings.probe_episodes if n_probes is None else n_probes)
    if target < 4:
        raise ValueError(
            f"n_probes must be at least 4 (two per treatment group), got {target}"
        )
    base = int(settings.probe_seed_base if seed_base is None else seed_base)
    gamma = float(learner.gamma if gamma is None else gamma)
    temperature = float(settings.temperature)
    support_floor = float(settings.support_floor)
    behaviour = settings.probe_behaviour

    frozen_online = copy.deepcopy(learner.q_network).to(learner.device)
    frozen_online.eval()
    offline_net = learner.offline_q_network
    offline_net.eval()

    seed_rng = np.random.default_rng(base)
    records: List[ProbeRecord] = []
    attempts = 0
    invalid = 0
    env_steps = 0
    max_attempts = max(target, int(math.ceil(target * settings.probe_max_attempt_factor)))

    while len(records) < target and attempts < max_attempts:
        attempts += 1
        env_seed = int(seed_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng((base, env_seed, attempts))
        base_env = env_factory(env_seed)
        replay_env = env_factory(env_seed)
        ret: Optional[float] = None
        replay_steps = 0
        try:
            with torch.inference_mode():
                base_steps = _base_length(
                    base_env,
                    frozen_online,
                    learner,
                    behaviour=behaviour,
                    temperature=temperature,
                    support_floor=support_floor,
                    rng=rng,
                )
                env_steps += int(base_steps)
                if base_steps <= 0:
                    invalid += 1
                    continue
                t_int = int(rng.integers(0, base_steps))
                z = int(rng.integers(0, 2))
                agents = sorted(
                    f"agent_{i}" for i in range(int(learner.n_agents))
                )
                actor = agents[int(rng.integers(0, len(agents)))]
                ret, replay_steps = _replay_probe(
                    replay_env,
                    frozen_online,
                    offline_net,
                    learner,
                    t_int=t_int,
                    actor=actor,
                    z=z,
                    gamma=gamma,
                    behaviour=behaviour,
                    temperature=temperature,
                    support_floor=support_floor,
                    rng=rng,
                )
                env_steps += int(replay_steps)
        finally:
            _close_env(base_env)
            _close_env(replay_env)
        if ret is None:
            invalid += 1
            continue
        records.append(
            ProbeRecord(
                episode=attempts,
                z=z,
                agent=agents.index(actor),
                intervention_step=t_int,
                ret=float(ret),
                base_steps=int(base_steps),
                replay_steps=int(replay_steps),
            )
        )

    bootstrap_rng = np.random.default_rng((base, 17))
    stats: Dict[str, Any] = estimate_prior_utility(
        records, settings, rng=bootstrap_rng
    )
    stats.update(
        {
            "probe_env_steps": int(env_steps),
            "probe_attempts": int(attempts),
            "probe_invalid": int(invalid),
            "probe_target": int(target),
            "records": records,
        }
    )
    return stats
