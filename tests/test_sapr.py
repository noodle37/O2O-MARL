#!/usr/bin/env python3
# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Unit tests for SAPR-MARL (Parts I-III).

Everything runs on CPU against constant-Q fake networks and a fake environment.
The tests pin the three pieces of the algorithm:

* Part I   - the legal-action softmax policy and its forward KL to the frozen prior.
* Part II  - the random local-intervention probe and its difference-in-means
             estimator with an analytic (or clustered-bootstrap) standard error.
* Part III - the bounded, smooth adaptive controller for the KL weight.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sapr.controller import (
    SAPRSettings,
    SmoothAdaptiveController,
    build_sapr_controller,
)
from sapr.probe import (
    ProbeRecord,
    _replay_probe,
    estimate_prior_utility,
    probe_prior_utility,
)
from sapr.regularizer import masked_policy, prior_kl

N_AGENTS = 2
N_ACTIONS = 2
OBS_DIM = 4
AGENTS = [f"agent_{i}" for i in range(N_AGENTS)]
GAMMA = 0.9


# --------------------------------------------------------------------- fixtures


class ConstantNet(torch.nn.Module):
    """Q network with a fixed per-action value; records its history inputs."""

    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.float32))
        self.seen_last_actions: list[int] = []

    def initial_state(self, batch_size, device):
        return None

    def forward(self, obs_in, hidden):
        self.seen_last_actions.append(int(obs_in[0, OBS_DIM:].argmax().item()))
        return self.q.unsqueeze(0), hidden


class FakeLearner:
    """Minimal learner surface used by the probe."""

    def __init__(self, offline_q, online_q, gamma=GAMMA):
        self.n_agents = N_AGENTS
        self.n_actions = N_ACTIONS
        self.device = torch.device("cpu")
        self.add_agent_id_to_obs = False
        self.gamma = gamma
        self.q_network = ConstantNet(online_q)
        self.offline_q_network = ConstantNet(offline_q)
        self.train_calls = 0

    def train_on_batch(self, *args, **kwargs):
        self.train_calls += 1
        raise AssertionError("probing must not perform gradient updates")


class ProbeEnv:
    """Deterministic-enough env that records the joint actions it receives."""

    def __init__(self, seed=0, horizon=3, reward_fn=None, dead_agent=None):
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.reward_fn = reward_fn or (lambda actions, t: 0.0)
        self.dead_agent = dead_agent
        self.t = 0
        self.actions: list[dict] = []
        self.closed = False

    def reset(self):
        self.t = 0
        self.actions = []
        return self._obs(), self._info()

    def _obs(self):
        return {a: np.zeros(OBS_DIM, dtype=np.float32) for a in AGENTS}

    def _info(self):
        legals = {}
        for a in AGENTS:
            if a == self.dead_agent:
                legals[a] = np.zeros(N_ACTIONS, dtype=np.float32)
            else:
                legals[a] = np.ones(N_ACTIONS, dtype=np.float32)
        return {"legals": legals, "state": np.zeros(8, dtype=np.float32)}

    def step(self, actions):
        self.actions.append(dict(actions))
        reward = float(self.reward_fn(actions, self.t))
        self.t += 1
        terminal = self.t >= self.horizon
        return (
            self._obs(),
            {a: np.array(reward, dtype=np.float32) for a in AGENTS},
            {a: np.array(terminal) for a in AGENTS},
            {a: np.array(False) for a in AGENTS},
            self._info(),
        )

    def close(self):
        self.closed = True


def _settings(**overrides) -> SAPRSettings:
    block = {
        "enabled": True,
        "temperature": 1.0,
        "support_floor": 1e-8,
        "lambda_base": 0.1,
        "lambda_min": 0.0,
        "lambda_max": 0.5,
        "delta_lambda": 0.1,
        "alpha": 0.5,
        "beta": 0.5,
        "rho": 0.5,
        "eps": 1e-8,
        "interval_steps": 100,
        "count_probe_in_budget": True,
        "probe_episodes": 8,
        "probe_seed_base": 1234,
        "probe_behaviour": "greedy",
        "probe_max_attempt_factor": 3.0,
        "se_mode": "analytic",
        "bootstrap_samples": 50,
    }
    block.update(overrides)
    return SAPRSettings.from_cfg({"sapr": block})


# ---------------------------------------------------------------- Part III: controller


def test_controller_initial_state_and_due():
    controller = SmoothAdaptiveController(_settings())
    assert controller.lambda_value == 0.1
    assert controller.m == 0.0 and controller.v == 0.0
    assert controller.version == 0
    assert controller.due(99) is False
    assert controller.due(100) is True


def test_controller_update_math():
    settings = _settings(lambda_base=0.1, delta_lambda=0.1, alpha=0.5, beta=0.5, rho=0.5)
    controller = SmoothAdaptiveController(settings)
    logs = controller.update(mu_hat=2.0, s_hat=1.0)

    m = 0.5 * 0.0 + 0.5 * 2.0
    v = 0.25 * 0.0 + 0.25 * 1.0
    z = m / math.sqrt(v + settings.eps)
    lambda_star = 0.1 + 0.1 * math.tanh(0.5 * z)
    expected = 0.5 * 0.1 + 0.5 * lambda_star

    assert abs(controller.m - m) < 1e-12
    assert abs(controller.v - v) < 1e-12
    assert abs(logs["sapr/z"] - z) < 1e-9
    assert abs(logs["sapr/lambda_star"] - lambda_star) < 1e-12
    assert abs(controller.lambda_value - expected) < 1e-12
    assert logs["sapr/updated"] == 1.0
    assert logs["sapr/version"] == 1.0
    assert controller.version == 1

    # Second recursion round uses the stored m, v.
    logs2 = controller.update(mu_hat=-0.5, s_hat=0.5)
    m2 = 0.5 * m + 0.5 * (-0.5)
    v2 = 0.25 * v + 0.25 * 0.25
    assert abs(controller.m - m2) < 1e-12
    assert abs(controller.v - v2) < 1e-12
    assert logs2["sapr/version"] == 2.0


def test_controller_bounds_monotonicity_and_uncertainty_suppression():
    lo = SmoothAdaptiveController(_settings())
    mid = SmoothAdaptiveController(_settings())
    hi = SmoothAdaptiveController(_settings())
    mid.update(mu_hat=5.0, s_hat=1.0)
    hi.update(mu_hat=5.0, s_hat=0.01)
    lo.update(mu_hat=-5.0, s_hat=1.0)
    assert lo.lambda_value < mid.lambda_value < hi.lambda_value
    # Lower uncertainty moves the weight further for the same signal.
    assert abs(hi.lambda_value - 0.1) > abs(mid.lambda_value - 0.1)
    # Repeated saturating updates never leave the bounds.
    for _ in range(200):
        hi.update(mu_hat=1e6, s_hat=1e-6)
        lo.update(mu_hat=-1e6, s_hat=1e-6)
    assert 0.0 <= lo.lambda_value <= 0.5
    assert 0.0 <= hi.lambda_value <= 0.5
    # The tanh target is confined to [base - delta, base + delta] = [0.0, 0.2],
    # and the convex smoothing converges to it.
    assert abs(hi.lambda_value - 0.2) < 1e-6
    assert abs(lo.lambda_value - 0.0) < 1e-6


def test_controller_skips_without_moving_state():
    controller = SmoothAdaptiveController(_settings())
    before = (controller.m, controller.v, controller.lambda_value)
    logs = controller.no_update(reason="insufficient_samples")
    assert (controller.m, controller.v, controller.lambda_value) == before
    assert logs["sapr/updated"] == 0.0
    assert logs["sapr/skip_reason"] == "insufficient_samples"
    assert controller.version == 1

    logs = controller.update(mu_hat=float("nan"), s_hat=1.0)
    assert logs["sapr/updated"] == 0.0
    assert (controller.m, controller.v, controller.lambda_value) == before

    logs = controller.update(mu_hat=1.0, s_hat=float("inf"))
    assert logs["sapr/updated"] == 0.0
    logs = controller.update(mu_hat=1.0, s_hat=-0.5)
    assert logs["sapr/updated"] == 0.0
    assert (controller.m, controller.v, controller.lambda_value) == before


def test_controller_state_roundtrip_and_clamping():
    controller = SmoothAdaptiveController(_settings())
    controller.update(mu_hat=3.0, s_hat=0.5)
    controller.last_probe_t = 250
    state = controller.state_dict()

    restored = SmoothAdaptiveController(_settings())
    restored.load_state_dict(state)
    assert restored.lambda_value == controller.lambda_value
    assert restored.m == controller.m and restored.v == controller.v
    assert restored.version == controller.version
    assert restored.last_probe_t == 250

    restored.load_state_dict({"lambda": 99.0})
    assert restored.lambda_value == 0.5


def test_settings_validation():
    bad_blocks = [
        {"temperature": 0.0},
        {"support_floor": 1.0},
        {"lambda_min": 0.2, "lambda_base": 0.1},
        {"lambda_max": 0.05, "lambda_base": 0.1},
        {"delta_lambda": 0.2},  # > min(base - min, max - base)
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"beta": 0.0},
        {"rho": 0.0},
        {"eps": 0.0},
        {"interval_steps": 0},
        {"probe_episodes": 3},
        {"probe_episodes": 1},
        {"probe_behaviour": "epsilon_greedy"},
        {"probe_max_attempt_factor": 0.5},
        {"se_mode": "jackknife"},
        {"se_mode": "cluster_bootstrap", "bootstrap_samples": 1},
        {"lambda_base": float("inf")},
    ]
    for bad in bad_blocks:
        try:
            _settings(**bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"expected validation failure for {bad}")

    assert build_sapr_controller({"sapr": {"enabled": False}}) is None
    assert build_sapr_controller({}) is None


def test_shipped_config_is_valid():
    OmegaConf = __import__("omegaconf").OmegaConf
    cfg = OmegaConf.load(str(ROOT / "configs" / "sapr_5m_vs_6m.yaml"))
    settings = SAPRSettings.from_cfg(cfg)
    assert settings.enabled
    assert settings.lambda_min <= settings.lambda_base <= settings.lambda_max
    assert settings.delta_lambda <= min(
        settings.lambda_base - settings.lambda_min,
        settings.lambda_max - settings.lambda_base,
    )
    assert bool(cfg["system"]["use_ovm"]) is False
    assert float(cfg["system"]["cql_weight"]) == 0.0
    assert str(cfg["transfer"]["mode"]) == "full_td"


# ------------------------------------------------------------- Part I: prior KL


def test_masked_policy_zeroes_illegal_actions():
    q = torch.tensor([[1.0, 2.0, 3.0]])
    avail = torch.tensor([[True, False, True]])
    probs = masked_policy(q, avail, temperature=1.0)
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    assert float(probs[0, 1]) == 0.0
    # Softmax is computed over the legal actions only.
    legal = torch.softmax(q[avail], dim=-1)
    assert torch.allclose(probs[avail], legal, atol=1e-6)


def test_prior_kl_zero_when_policies_match():
    torch.manual_seed(0)
    q = torch.randn(3, 5, N_AGENTS, 3)
    avail = torch.ones(3, 5, N_AGENTS, 3, dtype=torch.bool)
    mask = torch.ones(3, 5)
    assert float(prior_kl(q, q.clone(), avail, mask, 1.0)) < 1e-7

    # Identical Q distributions conditioned on different history still match
    # only if the values match; here we only test the zero-KL identity above.


def test_prior_kl_matches_manual_two_action_value():
    p_on = torch.softmax(torch.tensor([[1.0, 0.0]]), dim=-1)
    p_off = torch.softmax(torch.tensor([[0.0, 1.0]]), dim=-1)
    expected = float((p_on * (torch.log(p_on) - torch.log(p_off))).sum())
    got = float(
        prior_kl(
            torch.tensor([[[[1.0, 0.0]]]]),
            torch.tensor([[[[0.0, 1.0]]]]),
            torch.ones(1, 1, 1, 2, dtype=torch.bool),
            torch.ones(1, 1),
            1.0,
        )
    )
    assert abs(got - expected) < 1e-6
    assert got > 0.0


def test_prior_kl_masks_steps_and_dead_agents():
    # Step 0 has a large KL, step 1 is masked out; a dead agent is excluded.
    q_on = torch.zeros(1, 2, 2, 2)
    q_off = torch.zeros(1, 2, 2, 2)
    q_off[0, 0] = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    avail = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    avail[0, 1, 1] = False  # agent 1 dead on the masked step
    step0 = float(
        prior_kl(q_on[:, :1], q_off[:, :1], avail[:, :1], torch.ones(1, 1), 1.0)
    )
    both = float(prior_kl(q_on, q_off, avail, torch.tensor([[1.0, 0.0]]), 1.0))
    assert abs(both - step0) < 1e-6

    # A dead agent contributes no KL even when its Q values differ wildly.
    q_off_dead = q_off.clone()
    q_off_dead[0, 0, 1] = torch.tensor([100.0, -100.0])
    avail_dead = avail.clone()
    avail_dead[0, 0, 1] = False
    alive_only = float(
        prior_kl(q_on[:, :1], q_off_dead[:, :1], avail_dead[:, :1], torch.ones(1, 1), 1.0)
    )
    # Averaging over the only live agent equals the per-agent KL, so the dead
    # agent's extreme Q values leave the regularizer unchanged.
    assert abs(alive_only - step0) < 1e-6


def test_prior_kl_support_floor_keeps_finite():
    q_on = torch.tensor([[[[10.0, -10.0]]]])
    q_off = torch.tensor([[[[-10.0, 10.0]]]])
    avail = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    mask = torch.ones(1, 1)
    raw = float(prior_kl(q_on, q_off, avail, mask, 1.0, support_floor=0.0))
    floored = float(prior_kl(q_on, q_off, avail, mask, 1.0, support_floor=0.01))
    assert math.isfinite(raw) and math.isfinite(floored)
    assert 0.0 < floored < raw


# ------------------------------------------------------- Part II: probe mechanics


def _replay_learner():
    # Online greedily prefers action 1, offline prefers action 0.
    return FakeLearner(offline_q=[4.0, 0.0], online_q=[0.0, 4.0])


def test_replay_intervenes_on_exactly_one_agent():
    learner = _replay_learner()
    env = ProbeEnv(horizon=3)
    ret, steps = _replay_probe(
        env,
        learner.q_network,
        learner.offline_q_network,
        learner,
        t_int=1,
        actor="agent_0",
        z=1,
        gamma=GAMMA,
        behaviour="greedy",
        temperature=1.0,
        support_floor=0.0,
        rng=np.random.default_rng(0),
    )
    assert steps == 3
    assert ret == 0.0
    # Before the intervention everyone follows the online greedy action.
    assert env.actions[0] == {"agent_0": 1, "agent_1": 1}
    # At the intervention only the chosen unit switches to the prior.
    assert env.actions[1] == {"agent_0": 0, "agent_1": 1}
    # Afterwards everyone resumes the online policy.
    assert env.actions[2] == {"agent_0": 1, "agent_1": 1}


def test_replay_z_zero_keeps_online_action():
    learner = _replay_learner()
    env = ProbeEnv(horizon=2)
    _replay_probe(
        env,
        learner.q_network,
        learner.offline_q_network,
        learner,
        t_int=0,
        actor="agent_1",
        z=0,
        gamma=GAMMA,
        behaviour="greedy",
        temperature=1.0,
        support_floor=0.0,
        rng=np.random.default_rng(0),
    )
    assert env.actions[0] == {"agent_0": 1, "agent_1": 1}


def test_replay_conditions_online_history_on_executed_actions():
    learner = _replay_learner()
    env = ProbeEnv(horizon=3)
    _replay_probe(
        env,
        learner.q_network,
        learner.offline_q_network,
        learner,
        t_int=1,
        actor="agent_0",
        z=1,
        gamma=GAMMA,
        behaviour="greedy",
        temperature=1.0,
        support_floor=0.0,
        rng=np.random.default_rng(0),
    )
    # Three decision steps, two agents; forward history is the executed action.
    seen = learner.q_network.seen_last_actions
    assert seen[0] == 0 and seen[1] == 0  # step 0: empty history one-hot
    assert seen[2] == 1 and seen[3] == 1  # step 1 sees step-0 online action
    assert seen[4] == 0 and seen[5] == 1  # step 2 sees the intervention


def test_replay_returns_none_when_episode_ends_before_intervention():
    learner = _replay_learner()
    env = ProbeEnv(horizon=2)
    ret, steps = _replay_probe(
        env,
        learner.q_network,
        learner.offline_q_network,
        learner,
        t_int=5,
        actor="agent_0",
        z=1,
        gamma=GAMMA,
        behaviour="greedy",
        temperature=1.0,
        support_floor=0.0,
        rng=np.random.default_rng(0),
    )
    assert ret is None
    assert steps == 2


def test_replay_rejects_dead_actor():
    learner = _replay_learner()
    env = ProbeEnv(horizon=3, dead_agent="agent_0")
    ret, _ = _replay_probe(
        env,
        learner.q_network,
        learner.offline_q_network,
        learner,
        t_int=1,
        actor="agent_0",
        z=1,
        gamma=GAMMA,
        behaviour="greedy",
        temperature=1.0,
        support_floor=0.0,
        rng=np.random.default_rng(0),
    )
    assert ret is None


def test_probe_discounted_return_starts_at_intervention():
    # Reward 1 for every agent action 0 at every step; offline plays 0.
    def reward_fn(actions, t):
        return float(np.mean([int(a) == 0 for a in actions.values()]))

    learner = _replay_learner()
    env = ProbeEnv(horizon=3, reward_fn=reward_fn)
    ret, _ = _replay_probe(
        env,
        learner.q_network,
        learner.offline_q_network,
        learner,
        t_int=1,
        actor="agent_0",
        z=1,
        gamma=GAMMA,
        behaviour="greedy",
        temperature=1.0,
        support_floor=0.0,
        rng=np.random.default_rng(0),
    )
    assert abs(ret - 0.5) < 1e-6


def test_probe_estimator_points_toward_the_better_prior():
    def reward_fn(actions, t):
        return float(np.mean([int(a) == 1 for a in actions.values()]))

    # Offline plays 0, online plays 1 -> the prior is worse.
    learner = _replay_learner()
    stats = probe_prior_utility(
        lambda seed: ProbeEnv(seed=seed, horizon=3, reward_fn=reward_fn),
        learner,
        _settings(probe_episodes=32),
        gamma=GAMMA,
    )
    assert stats["valid"] == 1.0
    assert stats["n_off"] + stats["n_on"] == 32
    assert stats["mu_hat"] < 0.0
    assert math.isfinite(stats["s_hat"])
    assert stats["probe_env_steps"] == sum(
        r.base_steps + r.replay_steps for r in stats["records"]
    )
    assert stats["probe_attempts"] == 32 and stats["probe_invalid"] == 0

    # Swap the controller preferences: now the prior is better.
    flipped = FakeLearner(offline_q=[0.0, 4.0], online_q=[4.0, 0.0])
    flipped_stats = probe_prior_utility(
        lambda seed: ProbeEnv(seed=seed, horizon=3, reward_fn=reward_fn),
        flipped,
        _settings(probe_episodes=32, probe_seed_base=4321),
        gamma=GAMMA,
    )
    assert flipped_stats["mu_hat"] > 0.0


def test_probe_is_side_effect_free_and_reproducible():
    def reward_fn(actions, t):
        return float(np.mean([int(a) == 0 for a in actions.values()]))

    learner = _replay_learner()
    before = copy.deepcopy(learner.q_network.state_dict())
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    a = probe_prior_utility(
        lambda seed: ProbeEnv(seed=seed, horizon=4, reward_fn=reward_fn),
        learner,
        _settings(probe_episodes=8),
        gamma=GAMMA,
    )
    b = probe_prior_utility(
        lambda seed: ProbeEnv(seed=seed, horizon=4, reward_fn=reward_fn),
        learner,
        _settings(probe_episodes=8),
        gamma=GAMMA,
    )

    assert learner.train_calls == 0
    for key, value in learner.q_network.state_dict().items():
        assert torch.equal(value, before[key])
    after_np = np.random.get_state()
    assert np.array_equal(np_state[0], after_np[0])
    assert np.array_equal(np_state[1], after_np[1])
    assert np_state[2] == after_np[2]
    assert torch.equal(torch_state, torch.get_rng_state())

    assert [r.ret for r in a["records"]] == [r.ret for r in b["records"]]
    assert [r.episode for r in a["records"]] == [r.episode for r in b["records"]]
    if math.isnan(a["mu_hat"]):
        assert math.isnan(b["mu_hat"])
    else:
        assert a["mu_hat"] == b["mu_hat"]


def test_probe_envs_are_fresh_and_closed():
    created: list[ProbeEnv] = []

    def factory(seed):
        env = ProbeEnv(seed=seed, horizon=2)
        created.append(env)
        return env

    learner = _replay_learner()
    stats = probe_prior_utility(factory, learner, _settings(probe_episodes=4), gamma=GAMMA)
    assert len(created) == 2 * 4
    assert all(env.closed for env in created)
    assert stats["probe_invalid"] == 0


def test_probe_rejects_missing_offline_network():
    learner = _replay_learner()
    learner.offline_q_network = None
    try:
        probe_prior_utility(
            lambda seed: ProbeEnv(seed=seed), learner, _settings(probe_episodes=4)
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("the probe must require a frozen offline network")


# --------------------------------------------------------------------- estimator


def test_analytic_estimator_values():
    settings = _settings(probe_episodes=4)
    records = [
        ProbeRecord(0, 1, 0, 0, 1.0, 3, 3),
        ProbeRecord(1, 1, 1, 1, 2.0, 3, 3),
        ProbeRecord(2, 1, 0, 2, 3.0, 3, 3),
        ProbeRecord(3, 0, 1, 0, 0.0, 3, 3),
        ProbeRecord(4, 0, 0, 1, 1.0, 3, 3),
    ]
    stats = estimate_prior_utility(records, settings)
    assert stats["n_off"] == 3 and stats["n_on"] == 2
    assert stats["mu_hat"] == 2.0 - 0.5
    se = math.sqrt(np.var([1.0, 2.0, 3.0], ddof=1) / 3 + np.var([0.0, 1.0], ddof=1) / 2)
    assert abs(stats["s_hat"] - se) < 1e-12
    assert stats["valid"] == 1.0
    assert stats["mean_intervention_step"] == 0.8


def test_estimator_invalid_without_two_per_group():
    settings = _settings(probe_episodes=4)
    records = [
        ProbeRecord(0, 1, 0, 0, 1.0, 3, 3),
        ProbeRecord(1, 0, 0, 1, 0.0, 3, 3),
    ]
    stats = estimate_prior_utility(records, settings)
    assert stats["valid"] == 0.0
    assert stats["s_hat"] == float("inf")

    empty = estimate_prior_utility([], settings)
    assert empty["valid"] == 0.0
    assert math.isnan(empty["mu_hat"])


def test_cluster_bootstrap_estimator():
    settings = _settings(probe_episodes=4, se_mode="cluster_bootstrap", bootstrap_samples=200)
    # Two records per episode within each group.
    records = []
    episode = 0
    for z, base in ((1, 2.0), (0, 0.0)):
        for _ in range(3):
            for k in range(2):
                records.append(ProbeRecord(episode, z, k, k, base + 0.1 * episode, 3, 3))
            episode += 1
    stats = estimate_prior_utility(records, settings, rng=np.random.default_rng(0))
    assert stats["valid"] == 1.0
    assert math.isfinite(stats["s_hat"]) and stats["s_hat"] >= 0.0
    assert math.isfinite(stats["mu_ci_low"]) and math.isfinite(stats["mu_ci_high"])
    assert stats["mu_ci_low"] <= stats["mu_hat"] <= stats["mu_ci_high"]

    # Fewer than two clusters per group is undefined.
    thin = [
        ProbeRecord(0, 1, 0, 0, 1.0, 3, 3),
        ProbeRecord(0, 1, 1, 0, 1.0, 3, 3),
        ProbeRecord(1, 0, 0, 0, 0.0, 3, 3),
        ProbeRecord(2, 0, 1, 0, 0.0, 3, 3),
    ]
    thin_stats = estimate_prior_utility(thin, settings, rng=np.random.default_rng(1))
    assert thin_stats["valid"] == 0.0


# ------------------------------------------------------ learner integration (CPU)


def _dummy_batch(batch_size: int, seq: int, n_agents: int, obs_dim: int, state_dim: int, n_actions: int):
    device = torch.device("cpu")
    return {
        "obs": torch.randn(batch_size, seq, n_agents, obs_dim, device=device),
        "actions": torch.randint(0, n_actions, (batch_size, seq, n_agents, 1), device=device),
        "state": torch.randn(batch_size, seq, state_dim, device=device),
        "avail_actions": torch.ones(batch_size, seq, n_agents, n_actions, dtype=torch.bool, device=device),
        "reward": torch.randn(batch_size, seq, 1, device=device),
        "terminated": torch.zeros(batch_size, seq, 1, device=device),
        "filled": torch.ones(batch_size, seq, 1, device=device),
    }


def _learner_with_offline(seed: int = 0):
    from ovmse.learner import PaperQMIXLearner

    torch.manual_seed(seed)
    learner = PaperQMIXLearner(
        n_agents=3,
        n_actions=5,
        obs_dim=8,
        state_dim=16,
        device="cpu",
        cql_weight=0.0,
        use_ovm=False,
        add_agent_id_to_obs=True,
    )
    learner.offline_q_network = copy.deepcopy(learner.q_network).eval()
    learner.offline_mixer = copy.deepcopy(learner.mixer).eval()
    for param in learner.offline_q_network.parameters():
        param.requires_grad_(False)
    for param in learner.offline_mixer.parameters():
        param.requires_grad_(False)
    return learner


def test_learner_kl_is_zero_and_absent_without_weight():
    learner = _learner_with_offline()
    batch = _dummy_batch(4, 6, 3, 8, 16, 5)

    logs = learner.train_on_batch(batch, t_env=0)
    assert "sapr/prior_kl" not in logs
    assert learner.prior_kl_weight == 0.0

    # Restart from the same weights so the comparison is exact.
    learner = _learner_with_offline()
    learner.set_prior_weight(0.3)
    logs = learner.train_on_batch(batch, t_env=0)
    assert "sapr/prior_kl" in logs
    assert abs(logs["sapr/prior_kl"]) < 1e-7  # pi_online == pi_offline

    # Frozen prior weights are never updated by the optimizer.
    offline_before = copy.deepcopy(learner.offline_q_network.state_dict())
    learner.train_on_batch(batch, t_env=1)
    for key, value in learner.offline_q_network.state_dict().items():
        assert torch.equal(value, offline_before[key])


def test_learner_kl_adds_exactly_the_weighted_term():
    batch = _dummy_batch(4, 6, 3, 8, 16, 5)

    base = _learner_with_offline(seed=3)
    weighted = copy.deepcopy(base)
    for learner in (base, weighted):
        # Perturb the frozen prior so the KL is strictly positive.
        with torch.no_grad():
            learner.offline_q_network.fc2.bias.add_(0.5)
        learner.offline_q_network.eval()

    weight = 0.25
    weighted.set_prior_weight(weight)
    logs0 = base.train_on_batch(batch, t_env=0)
    logs_w = weighted.train_on_batch(batch, t_env=0)
    assert logs_w["sapr/prior_kl"] > 0.0
    delta = logs_w["loss"] - logs0["loss"]
    assert abs(delta - weight * logs_w["sapr/prior_kl"]) < 1e-5


def test_learner_rejects_positive_weight_without_offline_net():
    learner = _learner_with_offline()
    learner.offline_q_network = None
    learner.offline_mixer = None
    learner.set_prior_weight(0.1)
    try:
        learner.train_on_batch(_dummy_batch(2, 4, 3, 8, 16, 5), t_env=0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a positive KL weight without the frozen prior must fail loudly")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
    print("sapr tests ok")
