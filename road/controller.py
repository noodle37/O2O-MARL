# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""ROAD outer controller: picks the offline episode fraction per period.

The inner learner is untouched. Each period keeps one arm of Lambda fixed for
``period_steps`` gradient steps; at the boundary two no-grad QMixer forwards on
fresh offline and online batches give R_q, which updates the sliding-window UCB.
All sampling here uses the controller's own RNG so enabling ROAD never perturbs
the training batch stream of the fixed-ratio baselines.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from road.bandit import SlidingWindowUCB
from road.surrogate import joint_advantage_gap, surrogate_reward, transition_mask
from training.rollout import stack_episodes

DEFAULT_ARMS = (0.1, 0.2, 0.3, 0.4, 0.5)


class RoadController:
    """Owns the bandit, the current ratio and the period counters."""

    def __init__(self, learner, block: Dict[str, Any], seed: int = 0) -> None:
        block = dict(block or {})
        self.learner = learner
        self.arms = [float(arm) for arm in block.get("arms", DEFAULT_ARMS)]
        if not self.arms:
            raise ValueError("road.arms cannot be empty")
        if any(arm < 0.0 or arm > 1.0 for arm in self.arms):
            raise ValueError(f"road.arms must lie in [0, 1], got {self.arms}")
        self.kappa = float(block.get("kappa", 1.0))
        self.c = float(block.get("ucb_c", 2.0))
        self.window = int(block.get("window", 200))
        self.period_steps = int(block.get("period_steps", 10))
        self.delta_batch = int(block.get("delta_batch", 32))
        self.normalize_reward = bool(block.get("normalize_reward", False))
        self.mask_explored = bool(block.get("mask_explored", False))
        self.seed_offset = int(block.get("seed_offset", 7))
        for name, value in (
            ("window", self.window),
            ("period_steps", self.period_steps),
            ("delta_batch", self.delta_batch),
        ):
            if value < 1:
                raise ValueError(f"road.{name} must be positive, got {value}")
        if self.kappa < 0.0:
            raise ValueError(f"road.kappa must be non-negative, got {self.kappa}")
        if self.c < 0.0:
            raise ValueError(f"road.ucb_c must be non-negative, got {self.c}")

        self.bandit = SlidingWindowUCB(
            self.arms, c=self.c, window=self.window, seed=int(seed) + self.seed_offset
        )
        # Independent reward-estimation stream: never the online loop's rng.
        self.rng = np.random.default_rng(int(seed) + self.seed_offset)
        self.ratio = self.bandit.select()
        self.period = 0
        self.steps_in_period = 0
        self.reward_window: deque = deque(maxlen=self.window)
        self.last_row: Optional[Dict[str, float]] = None

    # ----------------------------------------------------------------- period

    def _normalized_reward(self, r_q: float) -> float:
        if not self.normalize_reward:
            return r_q
        values = np.asarray(self.reward_window, dtype=np.float64)
        if values.size < 2:
            return 0.0
        std = float(values.std())
        if std < 1e-8:
            return 0.0
        return float((r_q - float(values.mean())) / std)

    def _delta_batch(self, episodes, max_seq: int) -> Dict[str, Any]:
        return stack_episodes(episodes, max_seq, self.learner.device)

    def after_train_step(
        self, offline_data, online_buffer, t_env: int
    ) -> Optional[Dict[str, float]]:
        """Count one gradient step; at the period boundary return the log row."""
        self.steps_in_period += 1
        if self.steps_in_period < self.period_steps:
            return None
        if len(online_buffer) == 0 or offline_data is None:
            self.steps_in_period = 0
            return None

        max_seq = int(offline_data.stats.max_seq_length)
        off_batch = offline_data.sample(
            self.delta_batch, self.rng, device=self.learner.device
        )
        n_on = min(self.delta_batch, len(online_buffer))
        on_idx = self.rng.choice(len(online_buffer), size=n_on, replace=False)
        on_batch = self._delta_batch(
            [online_buffer[int(i)] for i in on_idx], max_seq
        )

        delta_off = joint_advantage_gap(self.learner, off_batch)
        extra_mask = None
        # Always report the exploration share for diagnostics; masking it out is
        # an appendix variant, not part of the paper's MaROAD reward.
        explored_frac = 0.0
        if "explored" in on_batch:
            explored = on_batch["explored"].float()
            valid = transition_mask(on_batch)
            denom = valid.sum().clamp(min=1.0)
            explored_frac = float((valid * explored).sum() / denom)
            if self.mask_explored:
                extra_mask = 1.0 - explored
        delta_on = joint_advantage_gap(self.learner, on_batch, extra_mask=extra_mask)
        r_q = surrogate_reward(delta_off, delta_on, self.kappa)

        self.reward_window.append(r_q)
        r_q_norm = self._normalized_reward(r_q)
        self.bandit.update(self.ratio, r_q_norm)

        previous = self.ratio
        self.period += 1
        self.ratio = self.bandit.select()
        self.steps_in_period = 0

        # Align the row with the bandit update: road/r_q and the UCB target
        # belong to the period that just ended, which ran with `previous`.
        # road/ratio therefore names the recorded arm, while road/next_ratio is
        # the arm selected for the next period.
        row: Dict[str, float] = {
            "road/period": float(self.period),
            "road/ratio": float(previous),
            "road/next_ratio": float(self.ratio),
            "road/delta_off": float(delta_off),
            "road/delta_on": float(delta_on),
            "road/r_q": float(r_q),
            "road/r_q_norm": float(r_q_norm),
            "road/explored_frac_on": float(explored_frac),
            "road/window_n": float(len(self.reward_window)),
            "road/count_total": float(sum(self.bandit.total_counts())),
            "t_env": float(t_env),
        }
        for arm, ucb in self.bandit.ucb_values().items():
            row[f"road/ucb_{arm:g}"] = float(min(ucb, 1e9))
        for arm, count in self.bandit.window_counts().items():
            row[f"road/count_{arm:g}"] = float(count)
        self.last_row = row
        return row

    # ------------------------------------------------------------- diagnostics

    def summary(self) -> Dict[str, float]:
        return {
            f"road/count_total_{arm:g}": float(count)
            for arm, count in zip(self.arms, self.bandit.total_counts())
        }

    # ---------------------------------------------------------------- snapshot

    def state_dict(self) -> Dict[str, Any]:
        return {
            "bandit": self.bandit.state_dict(),
            "ratio": float(self.ratio),
            "period": int(self.period),
            "steps_in_period": int(self.steps_in_period),
            "reward_window": [float(r) for r in self.reward_window],
            "rng": self.rng.bit_generator.state,
        }


def _check_exclusive(cfg: Dict[str, Any], learner) -> None:
    learner_type = str(cfg.get("learner_type", "ovmse_paper")).lower()
    if learner_type == "fsr" or hasattr(learner, "support_mix_ratio"):
        raise ValueError(
            "road.enabled is mutually exclusive with FSR: both controllers set step_mix"
        )
    if int(cfg.get("fork_at", -1) or -1) > 0 or cfg.get("resume_fork"):
        raise ValueError(
            "road.enabled does not support fork/resume: the bandit state is not snapshot"
        )
    if not hasattr(learner, "_forward_qs") or not hasattr(learner, "mixer"):
        raise ValueError(
            "road needs a QMIX learner with _forward_qs and a monotone mixer"
        )


def build_road_controller(cfg: Dict[str, Any], learner) -> Optional[RoadController]:
    """Build the controller when ``road.enabled``; None otherwise."""
    block = cfg.get("road") or {}
    if not bool(block.get("enabled", False)):
        return None
    _check_exclusive(cfg, learner)
    return RoadController(learner, block, seed=int(cfg.get("seed", 0)))
