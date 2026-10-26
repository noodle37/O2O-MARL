# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Controller wiring the joint proposer and the version-boundary judgement.

Training phase: ``policy`` is handed to ``collect_parallel`` and proposes joint
actions. Version boundary: the online parameters are frozen, two paired on-policy
arms estimate tau_v, and the proposer is permanently disabled once the offline
joint behaviour has no measurable positive value. After the exit the controller
runs no further boundaries, so the run degrades to plain online MARL.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from fhjp.eval_arms import paired_arm_estimate, should_exit
from fhjp.proposer import JointProposalPolicy


class ProposerController:
    """Owns the proposal policy, the version counter and the exit decision."""

    def __init__(
        self,
        learner,
        block: Dict[str, Any],
        env_factory: Callable[[int], Any],
    ) -> None:
        block = dict(block or {})
        self.learner = learner
        self.env_factory = env_factory
        self.version_len = int(block.get("version_len", 0) or 0)
        if self.version_len <= 0:
            raise ValueError(
                "proposer.version_len must be positive: a single version boundary "
                "over the whole online budget means the versioned re-judgement and "
                "the finite handover never actually happen"
            )
        self.eval_episodes_per_arm = int(block.get("eval_episodes_per_arm", 32))
        self.min_eval_episodes = int(block.get("min_eval_episodes", 16))
        self.alpha = float(block.get("alpha", 0.05))
        self.eval_seed_base = int(block.get("eval_seed_base", 0) or 0)
        self.count_eval_in_budget = bool(block.get("count_eval_in_budget", True))
        if self.min_eval_episodes > self.eval_episodes_per_arm:
            raise ValueError(
                "proposer.min_eval_episodes cannot exceed "
                "proposer.eval_episodes_per_arm: the exit test would never fire"
            )
        self.policy = JointProposalPolicy(learner, p=float(block.get("p", 0.5)))
        self.version = 0
        self.exited = False
        self.last_version_t = 0
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ state

    @property
    def proposer_on(self) -> bool:
        return self.policy.proposer_on

    def due(self, t_env: int) -> bool:
        """True at a version boundary that still needs a judgement."""
        if self.exited or self.version_len <= 0:
            return False
        return t_env - self.last_version_t >= self.version_len

    # --------------------------------------------------------------- boundary

    def run_boundary(self, t_env: int) -> Tuple[Dict[str, float], int]:
        """Run one version boundary; returns (logs, eval env steps)."""
        self.version += 1
        estimate = paired_arm_estimate(
            self.env_factory,
            self.learner,
            n_episodes=self.eval_episodes_per_arm,
            # Per-version seed base: reproducible, distinct across versions and
            # independent of the training seed stream.
            seed_base=self.eval_seed_base + self.version,
            gamma=float(self.learner.gamma),
        )
        exit_now, logs = should_exit(
            estimate,
            min_eval_episodes=self.min_eval_episodes,
            alpha=self.alpha,
        )
        logs.update(self.policy.stats(reset=True))
        logs.update(
            {
                "proposer/version": float(self.version),
                "proposer/proposer_on": float(not exit_now),
                "proposer/eval_env_steps": float(estimate["eval_env_steps"]),
                "t_env": t_env,
            }
        )
        if exit_now:
            # Irreversible: no further proposals and no further boundaries.
            self.policy.disable()
            self.exited = True
        self.history.append(
            {
                "version": float(self.version),
                "t_env": float(t_env),
                "tau_v": logs["proposer/tau_v"],
                "tau_v_ub": logs["proposer/tau_v_ub"],
                "exited": float(exit_now),
            }
        )
        return logs, int(estimate["eval_env_steps"])

    def boundary_logs(self, t_env: int) -> Tuple[Optional[Dict[str, float]], int]:
        """Run the boundary when due; otherwise report nothing."""
        if not self.due(t_env):
            return None, 0
        logs, eval_steps = self.run_boundary(t_env)
        if self.count_eval_in_budget:
            return logs, eval_steps
        return logs, 0


def build_controller(
    cfg: Dict[str, Any], learner, env_factory: Callable[[int], Any]
) -> Optional[ProposerController]:
    """Build the controller when the config enables the proposer."""
    block = cfg.get("proposer") or {}
    if not bool(block.get("enabled", False)):
        return None
    return ProposerController(learner, block, env_factory)
