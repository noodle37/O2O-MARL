# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Finite Handover of Joint Proposals (FHJP).

Training proposes two complete joint actions and executes one at p = 0.5; the
version boundary judges the offline joint behaviour with two paired on-policy
rollouts and permanently retires the proposer once it has no measurable value.
"""

from fhjp.controller import ProposerController, build_controller
from fhjp.eval_arms import (
    arm_return,
    one_sided_upper_bound,
    paired_arm_estimate,
    should_exit,
)
from fhjp.proposer import JointProposalPolicy, joint_greedy

__all__ = [
    "ProposerController",
    "build_controller",
    "JointProposalPolicy",
    "joint_greedy",
    "arm_return",
    "paired_arm_estimate",
    "one_sided_upper_bound",
    "should_exit",
]
