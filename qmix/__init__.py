"""Shared QMIX-family infrastructure (networks, unroll, mixing, epsilon-greedy policy)."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["QMIXBase", "QMIXCoreMixin", "MarlcRNN", "QMixer", "EvalPolicyMixin"]

if TYPE_CHECKING:
    from qmix.core import QMIXBase, QMIXCoreMixin
    from qmix.networks import MarlcRNN, QMixer
    from qmix.policy import EvalPolicyMixin


def __getattr__(name: str):
    if name in ("QMIXBase", "QMIXCoreMixin"):
        from qmix.core import QMIXBase as base_cls, QMIXCoreMixin as core_cls

        return base_cls if name == "QMIXBase" else core_cls
    if name in ("MarlcRNN", "QMixer"):
        from qmix import networks

        return getattr(networks, name)
    if name == "EvalPolicyMixin":
        from qmix.policy import EvalPolicyMixin as policy_cls

        return policy_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
