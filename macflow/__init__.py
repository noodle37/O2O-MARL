"""MAC-Flow: multi-agent coordination via flow matching (arXiv:2511.05005)."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["MACFlowLearner"]

if TYPE_CHECKING:
    from macflow.learner import MACFlowLearner


def __getattr__(name: str):
    if name == "MACFlowLearner":
        from macflow.learner import MACFlowLearner

        return MACFlowLearner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
