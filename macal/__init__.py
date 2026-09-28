"""MaCal-QL: Cal-QL calibrated conservatism on joint QMIX Q_tot."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["MaCalQLLearner"]

if TYPE_CHECKING:
    from macal.learner import MaCalQLLearner


def __getattr__(name: str):
    if name == "MaCalQLLearner":
        from macal.learner import MaCalQLLearner

        return MaCalQLLearner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
