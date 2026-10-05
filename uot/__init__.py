"""Ungated Ordinal Transfer: live QMIX TD-λ plus a hinge on recorded joint order."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["UOTLearner"]

if TYPE_CHECKING:
    from uot.learner import UOTLearner


def __getattr__(name: str):
    if name == "UOTLearner":
        from uot.learner import UOTLearner

        return UOTLearner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
