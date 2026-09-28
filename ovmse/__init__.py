"""OVMSE / QMIX + CQL paper-faithful learner."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["PaperQMIXLearner"]

if TYPE_CHECKING:
    from ovmse.learner import PaperQMIXLearner


def __getattr__(name: str):
    if name == "PaperQMIXLearner":
        from ovmse.learner import PaperQMIXLearner

        return PaperQMIXLearner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
