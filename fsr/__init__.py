"""FSR: Flow-Supported Reliance for offline-to-online MARL."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["FSRLearner"]

if TYPE_CHECKING:
    from fsr.learner import FSRLearner


def __getattr__(name: str):
    if name == "FSRLearner":
        from fsr.learner import FSRLearner

        return FSRLearner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
