"""Offline and online training loops."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["run_offline", "run_online"]

if TYPE_CHECKING:
    from training.offline import run_offline
    from training.online import run_online


def __getattr__(name: str):
    if name == "run_offline":
        from training.offline import run_offline

        return run_offline
    if name == "run_online":
        from training.online import run_online

        return run_online
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
