# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Sliding-window UCB over the discrete offline-mixing-ratio arms."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class SlidingWindowUCB:
    """UCB1 with a window of the most recent ``window`` rewards.

    Every arm is tried once before the UCB rule takes over. UCB scores use the
    window's empirical mean and count; an arm absent from the window is treated
    as having infinite score so it is revisited, which is what makes the window
    forget a regime change.

    ``select`` records the chosen arm, so the caller may keep using the returned
    ratio for a whole period and call ``select`` again only at the boundary.
    """

    def __init__(
        self,
        arms: Sequence[float],
        c: float = 2.0,
        window: int = 200,
        seed: int = 0,
    ) -> None:
        arms = [float(arm) for arm in arms]
        if not arms:
            raise ValueError("bandit needs at least one arm")
        if len(set(arms)) != len(arms):
            raise ValueError(f"bandit arms must be distinct, got {arms}")
        if window < 1:
            raise ValueError(f"bandit window must be positive, got {window}")
        self.arms = arms
        self.c = float(c)
        self.window = int(window)
        self._rng = np.random.default_rng(int(seed))
        self._entries: deque = deque(maxlen=self.window)
        self._total_counts = [0] * len(arms)
        self._rounds = 0

    # ------------------------------------------------------------------ lookup

    def _index(self, arm: float) -> int:
        for i, value in enumerate(self.arms):
            if abs(value - float(arm)) < 1e-12:
                return i
        raise ValueError(f"unknown arm {arm!r}; expected one of {self.arms}")

    def _window_stats(self) -> tuple[List[int], List[float]]:
        counts = [0] * len(self.arms)
        sums = [0.0] * len(self.arms)
        for idx, reward in self._entries:
            counts[idx] += 1
            sums[idx] += reward
        return counts, sums

    def _ucb(self, idx: int, counts: List[int], sums: List[float]) -> float:
        n = counts[idx]
        if n == 0:
            return float("inf")
        mean = sums[idx] / n
        horizon = min(self._rounds, self.window)
        if horizon <= 1:
            return mean
        return mean + math.sqrt(self.c * math.log(horizon) / n)

    # ------------------------------------------------------------------ actions

    def select(self) -> float:
        """Pick an arm. Untried arms come first; afterwards SW-UCB argmax."""
        untried = [i for i, n in enumerate(self._total_counts) if n == 0]
        if untried:
            idx = untried[0]
        else:
            counts, sums = self._window_stats()
            scores = np.asarray(
                [self._ucb(i, counts, sums) for i in range(len(self.arms))],
                dtype=np.float64,
            )
            best = np.flatnonzero(scores == scores.max())
            idx = int(best[0]) if best.size == 1 else int(self._rng.choice(best))
        self._total_counts[idx] += 1
        return self.arms[idx]

    def update(self, arm: float, reward: float) -> None:
        idx = self._index(arm)
        self._entries.append((idx, float(reward)))
        self._rounds += 1

    # ------------------------------------------------------------- diagnostics

    def window_counts(self) -> Dict[float, int]:
        counts, _ = self._window_stats()
        return {arm: counts[i] for i, arm in enumerate(self.arms)}

    def ucb_values(self) -> Dict[float, float]:
        counts, sums = self._window_stats()
        return {
            arm: self._ucb(i, counts, sums) for i, arm in enumerate(self.arms)
        }

    def total_counts(self) -> List[int]:
        return list(self._total_counts)

    @property
    def rounds(self) -> int:
        return self._rounds

    # ---------------------------------------------------------------- snapshot

    def state_dict(self) -> Dict[str, Any]:
        return {
            "arms": list(self.arms),
            "c": self.c,
            "window": self.window,
            "entries": [[int(idx), float(reward)] for idx, reward in self._entries],
            "total_counts": list(self._total_counts),
            "rounds": self._rounds,
            "rng": self._rng.bit_generator.state,
        }

    def load_state_dict(self, payload: Dict[str, Any]) -> None:
        self.arms = [float(arm) for arm in payload["arms"]]
        self.c = float(payload["c"])
        self.window = int(payload["window"])
        self._entries = deque(
            [(int(idx), float(reward)) for idx, reward in payload["entries"]],
            maxlen=self.window,
        )
        self._total_counts = [int(n) for n in payload["total_counts"]]
        self._rounds = int(payload["rounds"])
        self._rng = np.random.default_rng()
        if payload.get("rng") is not None:
            self._rng.bit_generator.state = payload["rng"]
