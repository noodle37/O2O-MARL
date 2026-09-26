# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Training loggers (terminal and Weights & Biases)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import wandb


class BaseLogger:
    def write(self, logs: Dict[str, Any], force: bool = False) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return


class TerminalLogger(BaseLogger):
    def __init__(
        self,
        log_every: int = 2,  # seconds
    ):
        self._log_every = log_every
        self._ctr = 0
        self._last_log = time.time()

    def write(self, logs: Dict[str, Any], force: bool = False) -> None:
        if time.time() - self._last_log > self._log_every or force:
            for key, log in logs.items():
                print(f"{key}: {float(log)} |", end=" ")
            print()

            if not force:
                self._last_log = time.time()

        self._ctr += 1


class WandbLogger(BaseLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        config: Dict = {},  # noqa: B006
        project: str = "default_project",
        notes: str = "",
        tags: List = ["default"],  # noqa: B006
        entity: Optional[str] = None,
        log_every: int = 2,  # seconds
    ):
        wandb.init(name=name, project=project, notes=notes, tags=tags, entity=entity, config=config)

        self._log_every = log_every
        self._ctr = 0
        self._last_log = time.time()

    def write(self, logs: Dict[str, Any], force: bool = False) -> None:
        if time.time() - self._last_log > self._log_every or force:
            wandb.log(logs)

            for key, log in logs.items():
                print(f"{key}: {float(log)} |", end=" ")
            print()

            if not force:
                self._last_log = time.time()

        self._ctr += 1

    def close(self) -> None:
        wandb.finish()
