# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Training loggers (terminal and Weights & Biases)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional



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


class JsonlLogger(BaseLogger):
    """Append each forced eval record to a jsonl file and echo it."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, logs: Dict[str, Any], force: bool = False) -> None:
        if not force:
            return
        row = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={row[k]}" for k in row), flush=True)


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
        import wandb

        self._wandb = wandb
        self._wandb.init(
            name=name, project=project, notes=notes, tags=tags, entity=entity, config=config
        )

        self._log_every = log_every
        self._ctr = 0
        self._last_log = time.time()

    def write(self, logs: Dict[str, Any], force: bool = False) -> None:
        if time.time() - self._last_log > self._log_every or force:
            self._wandb.log(logs)

            for key, log in logs.items():
                print(f"{key}: {float(log)} |", end=" ")
            print()

            if not force:
                self._last_log = time.time()

        self._ctr += 1

    def close(self) -> None:
        self._wandb.finish()
