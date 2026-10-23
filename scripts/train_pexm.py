#!/usr/bin/env python3
"""Train the PEX-M online pilot from an offline QMIX checkpoint."""

from __future__ import annotations

import argparse
import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.pexm import run_pexm_online  # noqa: E402
from utils.paths import ensure_project_root, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )
    if args.device:
        cfg["system"]["device"] = args.device
    if args.log_path:
        cfg["log_path"] = args.log_path
    if not cfg.get("log_path"):
        cfg["log_path"] = os.path.join(str(cfg["checkpoint_dir"]), "online.jsonl")
    log_path = resolve_path(cfg["log_path"])
    if log_path and os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
        raise SystemExit(
            f"refusing to append to a non-empty log {log_path}; "
            "remove it or pass a different checkpoint_dir/log_path"
        )
    run_pexm_online(cfg)


if __name__ == "__main__":
    main()
