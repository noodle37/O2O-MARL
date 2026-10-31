#!/usr/bin/env python3
"""Train one joint-reuse continuation branch."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joint_reuse.runner import DEFAULT_OFFLINE_REFERENCE, load_cfg, parse_offline_agents, run_branch
from utils.paths import ensure_project_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branch", choices=("joint", "independent", "none", "adaptive"), required=True)
    parser.add_argument("--offline-agents", default="")
    parser.add_argument("--adaptive-model", default=None)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--offline-reference", default=DEFAULT_OFFLINE_REFERENCE)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--test-interval", type=int, default=50_000)
    parser.add_argument("--save-every", type=int, default=100_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    ensure_project_root()
    cfg = load_cfg(args.config, list(args.overrides))
    offline_agents = parse_offline_agents(args.offline_agents)
    if args.branch == "none" and offline_agents:
        parser.error("--branch none requires --offline-agents to be empty")
    if args.branch in ("joint", "independent") and len(offline_agents) != 2:
        parser.error("retention branches must retain exactly two agents")
    if args.branch == "adaptive" and not args.adaptive_model:
        parser.error("--branch adaptive requires --adaptive-model")

    run_branch(
        cfg,
        seed=args.seed,
        branch=args.branch,
        offline_agents=offline_agents,
        start_checkpoint=args.start_checkpoint,
        offline_reference=args.offline_reference,
        run_dir=args.run_dir,
        steps=args.steps,
        eval_episodes=args.eval_episodes,
        test_interval=args.test_interval,
        save_every=args.save_every,
        smoke=args.smoke,
        adaptive_model=args.adaptive_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
