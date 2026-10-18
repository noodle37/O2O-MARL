#!/usr/bin/env python3
"""Fixed-state probe for the ranking-loss fork branches.

The three branches only share their state at the fork instant; afterwards their
trajectories differ, so a plain win-rate comparison cannot say *how* their action
preferences moved. This probe removes that confound:

  1. roll out the fork-time policy (the snapshot's live net) on fixed env seeds,
     eps=0, and freeze the resulting (obs, avail, state, executed joint) steps;
  2. replay those identical steps through every branch checkpoint through the very
     same code path the ranking loss uses (build_recorded_neighborhood + mixer);
  3. report, on those identical states, joint-greedy agreement with the fork-time
     policy and with Q-bar, Kendall tau between live and Q-bar Q_tot, and how much
     of the ordinal constraint is still active.

Usage:
  python scripts/probe_fork_branches.py --config configs/uot_5m_vs_6m_fork_branch.yaml \
      --fork-snapshot checkpoints/uot_fork_5m_vs_6m_seed1/fork_snapshot_main_t1002019.pt \
      --checkpoint continue=<path> --checkpoint off=<path> --checkpoint decay=<path> \
      --episodes 32 --out logs/probe_fork_seed1.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.factory import build_learner, load_vault_dataset, make_env  # noqa: E402
from training.rollout import collect_episode, stack_episodes  # noqa: E402
from uot.neighborhood import build_recorded_neighborhood  # noqa: E402
from utils.checkpoint import load_qmix_checkpoint, load_trainer_snapshot  # noqa: E402
from utils.paths import ensure_project_root  # noqa: E402


def _kendall(a: list[float], b: list[float]) -> float:
    """Kendall tau over two same-length score lists (ties dropped)."""
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = (a[i] > a[j]) - (a[i] < a[j])
            sb = (b[i] > b[j]) - (b[i] < b[j])
            if sa == 0 or sb == 0:
                continue
            if sa == sb:
                conc += 1
            else:
                disc += 1
    if conc + disc == 0:
        return float("nan")
    return (conc - disc) / (conc + disc)


def _copy_frozen_oracle(learner, reference) -> None:
    """Give `learner` the reference learner's Q-bar.

    Endpoint checkpoints do not carry the frozen oracle, and a freshly built learner
    has frozen_q_network=None because freeze_oracle() was never called, so the
    container is rebuilt from the live net before the state dict is loaded.
    """
    if reference is None:
        return
    import copy

    for attr, live_attr in (
        ("frozen_q_network", "q_network"),
        ("frozen_mixer", "mixer"),
    ):
        net = getattr(learner, attr, None)
        if net is None:
            net = copy.deepcopy(getattr(learner, live_attr))
            for param in net.parameters():
                param.requires_grad_(False)
            setattr(learner, attr, net)
        net.load_state_dict(getattr(reference, attr).state_dict())


def _build_learner(cfg, snapshot=None, checkpoint=None, frozen_from=None):
    dataset = load_vault_dataset(cfg)
    learner = build_learner(cfg, dataset)
    if snapshot:
        load_trainer_snapshot(snapshot, learner, map_location=learner.device)
    elif checkpoint:
        load_qmix_checkpoint(checkpoint, learner, map_location=learner.device)
    else:
        raise ValueError("need either snapshot or checkpoint")
    # Q-bar is the same frozen offline critic in every branch; endpoint checkpoints
    # do not carry it, so it is copied from the fork-time reference learner.
    _copy_frozen_oracle(learner, frozen_from)
    learner.q_network.eval()
    learner.mixer.eval()
    if learner.frozen_q_network is not None:
        learner.frozen_q_network.eval()
        learner.frozen_mixer.eval()
    return learner, dataset


@torch.no_grad()
def _candidate_scores(learner, batch, support, k, max_hamming):
    """cand/valid/q_live/q_bar on the batch, via the ranking loss's own code path."""
    mac_out = learner._forward_qs(batch, target=False)
    bar_out = learner._forward_net(learner.frozen_q_network, batch)
    sl = slice(None, -1)  # train_on_batch drops the bootstrap slot the same way
    avail, actions, states = batch["avail_actions"][sl], batch["actions"][sl], batch["state"][sl]
    cand, valid = build_recorded_neighborhood(
        mac_out[sl], bar_out[sl], avail, actions, support, k=k, max_hamming=max_hamming
    )
    q_live = learner._qtot_candidates(learner.mixer, mac_out[sl], cand, states)
    q_bar = learner._qtot_candidates(learner.frozen_mixer, bar_out[sl], cand, states)
    return cand, valid, q_live[:, :, :, 0], q_bar[:, :, :, 0]


@torch.no_grad()
def _probe(learner, batch, support, k, max_hamming, delta, margin, reference=None):
    """Per-state preference/ordering metrics on the frozen state set."""
    cand, valid, q_live, q_bar = _candidate_scores(learner, batch, support, k, max_hamming)
    ref_live = None
    if reference is not None:
        _, _, ref_live, _ = _candidate_scores(reference, batch, support, k, max_hamming)

    k_dim, b_dim, t_dim = valid.shape
    agree_fork = agree_bar = compared = 0
    tau_vals: list[float] = []
    pairs = hinge_active = sign_disagree = 0
    for b in range(b_dim):
        for t in range(t_dim):
            idx = [i for i in range(k_dim) if bool(valid[i, b, t])]
            if len(idx) < 2:
                continue
            live = [float(q_live[i, b, t]) for i in idx]
            bar = [float(q_bar[i, b, t]) for i in idx]
            compared += 1
            tau = _kendall(live, bar)
            if not math.isnan(tau):
                tau_vals.append(tau)
            if ref_live is not None:
                ref = [float(ref_live[i, b, t]) for i in idx]
                if int(np.argmax(live)) == int(np.argmax(ref)):
                    agree_fork += 1
            if int(np.argmax(live)) == int(np.argmax(bar)):
                agree_bar += 1
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    if abs(bar[x] - bar[y]) < margin:
                        continue
                    pairs += 1
                    sign = 1.0 if bar[x] > bar[y] else -1.0
                    if (delta - sign * (live[x] - live[y])) > 0:
                        hinge_active += 1
                    if (live[x] > live[y]) != (bar[x] > bar[y]):
                        sign_disagree += 1
    return {
        "n_states_compared": compared,
        "joint_greedy_agree_fork": (agree_fork / compared) if compared else float("nan"),
        "joint_greedy_agree_bar": (agree_bar / compared) if compared else float("nan"),
        "kendall_tau_vs_bar": (sum(tau_vals) / len(tau_vals)) if tau_vals else float("nan"),
        "n_pairs": pairs,
        "hinge_active_rate": (hinge_active / pairs) if pairs else float("nan"),
        "sign_disagree_rate": (sign_disagree / pairs) if pairs else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="Hydra-style overrides")
    ap.add_argument("--fork-snapshot", required=True)
    ap.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="name=path, repeatable (branch endpoints to probe)",
    )
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ensure_project_root()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)),
        resolve=True,
    )

    ref, dataset = _build_learner(cfg, snapshot=args.fork_snapshot)
    support = ref.support
    sys_cfg = cfg["system"]
    k = int(sys_cfg.get("uot_k", 8))
    max_hamming = int(sys_cfg.get("uot_max_hamming", 2))
    delta = float(sys_cfg.get("uot_delta", 0.1))
    margin = float(sys_cfg.get("uot_margin", 0.1))

    env = make_env(cfg, seed=int(cfg["seed"]))
    episodes = [collect_episode(env, ref, 0.0) for _ in range(args.episodes)]
    batch = stack_episodes(episodes, dataset.stats.max_seq_length, ref.device)
    print(
        f"[probe] frozen state set: {args.episodes} episodes, "
        f"{sum(int(e['length']) for e in episodes)} steps, "
        f"S_off={len(support)} k={k} hamming<={max_hamming}"
    )

    rows = []
    ref_metrics = _probe(ref, batch, support, k, max_hamming, delta, margin)
    rows.append({"name": "fork_reference", "checkpoint": args.fork_snapshot, **ref_metrics})

    for item in args.checkpoint:
        if "=" not in item:
            raise SystemExit(f"--checkpoint expects name=path, got {item!r}")
        name, path = item.split("=", 1)
        learner, _ = _build_learner(cfg, checkpoint=path, frozen_from=ref)
        metrics = _probe(
            learner, batch, support, k, max_hamming, delta, margin, reference=ref
        )
        rows.append({"name": name, "checkpoint": path, **metrics})
        print(f"[probe] {name}: {json.dumps(metrics)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "fork_snapshot": args.fork_snapshot,
                "episodes": args.episodes,
                "seed": int(cfg["seed"]),
                "k": k,
                "max_hamming": max_hamming,
                "rows": rows,
            },
            handle,
            indent=2,
        )
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
