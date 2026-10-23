# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Autoregressive joint behaviour model of the recorded data, and the teammate
deviation measure D built on top of it.

The model factorises the joint data behaviour as

    pi_hat(a | s) = prod_k pi_hat(a_k | s, a_{<k})

with the agent order drawn per batch element during training, so afterwards any
subset's conditional probability can be queried under any order.  The per-agent
input mirrors the QMIX learner's encoding (obs, previous-action one-hot, agent
id) plus, for the agent sitting at position k, the one-hot current-step actions
of the agents at positions < k (zero padded, so position 0 is the marginal).
Illegal actions are masked before the softmax.

Deviations used by the experiment design:

    D_i(s) = -1/(n-1) * [ log pi_hat(a | s) - log pi_hat(a_i | s) ]

where the joint term uses the order (i, then the remaining agents in ascending
index order), so the difference is exactly log pi_hat(a_{-i} | s, a_i); and

    D(s) = -1/n * log pi_hat(a | s)

with the ascending agent order.  Both are per-state quantities computed on the
*executed* joint action, so they also score online rollouts, not only data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from qmix.core import QMIXCoreMixin
from qmix.networks import MarlcRNN
from qmix.tensor_utils import (
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
)


class _Adapter(QMIXCoreMixin):
    """Minimal host so the tested QMIX unroll / reset helpers can be reused."""

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        add_agent_id_to_obs: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.add_agent_id_to_obs = bool(add_agent_id_to_obs)
        self.device = device


@dataclass
class BehaviourModelConfig:
    n_agents: int
    n_actions: int
    obs_dim: int
    hidden_dim: int = 64
    add_agent_id_to_obs: bool = True

    @property
    def input_dim(self) -> int:
        dim = self.obs_dim + self.n_actions + self.n_agents
        # One conditioning slot per position (slot 0 is always zero).
        return dim + self.n_agents * self.n_actions


class JointBehaviourModel:
    """Shared-per-agent GRU behaviour model with autoregressive conditioning."""

    def __init__(self, cfg: BehaviourModelConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_agents = int(cfg.n_agents)
        self.n_actions = int(cfg.n_actions)
        self.net = MarlcRNN(cfg.input_dim, cfg.hidden_dim, cfg.n_actions).to(self.device)
        self.adapter = _Adapter(
            self.n_agents, self.n_actions, cfg.add_agent_id_to_obs, device=self.device
        )

    # ---- forward ---------------------------------------------------------
    def _ordered(self, batch: Dict[str, torch.Tensor], order: torch.Tensor):
        """Reorder obs/actions/avail so array position k holds agent order[b, k]."""
        obs = batch["obs"]
        actions = batch["actions"]
        avail = batch["avail_actions"]
        b, t, n, obs_dim = obs.shape
        # All three tensors are [b, t, n, ...] and are reordered along the agent
        # axis, so the gather index is expanded along the trailing dimension.
        idx = order.view(b, 1, n, 1).to(obs.device)
        obs_ord = obs.gather(2, idx.expand(b, t, n, obs_dim))
        act_ord = actions.gather(2, idx.expand(b, t, n, 1))  # [b, t, n, 1]
        avail_ord = avail.gather(2, idx.expand(b, t, n, avail.shape[-1]))
        return obs_ord, act_ord, avail_ord

    def forward_logp(
        self, batch: Dict[str, torch.Tensor], order: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Log-probability of the recorded action at every position.

        Returns (logp [b, t, n], logits [b, t, n, n_actions]).
        """
        obs_ord, act_ord, avail_ord = self._ordered(batch, order)
        b, t, n, _ = obs_ord.shape
        last = obs_ord.new_zeros(b, t, n, self.n_actions)
        if t > 1:
            prev = act_ord[:, :-1, :, 0].long().clamp(0, self.n_actions - 1)
            last[:, 1:] = F.one_hot(prev, self.n_actions).float()
        agent_id = (
            F.one_hot(order.to(obs_ord.device), n).float().view(b, 1, n, n).expand(b, t, n, n)
        )
        oh_ord = F.one_hot(act_ord[..., 0].long(), self.n_actions).float()  # [b, t, n, a]
        # Conditioning slots for position k: the one-hot current-step actions of the
        # agents at positions j < k, zero padded (slot 0 is therefore all zeros).
        lower = torch.tril(torch.ones(n, n, device=obs_ord.device), diagonal=-1)
        cond = torch.einsum("btja,kj->btkja", oh_ord, lower)  # [b, t, k, j, a]
        cond = cond.reshape(b, t, n, n * self.n_actions)
        aug = torch.cat([obs_ord, last, agent_id, cond.reshape(b, t, n, n * self.n_actions)], dim=-1)
        resets = self.adapter._resets_from_batch(batch)
        logits = self.adapter._unroll(self.net, aug, resets)
        logits = logits.masked_fill(~avail_ord, -1e9)
        logp_all = F.log_softmax(logits.float(), dim=-1)
        target = act_ord[..., 0].long()
        logp = logp_all.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return logp, logits

    def _filled_mask(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        filled = batch["filled"]
        return (filled[..., 0] if filled.shape[-1] == 1 else filled.squeeze(-1)) > 0.5

    @staticmethod
    def ascending_order(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        b, _, n, _ = batch["obs"].shape
        return torch.arange(n, device=batch["obs"].device).view(1, n).expand(b, n).contiguous()

    @staticmethod
    def order_with_first(n_agents: int, first: int, batch_size: int, device) -> torch.Tensor:
        rest = [a for a in range(n_agents) if a != first]
        order = torch.tensor([first] + rest, device=device, dtype=torch.long)
        return order.view(1, -1).expand(batch_size, n_agents).contiguous()

    # ---- deviations ------------------------------------------------------
    def team_logp(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """log pi_hat(a | s) under the ascending order, [b, t]."""
        logp, _ = self.forward_logp(batch, self.ascending_order(batch))
        return logp.sum(dim=-1)

    def agent_marginal_and_joint(self, batch: Dict[str, torch.Tensor], agent: int):
        """(log pi_hat(a_agent | s), log pi_hat(a | s)) with `agent` first in the order."""
        b = batch["obs"].shape[0]
        order = self.order_with_first(self.n_agents, agent, b, batch["obs"].device)
        logp, _ = self.forward_logp(batch, order)
        return logp[:, :, 0], logp.sum(dim=-1)

    def deviations(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Per-state D_i for every agent and the team D, NaN outside filled steps."""
        mask = self._filled_mask(batch)
        team = self.team_logp(batch)
        out_team = torch.full_like(team, float("nan"))
        out_team[mask] = -team[mask] / self.n_agents
        per_agent = torch.full(
            (batch["obs"].shape[0], batch["obs"].shape[1], self.n_agents),
            float("nan"),
            device=team.device,
        )
        for agent in range(self.n_agents):
            marginal, joint = self.agent_marginal_and_joint(batch, agent)
            cond = joint - marginal  # log pi_hat(a_{-i} | s, a_i)
            val = -cond / max(self.n_agents - 1, 1)
            per_agent[:, :, agent] = torch.where(mask, val, torch.full_like(val, float("nan")))
        return {"per_agent": per_agent, "team": out_team, "mask": mask}

    def nll(self, batch: Dict[str, torch.Tensor]) -> float:
        """Mean per-agent NLL of the recorded action under the ascending order."""
        with torch.no_grad():
            logp, _ = self.forward_logp(batch, self.ascending_order(batch))
            mask = self._filled_mask(batch)
            denom = (mask.sum() * self.n_agents).clamp(min=1.0)
            return float((-logp[mask].sum() / denom).item())

    def uniform_nll(self, batch: Dict[str, torch.Tensor]) -> float:
        """Reference NLL: uniform over the legal actions of each agent."""
        avail = batch["avail_actions"].float()
        n_legal = avail.sum(dim=-1).clamp(min=1.0)
        per_step = n_legal.log().mean(dim=-1)  # [b, t]
        mask = self._filled_mask(batch)
        return float(per_step[mask].mean().item())

    def confidence_and_correct(self, batch: Dict[str, torch.Tensor]):
        """(max legal-action probability, argmax-matches-data flag) over filled steps."""
        with torch.no_grad():
            order = self.ascending_order(batch)
            _, logits = self.forward_logp(batch, order)
            probs = F.softmax(logits.float(), dim=-1)
            conf, argmax = probs.max(dim=-1)
            _, act_ord, _ = self._ordered(batch, order)
            target = act_ord[..., 0].long()
            correct = (argmax == target).float()
            mask = self._filled_mask(batch)
            return conf[mask].cpu(), correct[mask].cpu()


# ---- data helpers --------------------------------------------------------


def gather_episodes(dataset, indices: Sequence[int], device=None) -> Dict[str, torch.Tensor]:
    """Episode-batch gather by index (EpisodeDataset.sample only samples uniformly)."""
    idx = torch.as_tensor(np.asarray(indices), device=dataset.device, dtype=torch.long)
    keys = ("obs", "actions", "state", "avail_actions", "reward", "terminated", "filled")
    batch = {key: getattr(dataset, key).index_select(0, idx) for key in keys}
    if device is not None:
        target = torch.device(device)
        if batch["obs"].device != target:
            batch = {key: value.to(target) for key, value in batch.items()}
    return batch


def split_episodes(n_episodes: int, val_frac: float = 0.1, seed: int = 0) -> Tuple[List[int], List[int]]:
    """Deterministic episode-level split; returned lists are episode indices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_val = max(1, int(round(n_episodes * val_frac)))
    val = sorted(int(i) for i in perm[:n_val])
    train = sorted(int(i) for i in perm[n_val:])
    return train, val


def random_order(batch_size: int, n_agents: int, generator: torch.Generator) -> torch.Tensor:
    """One random agent permutation per batch element."""
    return torch.argsort(torch.rand(batch_size, n_agents, generator=generator), dim=1)


# ---- training ------------------------------------------------------------


@dataclass
class FitReport:
    train_nll: List[float] = field(default_factory=list)
    val_nll: List[float] = field(default_factory=list)
    best_val_nll: float = float("inf")
    best_update: int = -1
    updates: int = 0
    stopped_early: bool = False


def fit_behaviour_model(
    dataset,
    *,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    n_agents: int,
    n_actions: int,
    obs_dim: int,
    hidden_dim: int = 64,
    max_updates: int = 20000,
    batch_size: int = 128,
    lr: float = 1e-3,
    eval_every: int = 1000,
    patience: int = 5,
    device: str = "cuda",
    seed: int = 0,
    log_every: int = 2000,
) -> Tuple[JointBehaviourModel, FitReport]:
    """Fit the joint behaviour model on the training split, early-stopping on NLL."""
    cfg = BehaviourModelConfig(
        n_agents=n_agents, n_actions=n_actions, obs_dim=obs_dim, hidden_dim=hidden_dim
    )
    model = JointBehaviourModel(cfg, device=device)
    model.net.train()
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    report = FitReport()
    best_state = {k: v.detach().clone() for k, v in model.net.state_dict().items()}
    train_idx = list(train_indices)
    val_batch = gather_episodes(dataset, val_indices, device=model.device)
    n_train = len(train_idx)

    for update in range(1, max_updates + 1):
        pick = rng.choice(n_train, size=min(batch_size, n_train), replace=False)
        batch = gather_episodes(dataset, [train_idx[int(i)] for i in pick], device=model.device)
        order = random_order(len(pick), n_agents, gen).to(model.device)
        logp, _ = model.forward_logp(batch, order)
        mask = model._filled_mask(batch)
        loss = -(logp[mask].sum() / (mask.sum() * n_agents).clamp(min=1.0))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 10.0)
        opt.step()
        report.updates = update
        if update % eval_every == 0 or update == max_updates:
            model.net.eval()
            val_nll = model.nll(val_batch)
            model.net.train()
            report.val_nll.append(val_nll)
            report.train_nll.append(float(loss.item()))
            if val_nll < report.best_val_nll - 1e-5:
                report.best_val_nll = val_nll
                report.best_update = update
                best_state = {k: v.detach().clone() for k, v in model.net.state_dict().items()}
            elif update - report.best_update >= patience * eval_every:
                report.stopped_early = True
                break
        if update % log_every == 0:
            print(
                f"[bc] update={update} loss={float(loss.item()):.4f} "
                f"val_nll={report.val_nll[-1] if report.val_nll else float('nan'):.4f}",
                flush=True,
            )

    model.net.load_state_dict(best_state)
    model.net.eval()
    return model, report


class NoveltyEnsemble:
    """K behaviour models; disagreement of the team log-probability = state novelty."""

    def __init__(self, models: Sequence[JointBehaviourModel]):
        self.models = list(models)

    def team_logp_stack(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([m.team_logp(batch) for m in self.models], dim=0)

    def novelty(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-state std of the team log-probability across members, [b, t]."""
        stacked = self.team_logp_stack(batch)
        return stacked.std(dim=0)


# ---- persistence ---------------------------------------------------------


def save_behaviour_model(path: str, model: JointBehaviourModel, meta: Optional[dict] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "state_dict": model.net.state_dict(),
        "config": {
            "n_agents": model.cfg.n_agents,
            "n_actions": model.cfg.n_actions,
            "obs_dim": model.cfg.obs_dim,
            "hidden_dim": model.cfg.hidden_dim,
            "add_agent_id_to_obs": model.cfg.add_agent_id_to_obs,
        },
        "meta": meta or {},
    }
    torch.save(payload, path)
    return path


def load_behaviour_model(path: str, device: str = "cuda") -> JointBehaviourModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = BehaviourModelConfig(**payload["config"])
    model = JointBehaviourModel(cfg, device=device)
    model.net.load_state_dict(payload["state_dict"])
    model.net.eval()
    return model


def save_json(path: str, obj) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=str)
    return path
