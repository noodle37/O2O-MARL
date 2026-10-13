"""MAC-Flow learner: per-agent flow matching on one-hot actions.

Mirrors the original repo's `agents/discrete_macflow.py`:
- per-agent factorized flow over one-hot actions; obs prepended with agent-id one-hot
- one-step policy distilled (CE) against the Euler flow's argmax (hard labels)
- critic TD mixed by summing per-agent twin-mean Q (VDN), loss = 0.5 * MSE
- Q-guidance term uses the argmax one-hot (gradient reaches the critic only)
- single Adam optimizer (eps=1e-8), Polyak target update after each step
- deployment: z=0, legal-masked greedy argmax
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from macflow.flow import euler_integrate, flow_bc_loss
from macflow.networks import FlowField, OneStepPolicy, TwinQ, add_agent_id, mix_mean


class MACFlowLearner:
    checkpoint_stem = "mac_flow"
    moderate_lambda = False
    load_offline_optimizer = True

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        *,
        n_actions: int = 0,
        act_dim: int = 0,
        action_mode: str = "discrete",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        alpha_distill: float = 3.0,
        flow_steps: int = 10,
        polyak_tau: float = 0.005,
        grad_norm_clip: float = 0.0,
        normalize_q: bool = False,
        device: str = "cuda",
        batch_size_run: int = 1,
        sequence_length: int = 20,
        hidden_dims: Sequence[int] = (256, 256, 256, 256),
        actor_layer_norm: bool = False,
        t_embed_frequencies: int = 8,
        adam_eps: float = 1e-8,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.action_mode = str(action_mode).lower()
        self.continuous = self.action_mode == "continuous"
        self.per_agent_dim = int(act_dim if self.continuous else n_actions)
        self.n_actions = int(n_actions) if not self.continuous else int(act_dim)
        self.pair_dim = self.n_agents + self.obs_dim
        self.gamma = float(gamma)
        self.alpha_distill = float(alpha_distill)
        self.flow_steps = int(flow_steps)
        self.polyak_tau = float(polyak_tau)
        self.grad_norm_clip = float(grad_norm_clip)
        self.normalize_q = bool(normalize_q)
        self.batch_size_run = int(batch_size_run)
        self.sequence_length = int(sequence_length)
        self.t_env = 0
        self.training_step_ctr = 0
        self.last_diag: Dict[str, float] = {}

        hidden = [int(h) for h in hidden_dims]
        self.flow = FlowField(
            self.pair_dim,
            self.per_agent_dim,
            hidden,
            layer_norm=actor_layer_norm,
            t_freqs=t_embed_frequencies,
        ).to(self.device)
        self.policy = OneStepPolicy(
            self.pair_dim,
            self.per_agent_dim,
            self.per_agent_dim,
            hidden,
            layer_norm=actor_layer_norm,
        ).to(self.device)
        self.critic = TwinQ(self.pair_dim, self.per_agent_dim, hidden, layer_norm=True).to(
            self.device
        )
        self.target_critic = type(self.critic)(
            self.pair_dim, self.per_agent_dim, hidden, layer_norm=True
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            chain(self.flow.parameters(), self.critic.parameters(), self.policy.parameters()),
            lr=learning_rate,
            eps=adam_eps,
        )

        print(
            f"MAC-Flow mode={self.action_mode} agents={n_agents} obs={obs_dim} "
            f"pair={self.pair_dim} per_dim={self.per_agent_dim} seq={sequence_length} "
            f"hidden={hidden} flow_steps={flow_steps} alpha={alpha_distill} "
            f"gamma={gamma} normalize_q={normalize_q}"
        )

    @staticmethod
    def linear_epsilon(t_env: int, eps_start: float, eps_min: float, decay_steps: int) -> float:
        if decay_steps <= 0 or t_env >= decay_steps:
            return eps_min
        return eps_start + (eps_min - eps_start) * (t_env / decay_steps)

    def new_episode_state(self) -> Dict[str, Any]:
        return {}

    def reset_eval_hidden(self) -> None:
        return None

    def _window_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Sample a [B, W] window of consecutive steps per episode (repo sequence sampling)."""
        obs = batch["obs"]
        b, s = obs.shape[0], obs.shape[1]
        if s < 2:
            raise ValueError("MAC-Flow batch has no transitions")
        w = min(self.sequence_length, s)
        filled = batch["filled"][..., 0]
        lengths = filled.sum(dim=1)
        max_start = (lengths - w).clamp(min=0).long()
        u = torch.rand(b, device=obs.device) * (max_start + 1).float()
        starts = u.floor().long().clamp_(max=max_start)
        idx = starts[:, None] + torch.arange(w, device=obs.device)[None, :]
        rows = torch.arange(b, device=obs.device)[:, None]

        def take(x: torch.Tensor) -> torch.Tensor:
            return x[rows, idx]

        window = {
            "obs": take(obs),
            "actions": take(batch["actions"]),
            "filled": take(filled),
            "terminated": take(batch["terminated"][..., 0]),
            "reward": take(batch["reward"][..., 0]),
        }
        avail = batch.get("avail_actions")
        if avail is not None:
            window["avail"] = take(avail)
        return window

    def train_on_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        t_env: Optional[int] = None,
        episode_delta: int = 1,
        cql_weight: Optional[float] = None,
        use_ovm: Optional[bool] = None,
        eps: Optional[float] = None,
    ) -> Dict[str, float]:
        del cql_weight, use_ovm, eps, episode_delta
        if t_env is not None:
            self.t_env = int(t_env)

        wb = self._window_batch(batch)
        b, w, n = wb["obs"].shape[0], wb["obs"].shape[1], wb["obs"].shape[2]
        pair = add_agent_id(wb["obs"])
        device = pair.device

        # Actor-side losses use window steps 0..W-2 (repo `observations[:-1]`).
        pair_f = pair[:, :-1].reshape(-1, self.pair_dim)
        mask_t = wb["filled"][:, :-1] > 0.5
        mask = mask_t[:, :, None].expand(-1, -1, n).reshape(-1)
        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1.0)

        if self.continuous:
            x1 = wb["actions"][:, :-1].float().reshape(-1, self.per_agent_dim)
        else:
            idx_a = wb["actions"][:, :-1, :, 0].long().clamp(0, self.per_agent_dim - 1)
            x1 = F.one_hot(idx_a, self.per_agent_dim).float().reshape(-1, self.per_agent_dim)

        # --- Flow BC (Eq. 6) ---
        loss_flow = flow_bc_loss(self.flow, pair_f, x1, mask=mask_f)

        # --- Critic TD: next action from the one-step policy with fresh z (repo) ---
        pair_next = pair[:, 1:].reshape(-1, self.pair_dim)
        z_next = torch.randn(x1.shape, device=device)
        with torch.no_grad():
            pol_next = self.policy(pair_next, z_next)
            if self.continuous:
                a_next = pol_next.clamp(-1.0, 1.0)
            else:
                a_next = F.one_hot(pol_next.argmax(dim=-1), self.per_agent_dim).float()
            q_next = self.target_critic(pair_next, a_next).reshape(b, w - 1, n)
            r = wb["reward"][:, :-1]
            done = (wb["terminated"][:, 1:] > 0.5).float()
            target = r.unsqueeze(-1) + self.gamma * (1.0 - done).unsqueeze(-1) * q_next
            mixed_target = target.sum(dim=-1)

        q_cur = self.critic(pair_f, x1).reshape(b, w - 1, n)
        mixed_q = q_cur.sum(dim=-1)
        td_err = (mixed_q - mixed_target) ** 2
        mask_t_f = mask_t.float()
        td_loss = 0.5 * (td_err * mask_t_f).sum() / mask_t_f.sum().clamp(min=1.0)

        # --- One-step policy distillation (Eq. 9) with fresh z ---
        z_dist = torch.randn(x1.shape, device=device)
        logits = self.policy(pair_f, z_dist)
        with torch.no_grad():
            endpoint = euler_integrate(self.flow, pair_f, z_dist, self.flow_steps)
            teacher = endpoint.clamp(-1.0, 1.0) if self.continuous else endpoint.argmax(dim=-1)
        if self.continuous:
            distill_rows = ((logits - teacher) ** 2).mean(dim=-1)
        else:
            distill_rows = F.cross_entropy(logits, teacher, reduction="none")
        distill_loss = (distill_rows * mask_f).sum() / denom

        # --- Q-guidance: repo's argmax one-hot term; gradient reaches the critic ---
        if self.continuous:
            q_guid = self.critic(pair_f, logits.clamp(-1.0, 1.0)).reshape(b, w - 1, n)
        else:
            with torch.no_grad():
                onehot = F.one_hot(logits.argmax(dim=-1), self.per_agent_dim).float()
            q_guid = self.critic(pair_f, onehot).reshape(b, w - 1, n)
        q_guid = q_guid.mean(dim=-1)
        q_guid = (q_guid * mask_t_f).sum() / mask_t_f.sum().clamp(min=1.0)
        q_loss = -q_guid
        if self.normalize_q:
            q_loss = q_loss / q_guid.detach().abs().clamp(min=1e-6)

        actor_loss = loss_flow + self.alpha_distill * distill_loss + q_loss
        loss = td_loss + actor_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                chain(self.flow.parameters(), self.critic.parameters(), self.policy.parameters()),
                self.grad_norm_clip,
            )
        self.optimizer.step()

        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.polyak_tau).add_(p.data, alpha=self.polyak_tau)

        self.training_step_ctr += 1
        logs = {
            "loss": float(loss.detach().item()),
            "loss_flow": float(loss_flow.detach().item()),
            "td_loss": float(td_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "distill_loss": float(distill_loss.detach().item()),
            "q_loss": float(q_loss.detach().item()),
            "q_tot_mean": float(
                ((mixed_q.detach() * mask_t.float()).sum() / mask_t.float().sum().clamp(min=1.0)).item()
            ),
        }
        self.last_diag = logs
        return logs

    def export_state(self) -> Dict[str, Any]:
        return {
            "flow": self.flow.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "policy": self.policy.state_dict(),
            "meta": {
                "n_agents": self.n_agents,
                "obs_dim": self.obs_dim,
                "pair_dim": self.pair_dim,
                "per_agent_dim": self.per_agent_dim,
                "action_mode": self.action_mode,
                "sequence_length": self.sequence_length,
            },
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        self.flow.load_state_dict(payload["flow"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload.get("target_critic", payload["critic"]))
        self.policy.load_state_dict(payload["policy"])
        if "optimizer" in payload and self.load_offline_optimizer:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "training_step_ctr" in payload:
            self.training_step_ctr = int(payload["training_step_ctr"])

    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Optional[Dict[str, np.ndarray]],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        env_state: Any = None,
    ) -> Dict[str, Union[int, np.ndarray]]:
        del state, env_state
        agents = sorted(observations.keys())
        obs = torch.tensor(
            np.stack([observations[a].astype(np.float32) for a in agents]),
            device=self.device,
        ).unsqueeze(0)
        pair = add_agent_id(obs)
        if eps > 0:
            z = torch.randn(1, self.n_agents, self.per_agent_dim, device=self.device)
        else:
            z = torch.zeros(1, self.n_agents, self.per_agent_dim, device=self.device)
        with torch.no_grad():
            out = self.policy(pair, z)[0]
        if self.continuous:
            chosen = out.clamp(-1.0, 1.0)
            act = chosen.cpu().numpy()
            actions = {a: act[i].astype(np.float32) for i, a in enumerate(agents)}
            a_c = chosen.reshape(1, self.n_agents, self.per_agent_dim)
        else:
            logits = out
            if legals is not None:
                avail = torch.tensor(
                    np.stack([legals[a] for a in agents]),
                    device=self.device,
                    dtype=torch.bool,
                )
                logits = logits.masked_fill(~avail, -1e9)
            idx = logits.argmax(dim=-1).reshape(-1)
            if eps > 0:
                for i, a in enumerate(agents):
                    if np.random.rand() < eps / self.n_agents:
                        legal = np.flatnonzero(legals[a]) if legals else np.arange(self.per_agent_dim)
                        idx[i] = int(np.random.choice(legal))
            actions = {a: int(idx[i]) for i, a in enumerate(agents)}
            a_c = F.one_hot(idx, self.per_agent_dim).float().reshape(1, self.n_agents, self.per_agent_dim)
        if trace is not None:
            with torch.no_grad():
                trace["q_tot"] = float(mix_mean(self.critic(pair, a_c)).item())
        return actions
