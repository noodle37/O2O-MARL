"""MAC-Flow learner: joint flow BC + VDN twin-Q + one-step distillation."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from macflow.flow import (
    euler_integrate,
    flow_bc_loss,
    obs_joint_from_batch,
    split_joint_to_agents,
)
from macflow.networks import JointFlowField, OneStepPolicy, TwinQCritic, vdn_tot


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
        gamma: float = 0.995,
        alpha_distill: float = 3.0,
        flow_steps: int = 10,
        polyak_tau: float = 0.005,
        grad_norm_clip: float = 10.0,
        normalize_q: bool = True,
        device: str = "cuda",
        batch_size_run: int = 1,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_mode = str(action_mode).lower()
        self.continuous = self.action_mode == "continuous"
        self.per_agent_dim = int(act_dim if self.continuous else n_actions)
        self.n_actions = int(n_actions) if not self.continuous else int(act_dim)
        self.joint_dim = n_agents * self.per_agent_dim
        self.gamma = gamma
        self.alpha_distill = alpha_distill
        self.flow_steps = flow_steps
        self.polyak_tau = polyak_tau
        self.grad_norm_clip = grad_norm_clip
        self.normalize_q = normalize_q
        self.batch_size_run = batch_size_run
        self.t_env = 0
        self.training_step_ctr = 0
        self.last_diag: Dict[str, float] = {}
        self._eval_noise: Optional[Dict[str, np.ndarray]] = None

        obs_joint_dim = n_agents * obs_dim
        self.flow = JointFlowField(obs_joint_dim, self.joint_dim).to(self.device)
        self.critic = TwinQCritic(obs_dim, self.per_agent_dim, n_agents).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.policy = OneStepPolicy(
            obs_dim,
            self.per_agent_dim,
            self.per_agent_dim,
            n_agents,
            continuous=self.continuous,
        ).to(self.device)

        self.flow_opt = torch.optim.Adam(self.flow.parameters(), lr=learning_rate, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, eps=1e-5)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

        print(
            f"MAC-Flow mode={self.action_mode} agents={n_agents} "
            f"per_dim={self.per_agent_dim} joint={self.joint_dim} "
            f"flow_steps={flow_steps} alpha={alpha_distill}"
        )

    @staticmethod
    def linear_epsilon(t_env: int, eps_start: float, eps_min: float, decay_steps: int) -> float:
        if decay_steps <= 0 or t_env >= decay_steps:
            return eps_min
        return eps_start + (eps_min - eps_start) * (t_env / decay_steps)

    def new_episode_state(self) -> Dict[str, Any]:
        return {}

    def reset_eval_hidden(self) -> None:
        self._eval_noise = None

    def _velocity_fn(self, t: torch.Tensor, obs_joint: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.flow(t, obs_joint, x)

    def _actions_for_critic(self, actions_idx_or_vec: torch.Tensor) -> torch.Tensor:
        """[B,N,1] int or [B,N,D] float -> [B,N,per_dim]."""
        if self.continuous:
            return actions_idx_or_vec.float()
        idx = actions_idx_or_vec[..., 0].long().clamp(0, self.per_agent_dim - 1)
        return F.one_hot(idx, self.per_agent_dim).float()

    def _flow_sample_joint(
        self, obs_joint: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b = obs_joint.shape[0]
        if z is None:
            z = torch.randn(b, self.joint_dim, device=self.device, dtype=obs_joint.dtype)
        return euler_integrate(self._velocity_fn, obs_joint, z, self.flow_steps)

    def _policy_actions(
        self, obs: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return self.policy(obs, noise)

    def _distill_target(
        self, obs: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Euler slice per agent [B,N,per_dim]. Detached so only Eq. 6 trains the flow."""
        obs_j = obs_joint_from_batch(obs)
        joint = self._flow_sample_joint(obs_j, noise.reshape(obs.shape[0], self.joint_dim))
        per_agent = split_joint_to_agents(joint, self.n_agents, self.per_agent_dim)
        if self.continuous:
            per_agent = torch.tanh(per_agent)
        return per_agent.detach()

    def _gather_transitions(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = batch["obs"]
        actions = batch["actions"]
        avail = batch.get("avail_actions")
        reward = batch["reward"]
        term = batch["terminated"]
        filled = batch["filled"]

        # filled[t] is the transition; the following index holds the next obs,
        # including the bootstrap slot whose filled flag is 0.
        valid = filled[:, :-1, 0] > 0.5
        b_idx, t_idx = torch.where(valid)
        if b_idx.numel() == 0:
            raise ValueError("MAC-Flow batch has no valid transitions")

        obs_t = obs[b_idx, t_idx]
        obs_tp1 = obs[b_idx, t_idx + 1]
        act_t = actions[b_idx, t_idx]
        r = reward[b_idx, t_idx, 0]
        done = term[b_idx, t_idx, 0].clamp(0, 1)
        avail_t = avail[b_idx, t_idx] if avail is not None else None
        avail_tp1 = avail[b_idx, t_idx + 1] if avail is not None else None
        return {
            "obs_t": obs_t,
            "obs_tp1": obs_tp1,
            "act_t": act_t,
            "reward": r,
            "done": done,
            "avail_t": avail_t,
            "avail_tp1": avail_tp1,
            "b_idx": b_idx,
            "t_idx": t_idx,
        }

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
        tr = self._gather_transitions(batch)
        obs_t = tr["obs_t"]
        obs_tp1 = tr["obs_tp1"]
        act_t = tr["act_t"]
        r = tr["reward"]
        done = tr["done"]

        obs_j_t = obs_joint_from_batch(obs_t).detach()
        obs_j_tp1 = obs_joint_from_batch(obs_tp1).detach()
        bsz = act_t.shape[0]
        if self.continuous:
            x1 = act_t.reshape(bsz, self.joint_dim)
        else:
            idx = act_t[..., 0].long().clamp(0, self.per_agent_dim - 1)
            x1 = F.one_hot(idx, self.per_agent_dim).float().reshape(bsz, self.joint_dim)

        # --- Flow BC ---
        def vfn(t, oj, x):
            return self._velocity_fn(t, oj, x)

        loss_flow = flow_bc_loss(vfn, obs_j_t, x1)

        self.flow_opt.zero_grad()
        loss_flow.backward()
        if self.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.grad_norm_clip)
        self.flow_opt.step()

        # --- Per-agent critic TD; next action is the Euler vector itself ---
        with torch.no_grad():
            z_next = torch.randn(obs_tp1.shape[0], self.joint_dim, device=self.device)
            joint_next = self._flow_sample_joint(obs_j_tp1, z_next)
            a_next = split_joint_to_agents(joint_next, self.n_agents, self.per_agent_dim)
            if self.continuous:
                a_next = torch.tanh(a_next)
            q_next = self.target_critic(obs_tp1, a_next)
            target = r.view(-1, 1, 1) + self.gamma * (1.0 - done.view(-1, 1, 1)) * q_next

        a_t = self._actions_for_critic(act_t)
        q = self.critic(obs_t, a_t)
        q_tot = vdn_tot(q)
        td_loss = F.mse_loss(q, target)

        self.critic_opt.zero_grad()
        td_loss.backward()
        if self.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_opt.step()

        # --- Actor: -Q + alpha * distill ---
        z_pol = torch.randn(obs_t.shape[0], self.joint_dim, device=self.device)
        noise_agents = split_joint_to_agents(z_pol, self.n_agents, self.per_agent_dim)
        # Policy output lives in the same vector space as the Euler slice.
        # Discrete deployment argmax stays in select_actions_eval.
        pol_out = self._policy_actions(obs_t, noise_agents)
        teacher = self._distill_target(obs_t, z_pol)
        distill = F.mse_loss(pol_out, teacher)
        a_for_q = pol_out

        q_pol = self.critic(obs_t, a_for_q)
        q_pol_tot = vdn_tot(q_pol)
        q_term = -q_pol_tot.mean()
        if self.normalize_q:
            scale = q_pol_tot.detach().abs().mean().clamp(min=1e-6)
            q_term = q_term / scale

        actor_loss = q_term + self.alpha_distill * distill

        self.policy_opt.zero_grad()
        actor_loss.backward()
        if self.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm_clip)
        self.policy_opt.step()

        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.polyak_tau).add_(p.data, alpha=self.polyak_tau)

        self.training_step_ctr += 1
        logs = {
            "loss": float((loss_flow + td_loss + actor_loss).detach().item()),
            "loss_flow": float(loss_flow.detach().item()),
            "td_loss": float(td_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "distill_loss": float(distill.detach().item()),
            "q_tot_mean": float(q_tot.detach().mean().item()),
        }
        self.last_diag = logs
        return logs

    def export_state(self) -> Dict[str, Any]:
        return {
            "flow": self.flow.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "policy": self.policy.state_dict(),
            "flow_opt": self.flow_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "policy_opt": self.policy_opt.state_dict(),
            "meta": {
                "n_agents": self.n_agents,
                "obs_dim": self.obs_dim,
                "per_agent_dim": self.per_agent_dim,
                "action_mode": self.action_mode,
                "joint_dim": self.joint_dim,
            },
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        self.flow.load_state_dict(payload["flow"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload.get("target_critic", payload["critic"]))
        self.policy.load_state_dict(payload["policy"])
        if "flow_opt" in payload:
            self.flow_opt.load_state_dict(payload["flow_opt"])
        if "critic_opt" in payload:
            self.critic_opt.load_state_dict(payload["critic_opt"])
        if "policy_opt" in payload:
            self.policy_opt.load_state_dict(payload["policy_opt"])

    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Optional[Dict[str, np.ndarray]],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        env_state=None,
    ) -> Dict[str, Union[int, np.ndarray]]:
        del state, env_state
        agents = sorted(observations.keys())
        obs_list = [observations[a].astype(np.float32) for a in agents]
        obs = torch.tensor(np.stack(obs_list), device=self.device).unsqueeze(0)
        if self.continuous:
            if eps > 0:
                noise = torch.randn(1, self.n_agents, self.per_agent_dim, device=self.device)
            else:
                noise = torch.zeros(1, self.n_agents, self.per_agent_dim, device=self.device)
            act = self.policy(obs, noise)[0].cpu().numpy()
            out = {a: act[i].astype(np.float32) for i, a in enumerate(agents)}
        else:
            if eps > 0:
                z = torch.randn(1, self.joint_dim, device=self.device)
                noise_ag = split_joint_to_agents(z, self.n_agents, self.per_agent_dim)
            else:
                noise_ag = torch.zeros(1, self.n_agents, self.per_agent_dim, device=self.device)
            logits = self.policy(obs, noise_ag)
            if legals is not None:
                avail = np.stack([legals[a] for a in agents])
                avail_t = torch.tensor(avail, device=self.device, dtype=torch.bool).unsqueeze(0)
                logits = logits.masked_fill(~avail_t, -1e9)
            idx = logits.argmax(dim=-1).reshape(-1).cpu().numpy()
            if eps > 0:
                for i, a in enumerate(agents):
                    if np.random.rand() < eps / self.n_agents:
                        legal = np.flatnonzero(legals[a]) if legals else np.arange(self.per_agent_dim)
                        idx[i] = int(np.random.choice(legal))
            out = {a: int(idx[i]) for i, a in enumerate(agents)}
        if trace is not None:
            with torch.no_grad():
                if self.continuous:
                    a_c = act.reshape(1, self.n_agents, self.per_agent_dim)
                else:
                    a_c = self._actions_for_critic(
                        torch.tensor(idx, device=self.device).reshape(1, self.n_agents, 1)
                    )
                trace["q_tot"] = float(vdn_tot(self.critic(obs, a_c)).item())
        return out
