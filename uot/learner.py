"""UOT learner: live QMIX TD-λ plus an ordinal hinge on recorded neighbors.

The frozen oracle is a copy of the loaded QMIX. It is queried for sign(Q_bar)
only. Q_bar never enters the Bellman target and never selects actions.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch

from ovmse.targets import build_td_lambda_targets
from qmix.core import QMIXBase
from qmix.networks import MarlcRNN, QMixer
from uot.hinge import ordinal_hinge
from uot.neighborhood import build_recorded_neighborhood
from uot.support import RecordedJoints


class UOTLearner(QMIXBase):
    checkpoint_stem = "uot"

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        obs_dim: int,
        state_dim: int,
        support: RecordedJoints,
        *,
        recurrent_layer_dim: int = 64,
        mixer_embed_dim: int = 32,
        mixer_hyper_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        td_lambda: float = 0.6,
        grad_norm_clip: float = 10.0,
        target_update_interval: int = 200,
        batch_size_run: int = 1,
        add_agent_id_to_obs: bool = True,
        eta: float = 0.1,
        eta_drop_at: int = -1,
        delta: float = 0.1,
        margin: float = 0.1,
        n_candidates: int = 8,
        max_hamming: int = 2,
        eps_min: float = 0.05,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.grad_norm_clip = grad_norm_clip
        self.target_update_interval = target_update_interval
        self.batch_size_run = batch_size_run
        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.eta = float(eta)
        self.eta_drop_at = int(eta_drop_at)
        self.delta = float(delta)
        self.last_diag: Dict[str, float] = {}
        self.margin = float(margin)
        self.n_candidates = int(n_candidates)
        self.max_hamming = int(max_hamming)
        self.eps_min = eps_min
        self.use_ovm = False
        self.support = support

        self.obs_dim = obs_dim
        obs_in = obs_dim + n_actions
        if add_agent_id_to_obs:
            obs_in += n_agents

        self.q_network = MarlcRNN(obs_in, recurrent_layer_dim, n_actions).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.mixer = QMixer(n_agents, state_dim, mixer_embed_dim, mixer_hyper_dim).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.frozen_q_network: Optional[MarlcRNN] = None
        self.frozen_mixer: Optional[QMixer] = None

        params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        self.episode_counter = 0
        self.last_target_update_episode = 0
        self.t_env = 0
        self.eval_rnn: Dict[str, torch.Tensor] = {}
        self.eval_last_actions: Dict[str, Optional[int]] = {}
        self.eval_offline_rnn = None
        self.eval_offline_last_actions = None
        self.training_step_ctr = 0
        print(
            f"UOT S_off={len(self.support)} eta={self.eta} eta_drop_at={self.eta_drop_at} "
            f"delta={self.delta} margin={self.margin} k={self.n_candidates} "
            f"max_hamming={self.max_hamming}"
        )

    def freeze_oracle(self) -> None:
        """Copy the loaded live QMIX and stop its gradients. Call after checkpoint load."""
        frozen_q = copy.deepcopy(self.q_network).to(self.device).eval()
        frozen_mixer = copy.deepcopy(self.mixer).to(self.device).eval()
        for net in (frozen_q, frozen_mixer):
            for param in net.parameters():
                param.requires_grad_(False)
        self.frozen_q_network = frozen_q
        self.frozen_mixer = frozen_mixer
        print(
            "UOT froze oracle (bar_q, bar_f). Live QMIX stays trainable. "
            "Q_off is not a TD target."
        )

    def _qtot_candidates(
        self, mixer: QMixer, qvals: torch.Tensor, cand: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """cand [K,B,T,N], qvals [B,T,N,A] -> Q_tot [K,B,T,1]."""
        k, b, t, n = cand.shape
        idx = cand.reshape(k * b, t, n).unsqueeze(-1)
        q_rep = qvals.unsqueeze(0).expand(k, -1, -1, -1, -1).reshape(k * b, t, n, qvals.shape[-1])
        taken = torch.gather(q_rep, dim=3, index=idx).squeeze(-1)
        st = state.unsqueeze(0).expand(k, -1, -1, -1).reshape(k * b, t, state.shape[-1])
        return mixer(taken, st).view(k, b, t, -1)

    def _masked_ood_frac(self, joints: torch.Tensor, mask: torch.Tensor) -> float:
        actions = joints.long()
        if actions.dim() == mask.dim() + 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        inside = self.support.contains(actions)
        weight = (mask.reshape(inside.shape) > 0).float()
        total = float(weight.sum().item())
        if total < 1:
            return float("nan")
        return float(((1.0 - inside.float()) * weight).sum().item() / total)

    def _forward_net(self, net: MarlcRNN, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self._augment_obs(batch["obs"], batch["actions"])
        resets = self._resets_from_batch(batch)
        return self._unroll(net, obs, resets)

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
        del cql_weight, use_ovm, eps
        t_env = self.t_env if t_env is None else t_env

        actions = batch["actions"]
        avail = batch["avail_actions"]
        states = batch["state"]
        rewards = batch["reward"]
        terminated = batch["terminated"]
        filled = batch["filled"]

        term = terminated[:, :-1].float()
        if term.dim() == 3 and term.shape[2] > 1:
            term = term.max(dim=2, keepdim=True)[0]
        mask = filled[:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1.0 - term[:, :-1])

        with torch.no_grad():
            target_mac_out = self._forward_qs(batch, target=True)
        mac_out = self._forward_qs(batch, target=False)

        chosen_q = torch.gather(mac_out, 3, actions.long()).squeeze(-1)
        mac_sel = torch.where(avail, mac_out, torch.tensor(-9999999.0, device=self.device))
        max_actions = mac_sel.argmax(dim=3, keepdim=True)
        target_max_q = torch.gather(target_mac_out, 3, max_actions).squeeze(-1)

        chosen_tot = self.mixer(chosen_q, states)
        with torch.no_grad():
            target_max_tot = self.target_mixer(target_max_q, states)
        td_targets = build_td_lambda_targets(
            rewards[:, :-1],
            term,
            mask,
            target_max_tot,
            self.gamma,
            self.td_lambda,
        ).detach()
        td_err = 0.5 * (chosen_tot[:, :-1] - td_targets) ** 2
        loss_td = (td_err * mask).sum() / mask.sum().clamp(min=1.0)

        ord_stats = {
            "uot_cand_in_frac": 0.0,
            "uot_n_pairs": 0.0,
            "uot_hinge_rate": 0.0,
            "uot_hinge_rate_std": 0.0,
            "uot_sign_disagree": 0.0,
        }
        loss_ord = chosen_tot.new_zeros(())
        eta_now = 0.0 if self.eta_drop_at >= 0 and int(t_env) >= self.eta_drop_at else self.eta
        if self.eta > 0 and self.frozen_q_network is not None and self.frozen_mixer is not None:
            with torch.no_grad():
                bar_out = self._forward_net(self.frozen_q_network, batch)
            cand, valid = build_recorded_neighborhood(
                mac_out[:, :-1],
                bar_out[:, :-1],
                avail[:, :-1],
                actions[:, :-1],
                self.support,
                k=self.n_candidates,
                max_hamming=self.max_hamming,
            )
            q_live = self._qtot_candidates(self.mixer, mac_out[:, :-1], cand, states[:, :-1])
            with torch.no_grad():
                q_bar = self._qtot_candidates(
                    self.frozen_mixer, bar_out[:, :-1], cand, states[:, :-1]
                )
            loss_ord, ord_stats = ordinal_hinge(
                q_live,
                q_bar,
                cand,
                valid,
                mask.squeeze(-1) > 0,
                delta=self.delta,
                margin=self.margin,
            )
        loss = loss_td + eta_now * loss_ord

        self.training_step_ctr += 1
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.tensor(0.0, device=self.device)
        if self.grad_norm_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.q_network.parameters()) + list(self.mixer.parameters()),
                self.grad_norm_clip,
            )
        self.optimizer.step()

        self.episode_counter += int(episode_delta)
        if (
            self.episode_counter - self.last_target_update_episode
        ) / self.target_update_interval >= 1.0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.last_target_update_episode = self.episode_counter

        logs = {
            "loss": float(loss.detach().item()),
            "loss_td": float(loss_td.detach().item()),
            "loss_ord": float(loss_ord.detach().item()),
            "uot_eta_now": float(eta_now),
            "grad_norm": float(grad_norm.detach().item()),
            "mean_chosen_q_values": float(chosen_tot.detach().mean().item()),
            "n_soff": float(len(self.support)),
            "exec_ood_frac": self._masked_ood_frac(actions[:, :-1], mask),
            "unres_backup_ood_frac": self._masked_ood_frac(max_actions[:, :-1], mask),
        }
        logs.update(ord_stats)
        self.last_diag = {
            "uot_sign_disagree": float(ord_stats["uot_sign_disagree"]),
            "uot_hinge_rate": float(ord_stats["uot_hinge_rate"]),
            "uot_n_pairs": float(ord_stats["uot_n_pairs"]),
            "uot_eta_now": float(eta_now),
        }
        return logs
