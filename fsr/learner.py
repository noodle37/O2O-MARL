"""FSR learner: QMIX base + teammate flow support, never a value floor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from fsr.flow import flow_matching_loss, log_prob_ode, score_from_velocity
from fsr.networks import HistoryEncoder, ScaleNet, SupportFlowBank, splice_teammate_onehot, teammate_onehot
from fsr.prior import ordinal_q_hinge
from fsr.support import VisitDecay, mix_fraction, reliance_weight
from ovmse.diagnostics import ovm_target_stats
from ovmse.learner import PaperQMIXLearner
from ovmse.targets import build_td_lambda_targets


class FSRLearner(PaperQMIXLearner):
    """CTDE QMIX critic/actor with a frozen teammate-action flow as support."""

    checkpoint_stem = "fsr"
    moderate_lambda = False

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        obs_dim: int,
        state_dim: int,
        *,
        shared_flow: bool = False,
        shape_critic: bool = True,
        shape_weight: float = 0.1,
        flow_hidden: Sequence[int] = (128, 128, 128),
        encoder_hidden: int = 64,
        flow_lr: float = 3e-4,
        flow_steps: int = 8,
        score_t: float = 0.9,
        calib_quantile: float = 0.10,
        w_temperature: float = 1.0,
        mix_fn: str = "linear",
        mix_max: float = 0.5,
        prior_eta: float = 0.1,
        prior_margin: float = 0.1,
        prior_delta: float = 0.1,
        visit_scale: float = 50.0,
        visit_bins: int = 8,
        rotate_explore: bool = True,
        rotate_period: int = 1,
        logp_max_rows: int = 2048,
        shape_max_rows: int = 256,
        t_embed_frequencies: int = 8,
        **kwargs,
    ):
        super().__init__(
            n_agents=n_agents,
            n_actions=n_actions,
            obs_dim=obs_dim,
            state_dim=state_dim,
            **kwargs,
        )
        self.state_dim = int(state_dim)
        self.shared_flow = bool(shared_flow)
        self.shape_critic = bool(shape_critic)
        self.shape_weight = float(shape_weight)
        self.flow_steps = int(flow_steps)
        self.score_t = float(score_t)
        self.calib_quantile = float(calib_quantile)
        self.w_temperature = float(w_temperature)
        self.mix_fn = str(mix_fn)
        self.mix_max = float(mix_max)
        self.prior_eta = float(prior_eta)
        self.prior_margin = float(prior_margin)
        self.prior_delta = float(prior_delta)
        self.rotate_explore = bool(rotate_explore)
        self.rotate_period = max(1, int(rotate_period))
        self.logp_max_rows = int(logp_max_rows)
        self.shape_max_rows = int(shape_max_rows)
        self.teammate_dim = (self.n_agents - 1) * self.n_actions
        self.encoder_hidden = int(encoder_hidden)
        self.flow_frozen = False
        self.support_threshold = 0.0
        self._episode_ctr = 0
        self.last_diag: Dict[str, float] = {}
        self.last_support: Dict[str, float] = {}

        obs_in = self.obs_dim + self.n_actions
        if self.add_agent_id_to_obs:
            obs_in += self.n_agents
        hidden = [int(h) for h in flow_hidden]
        self.flow_encoder = HistoryEncoder(obs_in, self.encoder_hidden).to(self.device)
        self.support_flow = SupportFlowBank(
            self.n_agents,
            self.encoder_hidden,
            self.teammate_dim,
            hidden,
            shared=self.shared_flow,
            t_freqs=int(t_embed_frequencies),
        ).to(self.device)
        self.scale_net = ScaleNet(self.state_dim).to(self.device)
        self.flow_optimizer = torch.optim.Adam(
            list(self.flow_encoder.parameters()) + list(self.support_flow.parameters()),
            lr=float(flow_lr),
        )
        self.scale_optimizer = torch.optim.Adam(self.scale_net.parameters(), lr=float(flow_lr))
        self.visit = VisitDecay(
            self.state_dim,
            n_bins=int(visit_bins),
            scale=float(visit_scale),
            seed=0,
            device=self.device,
        )
        self.moderate_lambda = False
        print(
            f"FSR agents={n_agents} actions={n_actions} teammate_dim={self.teammate_dim} "
            f"shared_flow={self.shared_flow} shape={self.shape_critic} κ={self.shape_weight} "
            f"flow_steps={self.flow_steps} mix_max={self.mix_max} rotate={self.rotate_explore}"
        )

    def freeze_flow(self) -> None:
        self.flow_frozen = True
        self.flow_encoder.eval()
        self.support_flow.eval()
        for p in self.flow_encoder.parameters():
            p.requires_grad_(False)
        for p in self.support_flow.parameters():
            p.requires_grad_(False)

    def _history_hidden(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self._augment_obs(batch["obs"], batch["actions"])
        resets = self._resets_from_batch(batch)
        return self._unroll(self.flow_encoder, obs, resets)

    def _agent_velocity(self, agent_i: int):
        def _fn(cond: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.support_flow.velocity(agent_i, cond, x, t)

        return _fn

    def _packed_agent_rows(
        self,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        mask_bt: torch.Tensor,
        max_rows: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Per-agent packed (cond, x1, row_mask) on filled transition steps."""
        mask = mask_bt > 0.5
        packs = []
        for i in range(self.n_agents):
            cond = hidden[:, :-1, i]
            x1 = teammate_onehot(actions[:, :-1], i, self.n_actions)
            cond_f = cond.reshape(-1, cond.shape[-1])
            x1_f = x1.reshape(-1, x1.shape[-1])
            m = mask.reshape(-1)
            if max_rows is not None and int(m.sum().item()) > max_rows:
                idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
                pick = idx[torch.randperm(idx.numel(), device=idx.device)[:max_rows]]
                sel = torch.zeros_like(m)
                sel[pick] = True
                packs.append({"cond": cond_f[sel], "x1": x1_f[sel], "mask": None, "full_mask": m})
            else:
                packs.append({"cond": cond_f, "x1": x1_f, "mask": m.float(), "full_mask": m})
        return packs

    def _flow_bc(self, hidden: torch.Tensor, actions: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
        packs = self._packed_agent_rows(hidden, actions, mask_bt)
        loss = hidden.new_zeros(())
        for i, pack in enumerate(packs):
            loss = loss + flow_matching_loss(
                self._agent_velocity(i), pack["cond"], pack["x1"], mask=pack["mask"]
            )
        return loss / float(self.n_agents)

    @torch.no_grad()
    def _agent_logp(
        self, hidden: torch.Tensor, actions: torch.Tensor, mask_bt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logp, scored) with shape [B, T-1, N]; unscored entries are False."""
        b, t_full, n_ag, hid_dim = hidden.shape
        t = t_full - 1
        logp = hidden.new_zeros(b, t, n_ag)
        scored = torch.zeros(b, t, n_ag, dtype=torch.bool, device=hidden.device)
        filled = (mask_bt > 0.5).reshape(-1)
        for i in range(n_ag):
            cond = hidden[:, :-1, i].reshape(-1, hid_dim)
            x1 = teammate_onehot(actions[:, :-1], i, self.n_actions).reshape(
                -1, self.teammate_dim
            )
            idx = torch.nonzero(filled, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            if idx.numel() > self.logp_max_rows:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[: self.logp_max_rows]]
            values = log_prob_ode(
                self._agent_velocity(i), cond[idx], x1[idx], steps=self.flow_steps
            )
            flat = logp[:, :, i].reshape(-1)
            flag = scored[:, :, i].reshape(-1)
            flat[idx] = values
            flag[idx] = True
            logp[:, :, i] = flat.reshape(b, t)
            scored[:, :, i] = flag.reshape(b, t)
        return logp, scored

    def _support_from_logp(self, logp: torch.Tensor, scored: torch.Tensor) -> torch.Tensor:
        w = reliance_weight(logp, self.support_threshold, self.w_temperature)
        return w * scored.float()

    @torch.no_grad()
    def support_mix_ratio(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Score an online batch and return the offline replay fraction for this step."""
        hidden = self._history_hidden(batch)
        filled = batch["filled"][:, :-1, 0]
        logp, scored = self._agent_logp(hidden, batch["actions"], filled)
        w = self._support_from_logp(logp, scored)
        denom = scored.float().sum().clamp(min=1.0)
        w_mean = (w.sum() / denom).clamp(0.0, 1.0)
        logp_mean = (logp * scored.float()).sum() / denom
        frac = mix_fraction(w_mean, self.mix_fn, self.mix_max)
        info = {
            "fsr/w_mean": float(w_mean.item()),
            "fsr/logp_mean": float(logp_mean.item()),
            "fsr/offline_frac": float(frac.item() if torch.is_tensor(frac) else frac),
            "fsr/threshold": float(self.support_threshold),
            "fsr/scored_frac": float((denom / scored.numel()).item()),
        }
        self.last_support = info
        return info

    def _shape_loss(
        self,
        mac_out: torch.Tensor,
        hidden: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        mask_bt: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Match ∇_{a_{-i}} Q_tot to α(s) * score, score detached."""
        if not self.shape_critic or self.shape_weight <= 0:
            return mac_out.new_zeros(())
        actions = batch["actions"]
        states = batch["state"][:, :-1]
        oh = F.one_hot(actions[:, :-1, :, 0].long().clamp(0, self.n_actions - 1), self.n_actions).float()
        q_slice = mac_out[:, :-1]
        mask_f = (mask_bt > 0.5).float()
        flat_idx = torch.nonzero(mask_f.reshape(-1) > 0.5, as_tuple=False).squeeze(-1)
        if flat_idx.numel() == 0:
            return mac_out.new_zeros(())
        if flat_idx.numel() > self.shape_max_rows:
            flat_idx = flat_idx[torch.randperm(flat_idx.numel(), device=flat_idx.device)[: self.shape_max_rows]]
        loss = mac_out.new_zeros(())
        n_used = 0
        for i in range(self.n_agents):
            x1_full = teammate_onehot(actions[:, :-1], i, self.n_actions).reshape(-1, self.teammate_dim)
            x1 = x1_full[flat_idx].detach().requires_grad_(True)
            oh_flat = oh.reshape(-1, self.n_agents, self.n_actions)[flat_idx]
            joint = splice_teammate_onehot(oh_flat.detach(), x1, i)
            q_flat = q_slice.reshape(-1, self.n_agents, self.n_actions)[flat_idx]
            chosen = (q_flat * joint).sum(dim=-1).unsqueeze(1)
            st = states.reshape(-1, states.shape[-1])[flat_idx].unsqueeze(1)
            q_tot = self.mixer(chosen, st)
            grad = torch.autograd.grad(q_tot.sum(), x1, create_graph=True, retain_graph=True)[0]
            cond = hidden[:, :-1, i].reshape(-1, hidden.shape[-1])[flat_idx]
            score = score_from_velocity(
                self._agent_velocity(i), cond, x1.detach(), t_eval=self.score_t
            )
            alpha = self.scale_net(st.squeeze(1))
            resid = grad - alpha * score.detach()
            per = resid.pow(2).mean(dim=-1)
            w = mask_f.new_ones(flat_idx.shape[0])
            if weight is not None:
                w = weight[:, :, i].reshape(-1)[flat_idx]
            denom = w.sum().clamp(min=1.0)
            loss = loss + (per * w).sum() / denom
            n_used += 1
        return loss / float(max(n_used, 1))

    def calibrate_support(self, dataset, *, n_batches: int = 8, batch_size: int = 64, seed: int = 0) -> Dict[str, float]:
        """Hold-out real combinations → log-p quantile used as the support threshold."""
        was_frozen = self.flow_frozen
        self.flow_encoder.eval()
        self.support_flow.eval()
        rng = np.random.default_rng(int(seed) + 13)
        values: List[float] = []
        n_ep = min(int(dataset.n_episodes), max(1, n_batches * batch_size))
        for start in range(0, n_ep, batch_size):
            take = min(batch_size, n_ep - start)
            batch = dataset.sample(take, rng, device=self.device)
            hidden = self._history_hidden(batch)
            filled = batch["filled"][:, :-1, 0]
            logp, scored = self._agent_logp(hidden, batch["actions"], filled)
            values.extend(logp[scored].reshape(-1).detach().cpu().tolist())
        if not values:
            self.support_threshold = 0.0
            return {"fsr/threshold": 0.0, "fsr/calib_n": 0.0}
        arr = torch.tensor(values, dtype=torch.float32)
        self.support_threshold = float(torch.quantile(arr, self.calib_quantile).item())
        std = float(arr.std().clamp(min=1.0).item())
        self.w_temperature = max(self.w_temperature, std)
        stats = {
            "fsr/threshold": self.support_threshold,
            "fsr/calib_n": float(len(values)),
            "fsr/calib_mean": float(arr.mean().item()),
            "fsr/calib_std": float(arr.std().item()),
            "fsr/w_temperature": float(self.w_temperature),
        }
        print(
            f"FSR calibrated threshold={self.support_threshold:.4f} "
            f"q={self.calib_quantile} n={len(values)} mean={stats['fsr/calib_mean']:.4f}"
        )
        if was_frozen:
            self.freeze_flow()
        return stats

    def train_on_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        cql_weight: Optional[float] = None,
        use_ovm: Optional[bool] = None,
        t_env: Optional[int] = None,
        episode_delta: int = 1,
        offline_bs: int = 0,
        support_w_mean: Optional[float] = None,
    ) -> Dict[str, float]:
        del use_ovm
        cql_w = self.cql_weight if cql_weight is None else cql_weight
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
        mask_exp = mask
        mask_bt = mask[..., 0]

        with torch.no_grad():
            target_mac_out = self._forward_qs(batch, target=True)
        mac_out = self._forward_qs(batch, target=False)
        chosen_q = torch.gather(mac_out, 3, actions.long())
        mac_sel = torch.where(avail, mac_out, torch.tensor(-9999999.0, device=self.device))
        max_actions = torch.argmax(mac_sel, dim=3, keepdim=True)
        target_max_q = torch.gather(target_mac_out, 3, max_actions)
        chosen_tot, target_max_tot, mixed_rew = self._mixing(
            chosen_q, target_max_q, states, rewards
        )
        td_targets = build_td_lambda_targets(
            mixed_rew[:, :-1],
            terminated[:, :-1].max(dim=2, keepdim=True)[0],
            mask_exp,
            target_max_tot,
            self.gamma,
            self.td_lambda,
        ).detach()

        # Pure online / standard TD. OVM never enters the target.
        td_err = 0.5 * (chosen_tot[:, :-1] - td_targets) ** 2
        loss = (td_err * mask_exp).sum() / mask_exp.sum().clamp(min=1.0)

        hidden = self._history_hidden(batch)
        logs: Dict[str, float] = {}

        if not self.flow_frozen:
            flow_loss = self._flow_bc(hidden, actions, mask_bt)
            self.flow_optimizer.zero_grad()
            flow_loss.backward()
            self.flow_optimizer.step()
            logs["fsr/flow_loss"] = float(flow_loss.detach().item())
            if self.shape_critic and self.shape_weight > 0:
                shape = self._shape_loss(mac_out, hidden.detach(), batch, mask_bt)
                loss = loss + self.shape_weight * shape
                logs["fsr/shape_loss"] = float(shape.detach().item())
        else:
            logp, scored = self._agent_logp(hidden, actions, mask_bt)
            w = self._support_from_logp(logp, scored)
            denom = scored.float().sum().clamp(min=1.0)
            w_mean = (w.sum() / denom).detach()
            if support_w_mean is not None:
                w_mean = torch.tensor(float(support_w_mean), device=self.device)

            online_mask = torch.zeros_like(mask_bt)
            if offline_bs <= 0:
                online_mask = mask_bt
            elif offline_bs < mask_bt.shape[0]:
                online_mask[offline_bs:] = mask_bt[offline_bs:]
            lam = self.visit.lambdas(states[:, :-1], increment=True, mask=online_mask)
            if online_mask.sum() > 0:
                lam_mean = (lam * online_mask).sum() / online_mask.sum().clamp(min=1.0)
            else:
                lam_mean = mask_bt.new_zeros(())

            prior_strength = w_mean * lam_mean
            if self.prior_eta > 0 and self.offline_q_network is not None:
                with torch.no_grad():
                    off_q = self._unroll(
                        self.offline_q_network,
                        self._augment_obs(batch["obs"], batch["actions"]),
                        self._resets_from_batch(batch),
                    )
                hinge, hinge_stats = ordinal_q_hinge(
                    mac_out[:, :-1],
                    off_q[:, :-1],
                    avail[:, :-1],
                    mask_bt,
                    margin=self.prior_margin,
                    delta=self.prior_delta,
                )
                loss = loss + self.prior_eta * prior_strength * hinge
                logs.update(hinge_stats)
                logs["fsr/prior_loss"] = float(hinge.detach().item())
            logs["fsr/w_mean"] = float(w_mean.item())
            logs["fsr/logp_mean"] = float(((logp * scored.float()).sum() / denom).item())
            logs["fsr/lambda_visit"] = float(lam_mean.item()) if torch.is_tensor(lam_mean) else 0.0
            logs["fsr/prior_strength"] = (
                float(prior_strength.item()) if torch.is_tensor(prior_strength) else 0.0
            )
            logs["fsr/threshold"] = float(self.support_threshold)
            if self.shape_critic and self.shape_weight > 0:
                shape = self._shape_loss(
                    mac_out, hidden.detach(), batch, mask_bt, weight=w.detach()
                )
                loss = loss + self.shape_weight * w_mean * shape
                logs["fsr/shape_loss"] = float(shape.detach().item())

        if cql_w > 0:
            if self.raw_cql:
                cql_loss = self._compute_macql_loss(mac_out, batch, chosen_tot, mask_exp)
            else:
                cql_loss = self._compute_cfcql_loss(mac_out, batch, chosen_tot, mask_exp)
            loss = loss + cql_w * cql_loss
            logs["cql_loss"] = float(cql_loss.detach().item())
        else:
            logs["cql_loss"] = 0.0

        self.training_step_ctr += 1
        self.optimizer.zero_grad()
        if self.shape_critic:
            self.scale_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.tensor(0.0, device=self.device)
        if self.grad_norm_clip > 0:
            params = list(self.q_network.parameters()) + list(self.mixer.parameters())
            if self.shape_critic:
                params += list(self.scale_net.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self.grad_norm_clip)
        self.optimizer.step()
        if self.shape_critic:
            self.scale_optimizer.step()

        self.episode_counter += int(episode_delta)
        if (
            self.episode_counter - self.last_target_update_episode
        ) / self.target_update_interval >= 1.0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.last_target_update_episode = self.episode_counter

        logs.update(
            {
                "loss": float(loss.detach().item()),
                "grad_norm": float(grad_norm.detach().item()),
                "mean_chosen_q_values": float(chosen_tot.detach().mean().item()),
                "lambda_memory": 0.0,
            }
        )
        if self.ovm_diag_train_stats and self.offline_q_network is not None:
            stride_hit = (self.training_step_ctr % self.ovm_diag_stride) == 0
            if stride_hit:
                off_tot = self._offline_chosen_tot(batch, actions, states)
                logs.update(
                    ovm_target_stats(
                        off_tot,
                        td_targets,
                        mask_exp,
                        0.0,
                        chosen_tot=chosen_tot,
                        prefix="train_ovm/",
                    )
                )
        self.last_diag = logs
        return logs

    def new_episode_state(self) -> Dict[str, Any]:
        state = super().new_episode_state()
        offline_net = getattr(self, "offline_q_network", None)
        if self.rotate_explore and self.flow_frozen and offline_net is not None:
            state["offline_rnn"] = {
                f"agent_{i}": offline_net.initial_state(1, self.device)
                for i in range(self.n_agents)
            }
            state["offline_last"] = {}
            bucket = self._episode_ctr // self.rotate_period
            state["explorer"] = int(bucket % self.n_agents)
            self._episode_ctr += 1
        return state

    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Dict[str, np.ndarray],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        env_state=None,
    ) -> Dict[str, int]:
        rotate = (
            state is not None
            and self.rotate_explore
            and self.flow_frozen
            and getattr(self, "offline_q_network", None) is not None
            and "explorer" in state
        )
        if not rotate:
            return super().select_actions_eval(
                observations, legals, eps, state=state, trace=trace, env_state=env_state
            )

        rnn_state = state["rnn"]
        last_actions = state["last"]
        off_rnn = state["offline_rnn"]
        off_last = state["offline_last"]
        explorer = int(state["explorer"])
        epsilon_dec = 0.0 if eps <= 0.0 else eps / self.n_agents
        actions: Dict[str, int] = {}
        chosen_q: List[float] = []
        for i, agent in enumerate(sorted(observations.keys())):
            use_offline = i != explorer
            net = self.offline_q_network if use_offline else self.q_network
            hid = off_rnn if use_offline else rnn_state
            last = off_last if use_offline else last_actions
            obs = torch.from_numpy(observations[agent]).float().to(self.device)
            last_idx = last.get(agent)
            if last_idx is None:
                last_oh = torch.zeros(self.n_actions, device=self.device)
            else:
                last_oh = F.one_hot(
                    torch.tensor(int(last_idx), device=self.device), self.n_actions
                ).float()
            pieces = [obs, last_oh]
            if self.add_agent_id_to_obs:
                pieces.append(F.one_hot(torch.tensor(i, device=self.device), self.n_agents).float())
            obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
            key = agent if agent in hid else f"agent_{i}"
            q, hid[key] = net(obs_in, hid.get(key))
            legal = torch.from_numpy(legals[agent]).bool().to(self.device)
            q = q[0].masked_fill(~legal, -1e9)
            explore_this = (not use_offline) and np.random.rand() < epsilon_dec
            if explore_this:
                legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
                a = int(legal_idx[torch.randint(legal_idx.numel(), (1,))].item())
            else:
                a = int(torch.argmax(q).item())
            last[agent] = a
            # Keep the unused net's last-action aligned with the executed action.
            if use_offline:
                last_actions[agent] = a
            else:
                off_last[agent] = a
            actions[agent] = a
            if trace is not None:
                chosen_q.append(float(q[a].item()))
        if trace is not None:
            trace["chosen_q"] = chosen_q
            trace["n_disagree"] = 0
            trace["n_agents"] = len(chosen_q)
            trace["fsr/explorer"] = explorer
        return actions

    def state_dicts(self) -> Dict[str, Any]:
        payload = super().state_dicts()
        payload.update(
            {
                "fsr_encoder": self.flow_encoder.state_dict(),
                "fsr_flow": self.support_flow.state_dict(),
                "fsr_scale": self.scale_net.state_dict(),
                "fsr_visit": self.visit.state_dict(),
                "fsr_meta": {
                    "threshold": self.support_threshold,
                    "w_temperature": self.w_temperature,
                    "flow_frozen": self.flow_frozen,
                    "shared_flow": self.shared_flow,
                    "encoder_hidden": self.encoder_hidden,
                    "teammate_dim": self.teammate_dim,
                },
            }
        )
        return payload

    def load_fsr_state(self, payload: Dict[str, Any]) -> None:
        if "fsr_encoder" in payload:
            self.flow_encoder.load_state_dict(payload["fsr_encoder"])
        if "fsr_flow" in payload:
            self.support_flow.load_state_dict(payload["fsr_flow"])
        if "fsr_scale" in payload:
            self.scale_net.load_state_dict(payload["fsr_scale"])
        if "fsr_visit" in payload:
            self.visit.load_state_dict(payload["fsr_visit"])
        meta = payload.get("fsr_meta") or {}
        if "threshold" in meta:
            self.support_threshold = float(meta["threshold"])
        if "w_temperature" in meta:
            self.w_temperature = float(meta["w_temperature"])
        if meta.get("flow_frozen"):
            self.freeze_flow()
