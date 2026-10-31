# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""VCKR learner: validity-gated sub-team reuse on top of QMIX TD.

Value channel (replaces OVM's global max): the TD target receives a positive
advantage lower bound only for sub-teams whose counterfactual validity is
non-negative at that state.  A unit with Delta < 0 is absent from the target,
so an expired snippet can never hold the online policy below its own ceiling.

Exploration channel (replaces global epsilon scheduling): agents covered by a
selected unit hand over to their offline snippet for H steps; the remaining
agents keep the exploration budget.  A distilled local gate head decides this
decentralized at execution time, because the training-time selection uses the
centralized critic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ovmse.learner import PaperQMIXLearner
from ovmse.targets import build_td_lambda_targets
from vckr.gate import GateHead
from vckr.knowledge import SubteamKnowledge, VCKRParams, build_subteam_knowledge
from vckr.validity import validity_inference


class VCKRLearner(PaperQMIXLearner):
    checkpoint_stem = "vckr"

    def __init__(
        self,
        *args: Any,
        vckr_params: Optional[VCKRParams] = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("moderate_lambda", False)
        kwargs.setdefault("use_ovm", False)
        super().__init__(*args, **kwargs)
        self.vckr_params = vckr_params or VCKRParams()
        self.gate_head = GateHead(
            self.obs_dim,
            self.n_agents,
            hidden=self.vckr_params.gate_hidden,
            bias_init=self.vckr_params.gate_bias_init,
        ).to(self.device)
        self.gate_optimizer = torch.optim.Adam(
            self.gate_head.parameters(), lr=self.vckr_params.gate_lr
        )
        self.knowledge: Optional[SubteamKnowledge] = None
        self.sequential_explore = False
        self.eval_lock_left: Dict[str, int] = {}

    # ------------------------------------------------------------- offline

    def setup_vckr(self, dataset: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
        self.knowledge, info = build_subteam_knowledge(
            self, dataset, self.vckr_params, seed=int(cfg["seed"])
        )
        print(
            f"[vckr] units={self.knowledge.units} "
            f"coupling_max={info['vckr/coupling_max']:.4f} "
            f"delta_temp={self.knowledge.delta_temp:.4f} "
            f"return_std={self.knowledge.return_std:.4f}",
            flush=True,
        )
        return info

    # ------------------------------------------------------------- training

    def train_on_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        cql_weight: Optional[float] = None,
        use_ovm: Optional[bool] = None,
        t_env: Optional[int] = None,
        episode_delta: int = 1,
    ) -> Dict[str, float]:
        del cql_weight, use_ovm
        if self.knowledge is None:
            raise RuntimeError("VCKR knowledge is not built; call setup_vckr() first")
        t_env = self.t_env if t_env is None else t_env
        params = self.vckr_params

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

        with torch.no_grad():
            target_mac_out = self._forward_qs(batch, target=True)
        mac_out = self._forward_qs(batch, target=False)
        with torch.no_grad():
            off_out = self._offline_unroll(batch)

        chosen_q = torch.gather(mac_out, 3, actions.long())
        mac_sel = torch.where(avail, mac_out, torch.tensor(-1e9, device=self.device))
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

        validity = validity_inference(
            self, batch, mac_out.detach(), off_out, self.knowledge, params
        )
        bonus = float(params.bonus_scale) * validity["bonus"][:, :-1].unsqueeze(-1)
        targets = td_targets + bonus
        td_err = 0.5 * (chosen_tot[:, :-1] - targets) ** 2
        loss = (td_err * mask_exp).sum() / mask_exp.sum().clamp(min=1.0)

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

        gate_stats = self._train_gate(batch, validity, mask_exp)
        self.episode_counter += int(episode_delta)
        if (
            self.episode_counter - self.last_target_update_episode
        ) / self.target_update_interval >= 1.0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.last_target_update_episode = self.episode_counter

        fill = filled[:, :, 0] > 0.5
        stats = {
            "delta_on": validity["delta_on"].mean(dim=0)[fill].mean(),
            "delta_off": validity["delta_off"].mean(dim=0)[fill].mean(),
            "delta_hat": validity["delta_hat"].mean(dim=0)[fill].mean(),
            "alpha": validity["alpha"].mean(dim=0)[fill].mean(),
            "tv": validity["tv"].mean(dim=0)[fill].mean(),
            "selected_units": validity["selected"].float().sum(dim=0)[fill].mean(),
            "selected_frac": validity["member"].float()[fill].mean(),
            "bonus_mean": validity["bonus"][fill].mean(),
            "bonus_nonzero_frac": (validity["bonus"][fill] > 1e-6).float().mean(),
        }
        self.last_diag = {
            "vckr/diag_td_target": float(td_targets[fill[:, :-1]].mean().item()),
            "vckr/diag_bonus_mean": float(bonus[fill[:, :-1]].mean().item()),
            "vckr/diag_bonus_max": float(bonus.max().item()),
            "vckr/diag_chosen_tot": float(chosen_tot[:, :-1][fill[:, :-1]].mean().item()),
            "vckr/diag_loss": float(loss.detach().item()),
            **{f"vckr/diag_{k.rsplit('/', 1)[-1]}": float(v) for k, v in gate_stats.items()},
            **{f"vckr/diag_{k}": float(v.detach().item()) for k, v in stats.items()},
        }
        return {
            "loss": float(loss.detach().item()),
            "cql_loss": 0.0,
            "grad_norm": float(grad_norm.detach().item()),
            "mean_chosen_q_values": float(chosen_tot.detach().mean().item()),
            "lambda_memory": 0.0,
            **{f"vckr/{k}": float(v.item()) for k, v in stats.items()},
            **gate_stats,
        }

    def _train_gate(
        self,
        batch: Dict[str, torch.Tensor],
        validity: Dict[str, torch.Tensor],
        mask_exp: torch.Tensor,
    ) -> Dict[str, float]:
        obs = batch["obs"][:, :-1]
        logits = self.gate_head(obs)
        target = validity["member"][:, :-1].float()
        mask_agent = mask_exp.expand(-1, -1, self.n_agents)
        pos_weight = torch.tensor(
            float(self.vckr_params.gate_pos_weight), device=self.device
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=pos_weight
        )
        gate_loss = (bce * mask_agent).sum() / mask_agent.sum().clamp(min=1.0)
        self.gate_optimizer.zero_grad()
        gate_loss.backward()
        self.gate_optimizer.step()
        valid = mask_agent > 0.5
        pred = logits > 0.0
        return {
            "vckr/gate_loss": float(gate_loss.detach().item()),
            "vckr/gate_pos_rate": float(target[valid].mean().item()) if bool(valid.any()) else 0.0,
            "vckr/gate_pred_rate": float(pred[valid].float().mean().item()) if bool(valid.any()) else 0.0,
            "vckr/gate_acc": float((pred[valid] == (target[valid] > 0.5)).float().mean().item())
            if bool(valid.any())
            else 0.0,
        }

    # ------------------------------------------------------------ checkpoint

    def state_dicts(self) -> Dict[str, Any]:
        out = super().state_dicts()
        out["vckr_gate"] = self.gate_head.state_dict()
        return out

    def load_offline_state(self, payload: Dict[str, Any]) -> None:
        if "vckr_gate" in payload:
            self.gate_head.load_state_dict(payload["vckr_gate"])

    # --------------------------------------------------------------- rollout

    def _offline_initial_state(self) -> Dict[str, torch.Tensor]:
        assert self.offline_q_network is not None
        return {
            f"agent_{i}": self.offline_q_network.initial_state(1, self.device)
            for i in range(self.n_agents)
        }

    def new_episode_state(self) -> Dict[str, Any]:
        state = super().new_episode_state()
        state["lock_left"] = {f"agent_{i}": 0 for i in range(self.n_agents)}
        if self.offline_q_network is not None:
            state["offline_rnn"] = self._offline_initial_state()
        return state

    def reset_eval_hidden(self) -> None:
        super().reset_eval_hidden()
        self.eval_lock_left = {f"agent_{i}": 0 for i in range(self.n_agents)}
        if self.offline_q_network is not None and self.eval_offline_rnn is None:
            self.eval_offline_rnn = self._offline_initial_state()

    def select_actions_eval(
        self,
        observations: Dict[str, np.ndarray],
        legals: Dict[str, np.ndarray],
        eps: float,
        state: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        env_state=None,
    ) -> Dict[str, int]:
        del env_state
        if state is None:
            rnn_state = self.eval_rnn
            last_actions = self.eval_last_actions
            off_rnn = self.eval_offline_rnn
            lock_left = self.eval_lock_left
        else:
            rnn_state = state["rnn"]
            last_actions = state["last"]
            off_rnn = state.get("offline_rnn")
            lock_left = state.setdefault("lock_left", {})

        use_offline = self.offline_q_network is not None
        if use_offline and off_rnn is None:
            off_rnn = self._offline_initial_state()
            if state is None:
                self.eval_offline_rnn = off_rnn
            else:
                state["offline_rnn"] = off_rnn
        if lock_left is None:
            lock_left = {}
            if state is None:
                self.eval_lock_left = lock_left
            else:
                state["lock_left"] = lock_left

        agents = sorted(observations.keys())
        obs_stack = torch.stack(
            [torch.from_numpy(observations[a]).float() for a in agents]
        ).to(self.device)
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_head(obs_stack)).tolist()

        params = self.vckr_params
        epsilon_dec = 0.0 if eps <= 0.0 else eps / self.n_agents
        sequential = bool(getattr(self, "sequential_explore", False)) and eps > 0.0
        explorer = int(np.random.randint(self.n_agents)) if sequential else -1
        lock_h = int(params.lock_horizon)

        actions: Dict[str, int] = {}
        chosen_q: List[float] = []
        offline_chosen_q: List[float] = []
        explored_flags: List[bool] = []
        n_locked = 0
        for i, agent in enumerate(agents):
            obs = torch.from_numpy(observations[agent]).float().to(self.device)
            last_idx = last_actions.get(agent)
            if last_idx is None:
                last_oh = torch.zeros(self.n_actions, device=self.device)
            else:
                last_oh = F.one_hot(
                    torch.tensor(int(last_idx), device=self.device), self.n_actions
                ).float()
            pieces = [obs, last_oh]
            if self.add_agent_id_to_obs:
                pieces.append(
                    F.one_hot(torch.tensor(i, device=self.device), self.n_agents).float()
                )
            obs_in = torch.cat(pieces, dim=-1).unsqueeze(0)
            legal = torch.from_numpy(legals[agent]).bool().to(self.device)
            with torch.no_grad():
                q, rnn_state[agent] = self.q_network(obs_in, rnn_state.get(agent))
                q = q[0].masked_fill(~legal, -1e9)

                q_off = None
                if use_offline:
                    q_off, off_rnn[agent] = self.offline_q_network(
                        obs_in, off_rnn.get(agent)
                    )
                    q_off = q_off[0].masked_fill(~legal, -1e9)

            left = int(lock_left.get(agent, 0))
            if use_offline and left <= 0 and float(gates[i]) >= float(params.gate_threshold):
                left = lock_h
            locked = use_offline and left > 0
            if locked:
                n_locked += 1
                probs = torch.softmax(
                    q_off / max(float(params.exec_policy_temp), 1e-6), dim=-1
                ).clamp(min=1e-9)
                if eps > 0.0:
                    a = int(torch.multinomial(probs, 1).item())
                else:
                    a = int(torch.argmax(q_off).item())
                lock_left[agent] = left - 1
                explored_flags.append(False)
            else:
                lock_left[agent] = 0
                if sequential:
                    explore_here = i == explorer and np.random.rand() < eps
                else:
                    explore_here = np.random.rand() < epsilon_dec
                if explore_here:
                    legal_idx = torch.nonzero(legal, as_tuple=False).squeeze(-1)
                    a = int(legal_idx[torch.randint(legal_idx.numel(), (1,))].item())
                else:
                    a = int(torch.argmax(q).item())
                explored_flags.append(bool(explore_here))

            last_actions[agent] = a
            actions[agent] = a
            if trace is not None:
                chosen_q.append(float(q[a].item()))
                if q_off is not None:
                    offline_chosen_q.append(float(q_off[a].item()))

        if trace is not None:
            trace["chosen_q"] = chosen_q
            trace["n_disagree"] = 0
            trace["n_agents"] = len(chosen_q)
            trace["vckr_locked"] = n_locked
            if offline_chosen_q:
                trace["offline_chosen_q"] = offline_chosen_q
        if state is not None:
            state["explored"] = explored_flags
        return actions
