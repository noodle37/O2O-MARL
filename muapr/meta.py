# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Differentiable inner rollout and meta-gradient for MU-APR.

At a meta boundary the engine snapshots the live Q network + mixer, takes K
Adam steps on mixed-history batches, and evaluates the validation objective on
the resulting parameters. The inner steps use the live optimizer's Adam state
and are fully differentiable, so

    m_k = -d J_val(theta_{k+K}(z)) / dz + kappa * z

is exact.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.func import functional_call

from muapr.controller import MUAPRSettings
from muapr.history import build_mixed_batch, mixed_sizes, split_recent_episodes
from muapr.objective import mc_team_returns, validation_objective
from ovmse.targets import build_td_lambda_targets
from qmix.tensor_utils import (
    expand_batch_and_agent_dim_of_time_major_sequence,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
)
from sapr.regularizer import prior_kl
from training.rollout import stack_episodes

ParamKey = Tuple[str, str]


def _step_mask(filled: torch.Tensor) -> torch.Tensor:
    out = filled[:, :-1] if filled.shape[1] > 1 else filled
    if out.dim() == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    return out


class MetaGradientEngine:
    """Stateless meta step; the controller owns z and its Adam state."""

    def __init__(self, learner, settings: MUAPRSettings, offline_data):
        self.learner = learner
        self.settings = settings
        self.offline_data = offline_data
        self.device = learner.device
        self.max_seq_length = int(offline_data.stats.max_seq_length)
        param = next(learner.q_network.parameters())
        self._dtype = param.dtype
        self.gamma = float(settings.gamma)
        self.td_lambda = float(learner.td_lambda)

    # ------------------------------------------------------------------ params

    def _snapshot_params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        q = {
            name: value.detach().clone().requires_grad_(True)
            for name, value in self.learner.q_network.named_parameters()
        }
        mixer = {
            name: value.detach().clone().requires_grad_(True)
            for name, value in self.learner.mixer.named_parameters()
        }
        return {"q_network": q, "mixer": mixer}

    @staticmethod
    def _detached_copy(
        params: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            module: {name: value.detach().clone() for name, value in group.items()}
            for module, group in params.items()
        }

    @staticmethod
    def _flat(
        params: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[List[ParamKey], List[torch.Tensor]]:
        keys: List[ParamKey] = []
        tensors: List[torch.Tensor] = []
        for module in ("q_network", "mixer"):
            for name, value in params[module].items():
                keys.append((module, name))
                tensors.append(value)
        return keys, tensors

    def _snapshot_optimizer(
        self, keys: Sequence[ParamKey]
    ) -> Dict[ParamKey, Dict[str, Any]]:
        live: Dict[ParamKey, torch.nn.Parameter] = {}
        for name, value in self.learner.q_network.named_parameters():
            live[("q_network", name)] = value
        for name, value in self.learner.mixer.named_parameters():
            live[("mixer", name)] = value

        opt = self.learner.optimizer
        group_of = {}
        for group in opt.param_groups:
            for param in group["params"]:
                group_of[param] = group
        out: Dict[ParamKey, Dict[str, Any]] = {}
        for key in keys:
            param = live[key]
            group = group_of.get(param, opt.param_groups[0])
            state = opt.state.get(param, {})
            step = state.get("step", 0)
            if torch.is_tensor(step):
                step = int(step.item())
            out[key] = {
                "m": state.get("exp_avg", torch.zeros_like(param)).detach().to(param.dtype).clone(),
                "v": state.get("exp_avg_sq", torch.zeros_like(param)).detach().to(param.dtype).clone(),
                "step": int(step),
                "lr": float(group["lr"]),
                "betas": tuple(group.get("betas", (0.9, 0.999))),
                "eps": float(group.get("eps", 1e-8)),
            }
        return out

    # ------------------------------------------------------------- functional

    def _unroll_functional(
        self,
        params: Dict[str, torch.Tensor],
        obs: torch.Tensor,
        resets: torch.Tensor,
    ) -> torch.Tensor:
        net = self.learner.q_network
        b, t, n, _ = obs.shape
        resets = resets.expand(b, t, n)
        obs_tm = merge_batch_and_agent_dim_of_time_major_sequence(
            switch_two_leading_dims(obs)
        )
        resets_tm = merge_batch_and_agent_dim_of_time_major_sequence(
            switch_two_leading_dims(resets)
        )
        time_steps, batch_size = obs_tm.shape[:2]
        hidden = net.initial_state(batch_size, obs.device, dtype=obs_tm.dtype)
        outputs = []
        for i in range(time_steps):
            reset_mask = resets_tm[i].unsqueeze(0).unsqueeze(-1)
            hidden = torch.where(
                reset_mask,
                net.initial_state(batch_size, obs.device, dtype=obs_tm.dtype),
                hidden,
            )
            out, hidden = functional_call(net, params, (obs_tm[i], hidden))
            outputs.append(out)
        qs = torch.stack(outputs, dim=0)
        return switch_two_leading_dims(
            expand_batch_and_agent_dim_of_time_major_sequence(qs, b, n)
        )

    def _functional_qs(
        self, params: Dict[str, Dict[str, torch.Tensor]], batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        obs = self.learner._augment_obs(batch["obs"], batch["actions"])
        resets = self.learner._resets_from_batch(batch)
        return self._unroll_functional(params["q_network"], obs, resets)

    def _functional_mix(
        self,
        params: Dict[str, Dict[str, torch.Tensor]],
        chosen_q: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        return functional_call(self.learner.mixer, params["mixer"], (chosen_q, states))

    # --------------------------------------------------------------- inner step

    def _inner_loss(
        self,
        params: Dict[str, Dict[str, torch.Tensor]],
        batch: Dict[str, torch.Tensor],
        row_mask: torch.Tensor,
        lam: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        learner = self.learner
        actions = batch["actions"]
        avail = batch["avail_actions"]
        states = batch["state"]
        rewards = batch["reward"]
        terminated = batch["terminated"]
        filled = batch["filled"]

        term = terminated[:, :-1].float()
        if term.dim() == 3 and term.shape[2] > 1:
            term = term.max(dim=2, keepdim=True)[0]
        mask3 = filled[:, :-1].float()
        mask3[:, 1:] = mask3[:, 1:] * (1.0 - term[:, :-1])
        row = row_mask.to(mask3.device, mask3.dtype).view(-1, 1, 1)
        td_mask = mask3 * row

        with torch.no_grad():
            target_mac = learner._forward_qs(batch, target=True)
        mac = self._functional_qs(params, batch)
        chosen = torch.gather(mac, 3, actions.long())
        mac_sel = torch.where(avail, mac, torch.tensor(-9999999.0, device=self.device))
        max_actions = torch.argmax(mac_sel, dim=3, keepdim=True)
        target_max = torch.gather(target_mac, 3, max_actions)
        chosen_tot = self._functional_mix(params, chosen, states)
        with torch.no_grad():
            target_tot = learner.target_mixer(target_max, states)
        td_targets = build_td_lambda_targets(
            rewards[:, :-1],
            terminated[:, :-1].max(dim=2, keepdim=True)[0],
            mask3,
            target_tot,
            self.gamma,
            self.td_lambda,
        ).detach()
        td_err = 0.5 * (chosen_tot[:, :-1] - td_targets) ** 2
        td_loss = (td_err * td_mask).sum() / td_mask.sum().clamp(min=1.0)

        with torch.no_grad():
            off = learner._offline_unroll(batch)
        kl_mask = filled[:, :-1].float()
        if kl_mask.dim() == 3:
            kl_mask = kl_mask[..., 0]
        kl = prior_kl(
            mac[:, :-1],
            off[:, :-1],
            avail[:, :-1].bool(),
            kl_mask,
            self.settings.temperature,
            self.settings.support_floor,
        )
        loss = td_loss + lam * kl
        stats = {
            "muapr/inner_td_loss": float(td_loss.detach().item()),
            "muapr/inner_kl": float(kl.detach().item()),
            "muapr/inner_kl_weighted": float((lam * kl).detach().item()),
        }
        return loss, stats

    def _adam_step(
        self,
        params: Dict[str, Dict[str, torch.Tensor]],
        keys: Sequence[ParamKey],
        grads: Sequence[Optional[torch.Tensor]],
        opt_state: Dict[ParamKey, Dict[str, Any]],
    ) -> float:
        clip = float(self.settings.inner_grad_clip)
        norm_sq = 0.0
        for grad in grads:
            if grad is not None:
                norm_sq += float(grad.detach().pow(2).sum().item())
        norm = math.sqrt(norm_sq)
        scale = 1.0 if clip <= 0.0 or norm <= clip else clip / (norm + 1e-12)

        for key, grad in zip(keys, grads):
            module, name = key
            param = params[module][name]
            if grad is None:
                grad = torch.zeros_like(param)
            if scale != 1.0:
                grad = grad * scale
            state = opt_state[key]
            b1, b2 = state["betas"]
            state["m"] = b1 * state["m"] + (1.0 - b1) * grad
            state["v"] = b2 * state["v"] + (1.0 - b2) * grad * grad
            state["step"] += 1
            step = state["step"]
            m_hat = state["m"] / (1.0 - b1**step)
            v_hat = state["v"] / (1.0 - b2**step)
            # sqrt at exactly zero has an infinite derivative, which poisons the
            # second-order graph for parameters whose first-step gradient is zero
            # (e.g. Q-head entries of unselected actions). The clamp is a smooth
            # numerical guard; for v >> eps it matches the live Adam update.
            denom = torch.sqrt(v_hat.clamp_min(state["eps"]))
            params[module][name] = param - state["lr"] * m_hat / (
                denom + state["eps"]
            )
        return norm

    # ------------------------------------------------------------------- meta

    def run_inner_and_meta(
        self,
        z_value: float,
        inner_batches: Sequence[Tuple[Dict[str, torch.Tensor], torch.Tensor]],
        val_batch: Dict[str, torch.Tensor],
        *,
        create_graph: bool,
    ) -> Dict[str, Any]:
        """Run K inner steps and evaluate M(z).

        ``create_graph=False`` is the plain value path used by finite
        differences; ``meta_grad`` is then ``None``.
        """
        s = self.settings
        params = self._snapshot_params()
        keys, tensors = self._flat(params)
        opt_state = self._snapshot_optimizer(keys)
        params_ref = self._detached_copy(params)
        initial = [value.detach().clone() for value in tensors]

        z = torch.tensor(
            float(z_value), dtype=self._dtype, device=self.device, requires_grad=True
        )
        lam = s.lambda_ref * torch.exp(z.clamp(-s.z_clip, s.z_clip))

        inner_rows: List[Dict[str, float]] = []
        losses: List[float] = []
        for index, (batch, row_mask) in enumerate(inner_batches):
            loss, stats = self._inner_loss(params, batch, row_mask, lam)
            # Differentiate w.r.t. the *current* fast weights. Differentiating
            # w.r.t. the initial leaves would give the total derivative through
            # earlier Adam steps instead of the gradient the inner optimizer
            # actually consumes, and its value would depend on create_graph.
            current = [params[module][name] for module, name in keys]
            grads = torch.autograd.grad(
                loss,
                current,
                create_graph=create_graph,
                allow_unused=True,
            )
            norm = self._adam_step(params, keys, grads, opt_state)
            losses.append(float(loss.detach().item()))
            inner_rows.append(
                {
                    "muapr/inner_step_index": float(index),
                    "muapr/inner_step_loss": float(loss.detach().item()),
                    "muapr/inner_step_td": stats["muapr/inner_td_loss"],
                    "muapr/inner_step_kl": stats["muapr/inner_kl"],
                    "muapr/inner_step_kl_weighted": stats["muapr/inner_kl_weighted"],
                    "muapr/inner_step_grad_norm": float(norm),
                    "muapr/inner_step_z": float(z_value),
                    "muapr/inner_step_lambda": float(lam.detach().item()),
                }
            )

        with torch.no_grad():
            q_ref = self._functional_qs(params_ref, val_batch)
            returns = mc_team_returns(
                val_batch["reward"],
                val_batch["terminated"],
                val_batch["filled"],
                self.gamma,
            )
        q_final = self._functional_qs(params, val_batch)
        actions = val_batch["actions"]
        # The advantage baseline is a stop-gradient constant (PPO-style), so it
        # must come from the frozen meta-start reference. Evaluating it at the
        # updated parameters would leak its value change into finite differences
        # without contributing a gradient.
        q_ref_chosen = torch.gather(q_ref, 3, actions.long())
        q_tot = self._functional_mix(params_ref, q_ref_chosen, val_batch["state"])
        J, val_stats = validation_objective(
            q_final,
            q_ref,
            q_tot,
            val_batch["avail_actions"],
            actions,
            returns,
            val_batch["filled"],
            temperature=s.temperature,
            support_floor=s.support_floor,
            ratio_clip=s.ratio_clip,
            advantage_normalize=s.advantage_normalize,
        )

        kappa_penalty = 0.5 * s.kappa * z * z
        meta_loss = -J + kappa_penalty
        meta_grad = None
        if create_graph:
            meta_grad = torch.autograd.grad(meta_loss, z, allow_unused=False)[0]

        flat_final = [params[module][name] for module, name in keys]
        drift_sq = sum(
            float((final.detach() - start).pow(2).sum().item())
            for final, start in zip(flat_final, initial)
        )
        base_sq = sum(float(start.pow(2).sum().item()) for start in initial)
        stats: Dict[str, Any] = dict(val_stats)
        stats.update(
            {
                "muapr/inner_steps": float(len(inner_batches)),
                "muapr/inner_batch_size": float(s.inner_batch_size),
                "muapr/inner_loss_first": losses[0] if losses else 0.0,
                "muapr/inner_loss_last": losses[-1] if losses else 0.0,
                "muapr/inner_loss_mean": sum(losses) / len(losses) if losses else 0.0,
                "muapr/inner_td_loss_last": inner_rows[-1]["muapr/inner_step_td"] if inner_rows else 0.0,
                "muapr/inner_kl_last": inner_rows[-1]["muapr/inner_step_kl"] if inner_rows else 0.0,
                "muapr/inner_kl_weighted_last": (
                    inner_rows[-1]["muapr/inner_step_kl_weighted"] if inner_rows else 0.0
                ),
                "muapr/inner_grad_norm_mean": (
                    sum(row["muapr/inner_step_grad_norm"] for row in inner_rows) / len(inner_rows)
                    if inner_rows
                    else 0.0
                ),
                "muapr/inner_grad_norm_max": (
                    max(row["muapr/inner_step_grad_norm"] for row in inner_rows)
                    if inner_rows
                    else 0.0
                ),
                "muapr/param_drift": math.sqrt(drift_sq),
                "muapr/param_drift_rel": math.sqrt(drift_sq) / (math.sqrt(base_sq) + 1e-12),
                "muapr/meta_loss": float(meta_loss.detach().item()),
                "muapr/kappa_penalty": float(kappa_penalty.detach().item()),
                "muapr/J_val": float(J.detach().item()),
            }
        )
        return {
            "z_value": float(z_value),
            "lambda": float(lam.detach().item()),
            "meta_grad": None if meta_grad is None else float(meta_grad.detach().item()),
            "meta_loss": float(meta_loss.detach().item()),
            "kappa_penalty": float(kappa_penalty.detach().item()),
            "J_val": float(J.detach().item()),
            "stats": stats,
            "inner_rows": inner_rows,
        }

    def compute_meta_gradient(
        self,
        online_buffer: Sequence[Dict[str, Any]],
        rng: np.random.Generator,
        t_env: int,
        z_value: float,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        s = self.settings
        split = split_recent_episodes(
            online_buffer, s.val_recent_episodes, s.val_episodes, rng
        )
        if split is None:
            return self._invalid("insufficient_buffer")
        off_bs, on_bs = mixed_sizes(s.inner_batch_size, s.kl_offline_frac)
        if len(split["inner"]) < on_bs:
            return self._invalid("inner_pool_too_small")

        val_batch = stack_episodes(split["validation"], self.max_seq_length, self.device)
        inner_batches = []
        for _ in range(s.inner_steps):
            batch, row_mask, _, _ = build_mixed_batch(
                self.offline_data,
                split["inner"],
                rng,
                batch_size=s.inner_batch_size,
                offline_frac=s.kl_offline_frac,
                max_seq_length=self.max_seq_length,
                device=self.device,
            )
            inner_batches.append((batch, row_mask))

        result = self.run_inner_and_meta(
            z_value, inner_batches, val_batch, create_graph=True
        )
        if result["meta_grad"] is None or not math.isfinite(result["meta_grad"]):
            return self._invalid("invalid_meta_grad")
        if float(result["stats"].get("muapr/valid_steps", 0.0)) <= 0.0:
            return self._invalid("no_validation_steps")
        result.update(
            {
                "valid": True,
                "reason": "",
                "val_episodes": len(split["validation"]),
                "inner_episodes": len(split["inner"]),
                "offline_bs": off_bs,
                "online_bs": on_bs,
                "engine_time_s": time.perf_counter() - t_start,
            }
        )
        return result

    @staticmethod
    def _invalid(reason: str) -> Dict[str, Any]:
        return {
            "valid": False,
            "reason": reason,
            "meta_grad": 0.0,
            "meta_loss": 0.0,
            "kappa_penalty": 0.0,
            "J_val": 0.0,
            "stats": {},
            "inner_rows": [],
            "engine_time_s": 0.0,
        }
