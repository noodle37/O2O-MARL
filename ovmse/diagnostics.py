# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Read-only OVM mechanism diagnostics on a fixed batch of state-action samples.

Nothing here touches the optimizer, the RNG, or the loss: every function runs
under ``torch.no_grad`` and only reads tensors that ``train_on_batch`` already
computes. That is what makes the numbers comparable across runs, branches and
time points - the same fixed sample is re-scored with whatever networks are
live at that moment.

Two entry points:

* ``ovm_fixed_sample_diag`` - full recomputation on an externally supplied batch
  (the frozen sample), used at evaluation points.
* ``ovm_target_stats`` - statistics from tensors that are already in hand, used
  to summarise the *training* batch a step actually consumed.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ovmse.targets import build_td_lambda_targets


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    sel = mask > 0.5
    if sel.sum() == 0:
        return 0.0
    return float(values[sel].mean().item())


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> float:
    sel = mask > 0.5
    if sel.sum() == 0:
        return 0.0
    return float(torch.quantile(values[sel].float(), q).item())


def ovm_target_stats(
    off_tot: torch.Tensor,
    td_targets: torch.Tensor,
    mask_exp: torch.Tensor,
    lam: float,
    *,
    chosen_tot: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """Statistics of the OVM target versus the plain TD target.

    ``off_tot`` is the frozen offline Q_tot on the executed actions, ``td_targets``
    the TD-lambda target, ``mask_exp`` the per-transition loss mask, and ``lam``
    the OVM weight actually in force (``get_lambda_memory(t_env)``).
    """
    mask = mask_exp[:, :-1] if mask_exp.shape[1] == td_targets.shape[1] + 1 else mask_exp
    ovm_targets = torch.maximum(off_tot, td_targets)
    active = (off_tot > td_targets) & (mask > 0.5)
    lift = (ovm_targets - td_targets) * mask

    out: Dict[str, float] = {
        f"{prefix}off_q_mean": _masked_mean(off_tot, mask),
        f"{prefix}off_q_std": float(off_tot[mask > 0.5].std().item())
        if (mask > 0.5).sum() > 1
        else 0.0,
        f"{prefix}td_target_mean": _masked_mean(td_targets, mask),
        f"{prefix}td_target_std": float(td_targets[mask > 0.5].std().item())
        if (mask > 0.5).sum() > 1
        else 0.0,
        f"{prefix}ovm_target_mean": _masked_mean(ovm_targets, mask),
        f"{prefix}ovm_target_std": float(ovm_targets[mask > 0.5].std().item())
        if (mask > 0.5).sum() > 1
        else 0.0,
        f"{prefix}off_minus_td_mean": _masked_mean(off_tot - td_targets, mask),
        f"{prefix}activation_rate": float(active.float().sum().item())
        / max(float((mask > 0.5).sum().item()), 1.0),
        f"{prefix}lift_raw_mean": _masked_mean(lift, mask),
        f"{prefix}lift_active_mean": float(lift[active].mean().item())
        if active.sum() > 0
        else 0.0,
        f"{prefix}lift_active_p50": _masked_quantile(lift, active, 0.5),
        f"{prefix}lift_active_p90": _masked_quantile(lift, active, 0.9),
        f"{prefix}lift_effective_mean": _masked_mean(lam * lift, mask),
        f"{prefix}effective_target_mean": _masked_mean(
            (1.0 - lam) * td_targets + lam * ovm_targets, mask
        ),
        f"{prefix}lambda_memory": float(lam),
        f"{prefix}n_transitions": float((mask > 0.5).sum().item()),
    }
    if chosen_tot is not None:
        chosen = chosen_tot[:, :-1] if chosen_tot.shape[1] == mask.shape[1] + 1 else chosen_tot
        out[f"{prefix}online_q_mean"] = _masked_mean(chosen, mask)
        out[f"{prefix}online_q_std"] = (
            float(chosen[mask > 0.5].std().item()) if (mask > 0.5).sum() > 1 else 0.0
        )
        out[f"{prefix}online_minus_offline_q"] = _masked_mean(chosen - off_tot, mask)
        out[f"{prefix}td_residual_mean"] = _masked_mean(chosen - td_targets, mask)
        out[f"{prefix}ovm_residual_mean"] = _masked_mean(chosen - ovm_targets, mask)
        out[f"{prefix}ovm_minus_td_residual_mean"] = _masked_mean(
            (chosen - ovm_targets) - (chosen - td_targets), mask
        )
    return out


def _agent_greedy(q: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    """Per-agent argmax over legal actions; shape (b, t, n)."""
    masked = q.masked_fill(~avail, -1e9)
    return masked.argmax(dim=-1)


def ovm_fixed_sample_diag(
    learner,
    batch: Dict[str, torch.Tensor],
    t_env: int,
    *,
    prefix: str = "ovm_diag/",
    use_ovm: Optional[bool] = None,
) -> Dict[str, float]:
    """Score one fixed batch with the live networks and report the OVM mechanism.

    Mirrors the forward path of ``PaperQMIXLearner.train_on_batch`` exactly (same
    augmentation, same resets, same TD-lambda target, same frozen offline unroll)
    but never backpropagates. The frozen offline networks must already be loaded
    via ``load_offline_for_ovm``; when they are missing the offline-side entries
    are reported as zeros rather than raising, so an OVM-off arm still produces a
    comparable row.
    """
    ovm_on = learner.use_ovm if use_ovm is None else use_ovm
    if getattr(learner, "offline_q_network", None) is None or getattr(
        learner, "offline_mixer", None
    ) is None:
        ovm_on = False

    actions = batch["actions"]
    avail = batch["avail_actions"]
    states = batch["state"]
    terminated = batch["terminated"]
    filled = batch["filled"]

    term = terminated[:, :-1].float()
    if term.dim() == 3 and term.shape[2] > 1:
        term = term.max(dim=2, keepdim=True)[0]
    mask = filled[:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1.0 - term[:, :-1])
    mask_exp = mask

    with torch.no_grad():
        target_mac_out = learner._forward_qs(batch, target=True)
        mac_out = learner._forward_qs(batch, target=False)

        chosen_q = torch.gather(mac_out, 3, actions.long())
        mac_sel = torch.where(avail, mac_out, torch.tensor(-9999999.0, device=learner.device))
        max_actions = torch.argmax(mac_sel, dim=3, keepdim=True)
        target_max_q = torch.gather(target_mac_out, 3, max_actions)

        chosen_tot, target_max_tot, mixed_rew = learner._mixing(
            chosen_q, target_max_q, states, batch["reward"]
        )

        td_targets = build_td_lambda_targets(
            mixed_rew[:, :-1],
            terminated[:, :-1].max(dim=2, keepdim=True)[0],
            mask_exp,
            target_max_tot,
            learner.gamma,
            learner.td_lambda,
        ).detach()

        lam = float(learner.get_lambda_memory(t_env))

        # One frozen-offline unroll serves both the target comparison and the
        # ranking comparison; recomputing it would double the diagnostic cost.
        off_out = learner._unroll(
            learner.offline_q_network,
            learner._augment_obs(batch["obs"], batch["actions"]),
            learner._resets_from_batch(batch),
        )
        off_chosen = torch.gather(off_out[:, :-1], 3, actions[:, :-1].long())
        off_tot = learner.offline_mixer(off_chosen, states[:, :-1])
        stats = ovm_target_stats(
            off_tot, td_targets, mask_exp, lam, chosen_tot=chosen_tot, prefix=prefix
        )
        # The counterfactual is reported for both arms so they stay comparable;
        # ovm_applied records whether the live run actually consumed it.
        stats[f"{prefix}ovm_applied"] = 1.0 if ovm_on else 0.0

        # Ranking: online greedy vs frozen offline greedy on the same states.
        online_greedy = _agent_greedy(mac_out[:, :-1], avail[:, :-1])
        offline_greedy = _agent_greedy(off_out[:, :-1], avail[:, :-1])
        # mask_exp keeps the trailing singleton dim (b, t-1, 1); drop it so the
        # per-step mask lines up with the per-agent argmax (b, t-1, n).
        valid = (mask_exp > 0.5).squeeze(-1)
        valid_a = valid.unsqueeze(-1).expand_as(online_greedy)
        same = (online_greedy == offline_greedy) & valid_a
        stats[f"{prefix}greedy_agent_agree_rate"] = (
            float(same.float().sum().item()) / max(float(valid_a.sum().item()), 1.0)
        )
        per_step_all_agents = ((online_greedy == offline_greedy) | ~valid_a).all(dim=-1)
        decided = valid.any(dim=-1)
        stats[f"{prefix}greedy_joint_agree_rate"] = (
            float(per_step_all_agents[decided].float().mean().item())
            if decided.any()
            else 0.0
        )

        n_episodes = int(batch["obs"].shape[0])
        stats[f"{prefix}n_episodes"] = float(n_episodes)
    return stats
