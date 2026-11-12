# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""TCPA: Team-Conditioned Prior Adaptation for offline-to-online MARL.

The online phase starts from the CFCQL-D pretrained QMIX network.  Instead of
re-initialising it, the frozen offline utilities stay in place and a learnable
residual is added on top of them, so the online team value starts exactly at
the offline team value and is corrected from then on:

    q_on(tau, a)  = q_off(tau, a) + dq_phi(tau, a)
    Q_on(s, a)    = f_mix^psi(s, q_on)

The offline collaboration prior is retained through a *team-conditioned*
signal.  At every step the frozen offline joint action and the current online
joint action are scored by the same online target mixer, giving the team
conditioned regret

    R_t = Q_on(s_t, a_on*) - Q_on(s_t, a_off) >= 0,

and the retention coefficient ``g_t = sg[exp(-R_t / tau_g)]``.  ``g_t`` scales
an IGM ranking loss that keeps the offline action locally greedy, which -- by
QMIX monotonicity -- keeps the offline joint action globally greedy as well.

The whole update is

    L = L_TD(lambda) + eta_prior * L_IGM,

with the residual network (phi) and the mixing network (psi) as the only
trainable parameters.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

import torch

from ovmse.targets import build_td_lambda_targets
from qmix.core import QMIXBase
from qmix.networks import QMixer
from tcpa.losses import abs_q_retention_loss, igm_ranking_loss, value_anchor_loss
from tcpa.networks import build_residual_q_network
from tcpa.regret import RegretScaleTracker, gather_agent_q, team_regret


class TCPALearner(QMIXBase):
    """Residual QMIX learner with a team-conditioned prior-retention gate."""

    checkpoint_stem = "tcpa"

    VARIANTS = (
        "full",  # residual + dynamic gate + IGM ranking retention
        "no_residual",  # plain fine-tuning of the whole QMIX from the offline weights
        "no_igm",  # residual + TD only, no prior retention
        "fixed_gate",  # g_t == 1 instead of the dynamic team-regret gate
        "abs_q",  # absolute Q preservation instead of relative ordering
        "gate_anti",  # falsification arm: g_t -> 1 - g_t
        # --- attribution arms (see documentation/experiment_reports) ---
        "value_anchor",  # generic trust region to the offline value function
        "placebo",  # same hinge, but the retained action is a random legal one
    )

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        obs_dim: int,
        state_dim: int,
        *,
        variant: str = "full",
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
        eta_prior: float = 0.1,
        eta_anchor: Optional[float] = None,
        tau_g: Any = "auto",
        tau_g_fixed: float = 1.0,
        gate_mode: str = "dynamic",
        warmup_steps: int = 0,
        regret_ema_decay: float = 0.1,
        seed: int = 0,
        coop_mode: str = "default",
        coop_frozen_agents: Optional[Any] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(recurrent_layer_dim)
        self.variant = str(variant).lower()
        if self.variant not in self.VARIANTS:
            raise ValueError(
                f"tcpa.variant must be one of {self.VARIANTS}; got {variant!r}"
            )
        self.use_residual = self.variant != "no_residual"

        self.gamma = float(gamma)
        self.td_lambda = float(td_lambda)
        self.grad_norm_clip = float(grad_norm_clip)
        self.target_update_interval = int(target_update_interval)
        self.batch_size_run = int(batch_size_run)
        self.add_agent_id_to_obs = bool(add_agent_id_to_obs)

        self.eta_prior = float(eta_prior)
        self.eta_anchor = (
            float(eta_prior) if eta_anchor is None else float(eta_anchor)
        )
        self.tau_g_fixed = float(tau_g_fixed)
        self.tau_g_mode = "auto" if str(tau_g).lower() == "auto" else "fixed"
        self.gate_mode = str(gate_mode).lower()
        self.warmup_steps = max(0, int(warmup_steps))
        self.regret_tracker = RegretScaleTracker(decay=float(regret_ema_decay))

        # ------------------------------------------------- cooperation condition
        # "default"      -> every agent plays the online policy (cooperative team)
        # "frozen_offline"-> the listed agents keep playing the frozen offline
        #                    prior while the remaining ones play the online
        #                    policy.  The offline *joint* action then no longer
        #                    describes the current team, which is the condition
        #                    in which the prior should stop being retained.
        self.coop_mode = str(coop_mode).lower()
        if self.coop_mode not in ("default", "frozen_offline"):
            raise ValueError(
                f"tcpa.coop_mode must be 'default' or 'frozen_offline'; got {coop_mode!r}"
            )
        self.coop_frozen_agents = tuple(
            int(a) for a in (coop_frozen_agents or ())
        )
        self.coop_frozen_set = frozenset(self.coop_frozen_agents)

        # Deterministic generator for the placebo arm (random legal action).
        self._placebo_seed = int(seed)
        self._placebo_gen: Optional[torch.Generator] = None

        # Compatibility with the shared offline/online loops: TCPA uses none of
        # the offline-specific terms, its only objective is team-level TD plus
        # the gated prior-retention term.
        self.cql_weight = 0.0
        self.online_cql_weight = 0.0
        self.moderate_lambda = False
        self.use_ovm = False
        self.mixing_ratio = 0.0
        self.support_gate_mode = "off"
        self.support_cql_mode = "off"
        self.sequential_explore = False
        self.n_upc = 1
        # The offline checkpoint carries the Adam state of a different parameter
        # set (plain MarlcRNN + mixer), so it must not be loaded here.
        self.load_offline_optimizer = False

        obs_in = self.obs_dim + self.n_actions
        if self.add_agent_id_to_obs:
            obs_in += self.n_agents

        self.q_network = build_residual_q_network(
            obs_in, self.hidden_dim, self.n_actions
        ).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.mixer = QMixer(
            self.n_agents, state_dim, mixer_embed_dim, mixer_hyper_dim
        ).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)

        self.offline_q_network: Optional[torch.nn.Module] = None
        self.offline_mixer: Optional[QMixer] = None

        params: list[torch.nn.Parameter] = []
        if self.use_residual:
            params += list(self.q_network.residual.parameters())
        else:
            # The ablation fine-tunes the whole offline network instead of
            # learning a residual on top of it.
            self.q_network.freeze_base(False)
            params += list(self.q_network.base.parameters())
        params += list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        self._trainable_params = params

        self.episode_counter = 0
        self.last_target_update_episode = 0
        self.training_step_ctr = 0
        self.t_env = 0
        self.eval_rnn: Dict[str, torch.Tensor] = {}
        self.eval_last_actions: Dict[str, Optional[int]] = {}
        self.eval_offline_rnn: Dict[str, torch.Tensor] = {}
        self.eval_offline_last_actions: Dict[str, Optional[int]] = {}

    # ------------------------------------------------------------ offline setup
    def load_offline_for_ovm(self, path: str) -> None:
        """Install the frozen offline collaboration prior.

        The frozen base carries the CFCQL-D utilities, the online mixer is
        initialised from the offline mixer and stays trainable, and a frozen
        copy of the offline mixer is kept for the offline reference value.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "q_network" not in ckpt:
            raise KeyError(f"offline checkpoint '{path}' has no q_network")
        self.q_network.base.load_state_dict(ckpt["q_network"])
        if "mixer" in ckpt:
            self.mixer.load_state_dict(ckpt["mixer"])

        offline_mixer = copy.deepcopy(self.mixer).to(self.device)
        offline_mixer.eval()
        for param in offline_mixer.parameters():
            param.requires_grad_(False)
        self.offline_mixer = offline_mixer

        if self.use_residual:
            # The base of the online network *is* the frozen offline network,
            # so the prior is read from it instead of a second copy.
            self.q_network.freeze_base(True)
            self.offline_q_network = self.q_network.base
        else:
            reference = copy.deepcopy(self.q_network.base).to(self.device)
            reference.eval()
            for param in reference.parameters():
                param.requires_grad_(False)
            self.offline_q_network = reference

    def prepare_online_start(self) -> None:
        """Check the prior is in place and sync every target network."""
        if self.offline_q_network is None or self.offline_mixer is None:
            raise RuntimeError(
                "TCPA requires the frozen offline prior; enable transfer.load_offline_nets"
            )
        self._sync_targets()

    def _sync_targets(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def load_offline_state(self, payload: Dict[str, Any]) -> None:
        """Restore TCPA-specific state from a checkpoint, if present."""
        if "tcpa_optimizer" in payload:
            self.optimizer.load_state_dict(payload["tcpa_optimizer"])

    def state_dicts(self) -> Dict[str, Any]:
        return {
            "tcpa_variant": self.variant,
            "tcpa_optimizer": self.optimizer.state_dict(),
        }

    # ------------------------------------------------------------------ helpers
    def _online_utilities(
        self, obs_aug: torch.Tensor, resets: torch.Tensor, target: bool
    ) -> torch.Tensor:
        """Online utilities ``[B, T, N, A]`` (base + residual, or base only)."""
        net = self.target_q_network if target else self.q_network
        if self.use_residual:
            residual = net.residual
            return self._unroll(residual, obs_aug, resets)
        return self._unroll(net.base, obs_aug, resets)

    def _offline_utilities(
        self, obs_aug: torch.Tensor, resets: torch.Tensor
    ) -> torch.Tensor:
        """Frozen offline utilities ``[B, T, N, A]`` (no gradient)."""
        if self.offline_q_network is None:
            raise RuntimeError("TCPA has no frozen offline network")
        with torch.no_grad():
            return self._unroll(self.offline_q_network, obs_aug, resets)

    def _random_legal_actions(self, avail: torch.Tensor) -> torch.Tensor:
        """A uniformly random legal action per (step, agent). ``[B, T, N]``.

        Used by the ``placebo`` arm: the hinge keeps its form, its gate and its
        weight, but the action it is asked to retain carries no collaborative
        content.  The generator is seeded from the run seed, so the placebo
        target is reproducible across arms and seeds.
        """
        if self._placebo_gen is None:
            self._placebo_gen = torch.Generator(device=self.device)
            self._placebo_gen.manual_seed(self._placebo_seed + 9973)
        legal = avail.bool()
        scores = torch.rand(
            legal.shape, generator=self._placebo_gen, device=self.device
        )
        scores = scores.masked_fill(~legal, -1.0)
        return torch.argmax(scores, dim=3)

    # ----------------------------------------------------------------- training
    def train_on_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        cql_weight: Optional[float] = None,
        use_ovm: Optional[bool] = None,
        t_env: Optional[int] = None,
        episode_delta: int = 1,
        td_row_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if cql_weight is not None and float(cql_weight) > 0.0:
            raise RuntimeError("TCPA does not use a CQL term; set online_cql_weight=0")
        if use_ovm:
            raise RuntimeError("TCPA does not use the OVM target; set use_ovm=false")
        if t_env is not None:
            self.t_env = int(t_env)

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
        if td_row_mask is not None:
            row = td_row_mask.to(mask.device, mask.dtype).view(-1, 1, 1)
            mask = mask * row

        obs_aug = self._augment_obs(batch["obs"], batch["actions"])
        resets = self._resets_from_batch(batch)

        q_off = self._offline_utilities(obs_aug, resets)
        if self.use_residual:
            dq_on = self._online_utilities(obs_aug, resets, target=False)
            q_on = q_off + dq_on
        else:
            dq_on = torch.zeros_like(q_off)
            q_on = self._online_utilities(obs_aug, resets, target=False)
        with torch.no_grad():
            if self.use_residual:
                q_on_target = q_off + self._online_utilities(
                    obs_aug, resets, target=True
                )
            else:
                q_on_target = self._online_utilities(obs_aug, resets, target=True)

        # ------------------------------------------------------------- TD target
        chosen_q = torch.gather(q_on, 3, actions.long())
        mac_sel = q_on.masked_fill(~avail.bool(), -1e9)
        max_actions = torch.argmax(mac_sel, dim=3, keepdim=True)
        target_max_q = torch.gather(q_on_target, 3, max_actions)

        chosen_tot = self.mixer(chosen_q, states)
        with torch.no_grad():
            target_max_tot = self.target_mixer(target_max_q, states)

        td_targets = build_td_lambda_targets(
            rewards[:, :-1],
            terminated[:, :-1].max(dim=2, keepdim=True)[0],
            mask,
            target_max_tot,
            self.gamma,
            self.td_lambda,
        ).detach()
        td_err = 0.5 * (chosen_tot[:, :-1] - td_targets) ** 2
        loss_td = (td_err * mask).sum() / mask.sum().clamp(min=1.0)

        # ------------------------------------------------- team-conditioned regret
        regret_info = team_regret(
            q_on_target, q_off, avail, states, self.target_mixer
        )
        regret = regret_info["regret"]
        violations = int((regret < -1e-6).sum().item())
        regret = regret.clamp(min=0.0)

        self.regret_tracker.update(regret[:, :-1], mask)
        tau_g = self.regret_tracker.temperature(self.tau_g_mode, self.tau_g_fixed)
        gate = self._gate(regret, tau_g)

        # -------------------------------------------------- prior-retention loss
        prior_loss = torch.zeros((), device=self.device)
        prior_stats: Dict[str, torch.Tensor] = {}
        # Every constraint arm uses its own weight so that the attribution
        # arms can be compared without a hidden rescaling.
        eta_eff = (
            self.eta_anchor if self.variant == "value_anchor" else self.eta_prior
        )
        if self.training_step_ctr < self.warmup_steps:
            eta_eff = 0.0
        if self.variant != "no_igm" and eta_eff > 0.0:
            if self.variant == "value_anchor":
                # Generic trust region towards the offline value function: a
                # constraint that is *not* about any particular action.
                prior_loss, prior_stats = value_anchor_loss(
                    q_on[:, :-1],
                    q_off[:, :-1],
                    avail[:, :-1],
                    mask,
                )
            elif self.variant == "abs_q":
                prior_loss, prior_stats = abs_q_retention_loss(
                    q_on[:, :-1],
                    q_off[:, :-1],
                    regret_info["a_off"][:, :-1],
                    gate[:, :-1],
                    mask,
                    avail[:, :-1],
                )
            elif self.variant == "placebo":
                # Matched negative control: same hinge, same gate (the gate is
                # still computed from the *real* offline joint action), same
                # weight -- only the retained action is a random legal one.
                a_placebo = self._random_legal_actions(avail)
                prior_loss, prior_stats = igm_ranking_loss(
                    q_on[:, :-1],
                    a_placebo[:, :-1],
                    avail[:, :-1],
                    gate[:, :-1],
                    mask,
                )
            else:
                prior_loss, prior_stats = igm_ranking_loss(
                    q_on[:, :-1],
                    regret_info["a_off"][:, :-1],
                    avail[:, :-1],
                    gate[:, :-1],
                    mask,
                )
        loss = loss_td + eta_eff * prior_loss

        # ------------------------------------------------------------------ step
        self.training_step_ctr += 1
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.tensor(0.0, device=self.device)
        if self.grad_norm_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._trainable_params, self.grad_norm_clip
            )
        self.optimizer.step()

        self.episode_counter += int(episode_delta)
        if (
            self.episode_counter - self.last_target_update_episode
        ) / self.target_update_interval >= 1.0:
            self._sync_targets()
            self.last_target_update_episode = self.episode_counter

        stats = self._diagnostics(
            regret=regret[:, :-1],
            gate=gate[:, :-1],
            mask=mask,
            q_off=q_off[:, :-1],
            q_on=q_on[:, :-1],
            a_off=regret_info["a_off"][:, :-1],
            a_on=regret_info["a_on"][:, :-1],
            avail=avail[:, :-1],
            tau_g=tau_g,
            violations=violations,
            prior_stats=prior_stats,
        )
        return {
            "loss": float(loss.detach().item()),
            "td_loss": float(loss_td.detach().item()),
            "tcpa/prior_loss": float(prior_loss.detach().item()),
            "tcpa/eta_prior": float(eta_eff),
            "grad_norm": float(grad_norm.detach().item()),
            "mean_chosen_q_values": float(chosen_tot.detach().mean().item()),
            **stats,
        }

    def _gate(self, regret: torch.Tensor, tau_g: float) -> torch.Tensor:
        """Retention coefficient, with the fixed and falsification arms."""
        if self.variant == "fixed_gate":
            return torch.ones_like(regret)
        gate = torch.exp(-regret.detach() / max(float(tau_g), 1e-6)).detach()
        if self.variant == "gate_anti":
            gate = 1.0 - gate
        return gate

    def _diagnostics(
        self,
        *,
        regret: torch.Tensor,
        gate: torch.Tensor,
        mask: torch.Tensor,
        q_off: torch.Tensor,
        q_on: torch.Tensor,
        a_off: torch.Tensor,
        a_on: torch.Tensor,
        avail: torch.Tensor,
        tau_g: float,
        violations: int,
        prior_stats: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        valid = mask.bool()
        if valid.dim() == 3:
            valid = valid[..., 0]
        if bool(valid.any()):
            reg_valid = regret[..., 0][valid]
            gate_valid = gate[..., 0][valid]
            regret_mean = float(reg_valid.mean().item())
            regret_std = float(reg_valid.std().item()) if reg_valid.numel() > 1 else 0.0
            gate_mean = float(gate_valid.mean().item())
            gate_p10 = float(torch.quantile(gate_valid, 0.1).item())
            gate_frac = float((gate_valid > 0.5).float().mean().item())
        else:
            regret_mean = regret_std = gate_mean = gate_p10 = gate_frac = 0.0

        agent_valid = avail.bool().any(dim=-1) & valid.unsqueeze(-1)
        if bool(agent_valid.any()):
            agree = float(
                ((a_off == a_on).to(torch.float32) * agent_valid.float()).sum().item()
                / agent_valid.float().sum().item()
            )
        else:
            agree = 0.0

        dq_abs = float(q_on.sub(q_off).abs().mean().item())
        stats = {
            "tcpa/regret_mean": regret_mean,
            "tcpa/regret_std": regret_std,
            "tcpa/gate_mean": gate_mean,
            "tcpa/gate_p10": gate_p10,
            "tcpa/frac_gate_gt_half": gate_frac,
            "tcpa/tau_g": float(tau_g),
            "tcpa/offline_action_agreement": agree,
            "tcpa/dq_absmean": dq_abs,
            "tcpa/regret_violations": float(violations),
        }
        for key, value in prior_stats.items():
            stats[f"tcpa/{key}"] = float(value.detach().item())
        return stats
