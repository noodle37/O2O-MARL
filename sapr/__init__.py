# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""SAPR-MARL: smooth adaptive prior regularization for O2O MARL.

Part I freezes the offline Q-policy and adds a KL regularizer towards it;
Part II estimates the prior's average team value with random local
interventions; Part III maps that estimate to a bounded, smoothly updated KL
weight.
"""

from sapr.controller import SAPRSettings, SmoothAdaptiveController, build_sapr_controller
from sapr.probe import ProbeRecord, estimate_prior_utility, probe_prior_utility
from sapr.regularizer import SAPRPriorMixin, masked_policy, prior_kl

__all__ = [
    "SAPRSettings",
    "SmoothAdaptiveController",
    "build_sapr_controller",
    "ProbeRecord",
    "estimate_prior_utility",
    "probe_prior_utility",
    "SAPRPriorMixin",
    "masked_policy",
    "prior_kl",
]
