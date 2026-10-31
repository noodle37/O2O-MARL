# Copyright 2023 InstaDeep Ltd. All rights reserved.

"""Meta-Utility Adaptive Prior Regularization (MU-APR).

The offline policy prior is the same frozen network used by SAPR. The weight on
the KL term is no longer driven by a probe estimate; it is the exponential of a
scalar ``z`` that is updated by a meta-gradient on a held-out validation split of
recent online episodes.
"""

from muapr.controller import (
    MUAPRSettings,
    MetaLambdaController,
    build_muapr_controller,
)
from muapr.meta import MetaGradientEngine

__all__ = [
    "MUAPRSettings",
    "MetaLambdaController",
    "MetaGradientEngine",
    "build_muapr_controller",
]
