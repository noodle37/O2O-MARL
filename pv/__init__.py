# Copyright 2024 InstaDeep Ltd. All rights reserved.

"""Problem-validation shared package.

Contents:
  * behaviour  -- autoregressive joint behaviour cloning (the data behaviour model
                  behind the teammate-deviation measure D), plus D_i / D scoring.
  * exp1_metrics -- Drop, censored recovery steps (Kaplan-Meier), WR@1M, paired
                  bootstrap differences, Holm correction.
  * exp2_shift  -- controlled teammate-shift evaluation: per-agent weight
                  transplantation, shadow offline Q_tot forward, natural (N) and
                  random (R) shifts, g_i and the 2x2 interaction term I_ij.
  * probe       -- per-state records logged during arm C's training (experiment 2b).
"""

from __future__ import annotations

__all__ = ["behaviour", "exp1_metrics", "exp2_shift", "probe"]
