"""QMIX learner with SO2's perturbed bootstrap action.

The TD loss, mixer, CQL hooks and target-network interval stay on
``PaperQMIXLearner``. Only the action plugged into the target mixer changes.
``n_upc`` is read by ``training/online.py`` and does not affect other learners.
"""

from __future__ import annotations

import torch

from ovmse.learner import PaperQMIXLearner
from so2.perturb import perturbed_bootstrap_actions


class SO2Learner(PaperQMIXLearner):
    checkpoint_stem = "so2"

    def __init__(
        self,
        *args,
        pvu_sigma: float = 0.3,
        pvu_clip: float = 0.6,
        n_upc: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        sigma = float(pvu_sigma)
        clip = float(pvu_clip)
        updates = int(n_upc)
        if not 0.0 <= sigma <= 1.0:
            raise ValueError(f"so2 sigma must be in [0, 1], got {sigma}")
        if clip < 0.0:
            raise ValueError(f"so2 clip must be >= 0, got {clip}")
        if updates < 1:
            raise ValueError(f"so2 n_upc must be >= 1, got {updates}")
        self.pvu_sigma = sigma
        self.pvu_clip = clip
        self.n_upc = updates
        self._bootstrap_stats: dict = {}
        self._pvu_generator = torch.Generator(device="cpu")
        seed = torch.initial_seed() % (2**31 - 1)
        self._pvu_generator.manual_seed(int(seed) + 2312)

    def _select_bootstrap_actions(
        self, mac_sel: torch.Tensor, avail: torch.Tensor
    ) -> torch.Tensor:
        actions, changed = perturbed_bootstrap_actions(
            mac_sel,
            avail,
            self.pvu_sigma,
            generator=self._pvu_generator,
        )
        self._bootstrap_stats = {
            "so2/perturb_frac": float(changed.float().mean().item()),
            "so2/sigma": float(self.pvu_sigma),
            "so2/clip": float(self.pvu_clip),
            "so2/n_upc": float(self.n_upc),
        }
        return actions
