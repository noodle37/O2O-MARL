"""VCKR: validity-constrained sub-team knowledge reuse for online MARL fine-tuning.

The three pieces mirror the algorithm spec:
  knowledge.py  offline sub-team units (coupling graph, start domain, return memory)
  validity.py   counterfactual validity of a unit under the current team context
  learner.py    validity-gated value target plus decentralized reuse/exploration
"""

from vckr.knowledge import SubteamKnowledge, VCKRParams, build_subteam_knowledge
from vckr.learner import VCKRLearner

__all__ = [
    "SubteamKnowledge",
    "VCKRParams",
    "VCKRLearner",
    "build_subteam_knowledge",
]
