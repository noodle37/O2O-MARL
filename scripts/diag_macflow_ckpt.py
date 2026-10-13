"""Diagnostics for a MAC-Flow checkpoint against the og_marl vault dataset."""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from data.vault_dataset import load_episode_dataset_from_vault  # noqa: E402
from macflow.flow import euler_integrate
from macflow.networks import add_agent_id

torch.manual_seed(0)
np.random.seed(0)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/macflow_2s3z/mac_flow_seed1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ds = load_episode_dataset_from_vault("og_marl", "smac_v1", "2s3z", "Medium", device=DEVICE)

from macflow.learner import MACFlowLearner  # noqa: E402
from training.factory import build_learner  # noqa: E402

cfg = OmegaConf.to_container(OmegaConf.load("configs/macflow_2s3z.yaml"), resolve=True)
cfg["system"]["device"] = DEVICE
learner = build_learner(cfg, ds)
payload = torch.load(CKPT, map_location=learner.device, weights_only=False)
learner.load_state(payload)
print("loaded checkpoint:", CKPT)

rng = np.random.default_rng(7)
batch = ds.sample(32, rng, device=learner.device)
wb = learner._window_batch(batch)
pair = add_agent_id(wb["obs"])
b, w, n = pair.shape[:3]
pair_f = pair[:, :-1].reshape(-1, learner.pair_dim)
act_f = wb["actions"][:, :-1, :, 0].long().reshape(-1)
mask = (wb["filled"][:, :-1] > 0.5)[:, :, None].expand(-1, -1, n).reshape(-1)
pair_v = pair_f[mask]
act_v = act_f[mask]
avail_v = wb["avail"][:, :-1].reshape(-1, learner.n_actions)[mask]
print("eval transitions:", int(mask.sum()))

with torch.no_grad():
    # 1) flow endpoint accuracy from z=0 and z~N(0, I)
    for tag, z in [
        ("z=0", torch.zeros(pair_v.shape[0], learner.per_agent_dim, device=learner.device)),
        ("z~N", torch.randn(pair_v.shape[0], learner.per_agent_dim, device=learner.device)),
    ]:
        endpoint = euler_integrate(learner.flow, pair_v, z, learner.flow_steps)
        pred = endpoint.argmax(dim=-1)
        acc = (pred == act_v).float().mean().item()
        per_agent = (pred == act_v).float().reshape(-1, n).mean(0)
        legal = avail_v.gather(1, pred.unsqueeze(1)).float().mean().item()
        print(
            f"flow BC argmax acc ({tag}): {acc:.3f}  per-agent: "
            f"{[round(x, 3) for x in per_agent.tolist()]}  raw legal rate: {legal:.3f}"
        )

    # 2) one-step policy (z=0) agreement with dataset and with the flow teacher
    z0 = torch.zeros(pair_v.shape[0], learner.per_agent_dim, device=learner.device)
    logits = learner.policy(pair_v, z0)
    pol_act = logits.argmax(dim=-1)
    print("policy argmax == dataset acc:", round((pol_act == act_v).float().mean().item(), 3))
    endpoint = euler_integrate(learner.flow, pair_v, z0, learner.flow_steps)
    flow_act = endpoint.argmax(dim=-1)
    print("policy argmax == flow(z=0) argmax acc:", round((pol_act == flow_act).float().mean().item(), 3))

    # 3) mixed Q (VDN sum) at dataset actions vs at policy actions
    a_data = F.one_hot(act_v, learner.per_agent_dim).float()
    q_data = learner.critic(pair_v, a_data).reshape(-1, n).sum(-1).mean().item()
    a_pol = F.one_hot(pol_act, learner.per_agent_dim).float()
    q_pol = learner.critic(pair_v, a_pol).reshape(-1, n).sum(-1).mean().item()
    print(f"mixed Q(dataset actions): {q_data:.3f}   mixed Q(policy actions): {q_pol:.3f}")
    print("policy logit stats: mean {:.3f} std {:.3f} min {:.3f} max {:.3f}".format(
        logits.mean().item(), logits.std().item(), logits.min().item(), logits.max().item()))

    # 4) dataset behaviour stats: majority-class baseline and marginal entropy
    maj = []
    probs = []
    for i in range(n):
        c = Counter(act_v.reshape(-1, n)[:, i].tolist())
        maj.append(c.most_common(1)[0][1] / act_v.reshape(-1, n).shape[0])
        p = np.array([v for _, v in sorted(c.items())]) / act_v.reshape(-1, n).shape[0]
        probs.append(p)
    ent = [float(-(p * np.log(p + 1e-9)).sum()) for p in probs]
    print("majority-class acc per agent:", [round(x, 3) for x in maj],
          "mean:", round(float(np.mean(maj)), 3))
    print("per-agent marginal entropy (nats):", [round(e, 3) for e in ent])
