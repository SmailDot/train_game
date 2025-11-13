"""Simple PPO-like agent wrapper. If torch is installed it can be used; otherwise
it implements a RandomAgent fallback with the same interface for smoke tests.
"""
from typing import Tuple

try:
    import torch
    import torch.nn.functional as F
    from .networks import ActorCritic

    class PPOAgent:
        def __init__(self, lr=3e-4, device=None):
            self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            self.net = ActorCritic().to(self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        def act(self, state):
            # state: numpy array
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(s)
            prob_t = torch.sigmoid(logits).squeeze(0)  # tensor
            prob = prob_t.detach().cpu().numpy()[0] if prob_t.dim() else prob_t.detach().cpu().item()
            action = int((prob > 0.5))
            # compute log-prob using tensors to avoid mixing types
            if action:
                logp_t = torch.log(prob_t + 1e-8)
            else:
                logp_t = torch.log(1 - prob_t + 1e-8)
            logp = float(logp_t.detach().cpu().item())
            return action, logp, float(value.detach().cpu().numpy()[0, 0])

        # A full update implementation is omitted here (heavy). Trainer will use act() for smoke tests.

except Exception:
    import numpy as np

    class RandomAgent:
        def __init__(self, seed=1):
            self.rng = np.random.RandomState(seed)

        def act(self, state) -> Tuple[int, float, float]:
            a = int(self.rng.rand() > 0.9)
            return a, 0.0, 0.0

    PPOAgent = RandomAgent
