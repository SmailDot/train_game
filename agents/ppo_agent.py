"""Simple PPO-like agent wrapper. If torch is installed it can be used; otherwise
it implements a RandomAgent fallback with the same interface for smoke tests.
"""

from typing import Tuple

try:
    import torch

    from .networks import ActorCritic

    class PPOAgent:
        def __init__(self, lr=3e-4, device=None):
            self.device = device or (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.net = ActorCritic().to(self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        def act(self, state, explore: bool = False):
            """Return an action for the provided state.

            By default the policy behaves deterministically (threshold at 0.5)
            so the on-screen agent produces stable trajectories. Set
            ``explore=True`` to sample from the Bernoulli distribution instead
            (useful for debugging or custom rollouts).
            """

            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            logits, value = self.net(s)
            prob = torch.sigmoid(logits)
            prob = prob.clamp(min=1e-6, max=1 - 1e-6)

            if explore:
                dist = torch.distributions.Bernoulli(probs=prob)
                action_t = dist.sample()
                logp_t = dist.log_prob(action_t)
            else:
                action_val = (prob >= 0.5).float()
                action_t = action_val
                # log probability for the chosen deterministic action
                logp_t = torch.where(
                    action_val > 0.5,
                    torch.log(prob),
                    torch.log(1.0 - prob),
                )

            action = int(action_t.item())
            logp = float(logp_t.detach().cpu().item())
            value_out = float(value.detach().cpu().item())
            return action, logp, value_out

    # Full update not implemented here (omitted for brevity).

except Exception:
    import numpy as np

    class RandomAgent:
        def __init__(self, seed=1):
            self.rng = np.random.RandomState(seed)

        def act(self, state, explore: bool = False) -> Tuple[int, float, float]:
            a = int(self.rng.rand() > 0.9)
            return a, 0.0, 0.0

    PPOAgent = RandomAgent
