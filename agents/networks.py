"""Actor-Critic network placeholder.
Tries to provide a PyTorch implementation if torch is available.
Otherwise provides a tiny numpy-based fallback interface used by trainer/tests.
"""

try:
    import torch.nn as nn
    import torch.nn.functional as F

    class ActorCritic(nn.Module):
        def __init__(self, input_dim=5, hidden=64):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.actor = nn.Linear(hidden, 1)  # binary action logits
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            logits = self.actor(x)
            value = self.critic(x)
            return logits, value

        def get_weight_matrix(self):
            """Return first hidden layer weights for better visualization.

            Returns fc1 weights (input -> hidden1) so we can see all 64
            hidden nodes' activity.
            """
            try:
                # 返回第一層權重（5x64），這樣可以看到所有隱藏層節點的變化
                w = self.fc1.weight.detach().cpu().numpy()
                return w.T  # 轉置為 (5, 64) 方便可視化
            except Exception:
                return None

except Exception:
    # Numpy fallback
    import numpy as np

    class ActorCritic:
        def __init__(self, input_dim=5, hidden=64):
            self.input_dim = input_dim
            self.hidden = hidden
            # random weights for deterministic output in fallback
            rng = np.random.RandomState(1)
            self.w = rng.randn(input_dim, hidden) * 0.01
            self.wa = rng.randn(hidden, 1) * 0.01
            self.wv = rng.randn(hidden, 1) * 0.01

        def forward(self, x):
            h = np.tanh(x.dot(self.w))
            logits = h.dot(self.wa)
            value = h.dot(self.wv)
            return logits, value

        def get_weight_matrix(self):
            # return actor weights approximation
            try:
                return self.wa.copy()
            except Exception:
                return None
