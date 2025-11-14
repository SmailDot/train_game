"""
DQN Agent for discrete action spaces.
"""

import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Experience Replay Buffer ---
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim=5, action_dim=2):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim=5, action_dim=2, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference

        self.optimizer = None  # Optimizer will be set by the trainer
        self.memory = None  # Replay memory will be set by the trainer

    def act(self, state, epsilon=0.0):
        """
        Returns actions for given state as per current policy.
        epsilon: float, for epsilon-greedy action selection
        """
        if random.random() > epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                s = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                q_values = self.policy_net(s)
                action = q_values.max(1)[1].view(1, 1)
                return action.item(), 0.0, 0.0  # return dummy logp and value
        else:
            return random.randrange(self.action_dim), 0.0, 0.0
