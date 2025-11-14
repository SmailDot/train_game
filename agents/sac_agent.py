"""
Soft Actor-Critic (SAC) Agent for discrete action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    """Policy network for discrete SAC."""

    def __init__(self, state_dim=5, action_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        logits = self.net(state)
        return logits

    def get_action_probs(self, state):
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    """Double Q-network for discrete SAC."""

    def __init__(self, state_dim=5, action_dim=2, hidden_dim=256):
        super().__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        q1 = self.q1(state)
        q2 = self.q2(state)
        return q1, q2


class SACAgent:
    def __init__(self, state_dim=5, action_dim=2, device=None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor_optimizer = None
        self.critic_optimizer = None
        self.memory = None  # Replay memory will be set by the trainer

    def act(self, state, explore=True):
        """
        Returns an action for the given state.
        If explore is True, samples from the policy distribution.
        Otherwise, returns the most likely action.
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor.get_action_probs(state_t)

        dist = Categorical(action_probs)
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs, dim=1)

        # Return dummy logp and value for compatibility with the UI
        return action.item(), 0.0, 0.0
