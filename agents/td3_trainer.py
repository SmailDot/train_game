from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise ImportError("td3_trainer requires PyTorch to be installed.") from exc

from agents.replay_buffer import ReplayBuffer
from game.environment import GameEnv


class TD3Actor(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))

    def get_weight_matrix(self):  # pragma: no cover
        try:
            return self.out.weight.detach().cpu().numpy()
        except Exception:
            return None


class TD3Critic(nn.Module):
    def __init__(self, input_dim: int = 5, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class TD3Agent:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.actor = TD3Actor().to(self.device)

    def attach(self, trainer: "TD3Trainer") -> None:
        self.device = trainer.device
        self.actor = trainer.actor

    def act(self, state, explore: bool = False) -> Tuple[int, float, float]:
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).clamp(-1.0, 1.0)
        if explore:
            noise = torch.randn_like(action) * 0.1
            action = (action + noise).clamp(-1.0, 1.0)
        continuous = float(action.item())
        discrete = 1 if continuous > 0 else 0
        logp = float(-abs(continuous))
        value = float(continuous)
        return discrete, logp, value


@dataclass
class TD3Config:
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    min_buffer: int = 5_000


class TD3Trainer:
    def __init__(self, config: Optional[TD3Config] = None):
        self.config = config or TD3Config()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.actor = TD3Actor().to(self.device)
        self.actor_target = TD3Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = TD3Critic().to(self.device)
        self.critic2 = TD3Critic().to(self.device)
        self.critic1_target = TD3Critic().to(self.device)
        self.critic2_target = TD3Critic().to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr)
        critic_params = list(self.critic1.parameters()) + list(
            self.critic2.parameters()
        )
        self.critic_opt = torch.optim.Adam(critic_params, lr=self.config.lr)

        self.replay = ReplayBuffer(self.config.buffer_size)
        self.agent = TD3Agent(self.device)
        self.agent.attach(self)
        self._step = 0

    def build_agent(self) -> TD3Agent:
        agent = TD3Agent(self.device)
        agent.attach(self)
        return agent

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.copy_(
                tgt_param.data * (1.0 - self.config.tau)
                + src_param.data * self.config.tau
            )

    def _update(self) -> Dict[str, float]:
        batch = self.replay.sample(self.config.batch_size)
        states = torch.tensor(
            np.array([b["state"] for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            np.array([b["continuous_action"] for b in batch]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.tensor(
            np.array([b["reward"] for b in batch]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.array([b["next_state"] for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            np.array([b["done"] for b in batch], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target_q1 = self.critic1_target(next_states, next_action)
            target_q2 = self.critic2_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + self.config.gamma * (1.0 - dones) * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        entropy = 0.0
        if self._step % self.config.policy_delay == 0:
            actor_action = self.actor(states)
            actor_loss = -self.critic1(states, actor_action).mean()
            entropy = float(torch.mean(1 - actor_action.pow(2)).item())

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)

        return {
            "loss": float((actor_loss + critic_loss).item()),
            "policy_loss": float(actor_loss.item()),
            "value_loss": float(critic_loss.item()),
            "entropy": float(entropy),
        }

    def save(self, step: int) -> str:
        path = f"checkpoints/td3_{step}.pt"
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            path,
        )
        return path

    def train(
        self,
        total_timesteps: Optional[int] = None,
        env: Optional[GameEnv] = None,
        envs: Optional[Callable[[], GameEnv]] = None,
        metrics_callback: Optional[Callable[[Dict], None]] = None,
        stop_event: Optional[object] = None,
        initial_iteration: int = 0,
    ) -> None:
        environment = env or GameEnv()
        state = environment.reset()
        timesteps = 0
        iteration = initial_iteration
        episode_reward = 0.0

        while True:
            if (
                stop_event is not None
                and getattr(stop_event, "is_set", lambda: False)()
            ):
                break

            action, logp, value = self.agent.act(state, explore=True)
            # store the continuous signal before discretisation
            continuous_action = value
            env_action = 1 if continuous_action > 0 else 0
            next_state, reward, done, _ = environment.step(env_action)
            timesteps += 1
            self._step += 1
            episode_reward += reward

            self.replay.push(
                {
                    "state": state,
                    "continuous_action": continuous_action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                    "logp": logp,
                }
            )
            state = next_state

            metrics: Dict[str, float] = {
                "it": iteration,
                "timesteps": timesteps,
                "episode_count": 0,
                "mean_reward": None,
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            }

            if len(self.replay) >= self.config.min_buffer:
                metrics.update(self._update())
                iteration += 1

            if done:
                metrics["episode_count"] = 1
                metrics["mean_reward"] = episode_reward
                episode_reward = 0.0
                state = environment.reset()

            if metrics_callback is not None:
                try:
                    metrics_callback(metrics)
                except Exception:
                    pass

            if total_timesteps is not None and timesteps >= total_timesteps:
                break
