"""
Q-Learning Trainer (DQN) using PyTorch.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agents.q_learning_agent import QLearningAgent, ReplayMemory, Transition
from game.environment import GameEnv


class QLearningTrainer:
    def __init__(
        self,
        mode="dqn",  # 'dqn' or 'double_dqn'
        save_dir="checkpoints_q_learning",
        lr=1e-4,
        gamma=0.99,
        batch_size=128,
        memory_size=10000,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        target_update_interval=10,
        device=None,
    ):
        self.mode = mode
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.agent = QLearningAgent(device=self.device)
        self.agent.optimizer = optim.Adam(self.agent.policy_net.parameters(), lr=lr)
        self.agent.memory = ReplayMemory(memory_size)

        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_interval = target_update_interval

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
        self.steps_done = 0

    def _select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        return self.agent.act(state, epsilon=eps_threshold)

    def q_learning_update(self):
        if len(self.agent.memory) < self.batch_size:
            return None, None, None

        transitions = self.agent.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.agent.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if self.mode == "double_dqn":
            # Double DQN: Use policy_net to select action, target_net to evaluate
            best_actions = (
                self.agent.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            )
            next_state_values[non_final_mask] = (
                self.agent.target_net(non_final_next_states)
                .gather(1, best_actions)
                .squeeze(1)
                .detach()
            )
        else:
            # Standard DQN: Use target_net for both selection and evaluation
            next_state_values[non_final_mask] = (
                self.agent.target_net(non_final_next_states).max(1)[0].detach()
            )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.agent.optimizer.zero_grad()
        loss.backward()
        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.agent.optimizer.step()

        return (
            loss.item(),
            state_action_values.mean().item(),
            expected_state_action_values.mean().item(),
        )

    def save(self, step):
        path = os.path.join(self.save_dir, f"checkpoint_{step}.pt")
        torch.save(
            {
                "model_state": self.agent.policy_net.state_dict(),
                "optimizer_state": self.agent.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )
        return path

    def build_agent(self) -> QLearningAgent:
        """Return an agent suitable for UI inference (shares trainer networks)."""
        # The trainer already instantiates self.agent in __init__, return it so
        # the UI can call act() against the trainer's live network weights.
        return self.agent

    def train(
        self,
        total_timesteps=50000,
        env=None,
        log_interval=1,
        metrics_callback=None,
        stop_event=None,
        initial_iteration=0,
    ):
        env = env or GameEnv()
        state = env.reset()
        episode_reward = 0
        episode_rewards = []

        it = initial_iteration

        for t in range(total_timesteps):
            if stop_event is not None and stop_event.is_set():
                break

            action, _, _ = self._select_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            s_tensor = torch.tensor([state], device=self.device, dtype=torch.float32)
            a_tensor = torch.tensor([[action]], device=self.device, dtype=torch.long)
            r_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
            ns_tensor = (
                torch.tensor([next_state], device=self.device, dtype=torch.float32)
                if not done
                else None
            )

            self.agent.memory.push(s_tensor, a_tensor, ns_tensor, r_tensor)

            state = next_state

            loss, q_values, expected_q_values = self.q_learning_update()

            if done:
                episode_rewards.append(episode_reward)
                state = env.reset()
                episode_reward = 0

            if t > 0 and t % self.target_update_interval == 0:
                self.agent.target_net.load_state_dict(
                    self.agent.policy_net.state_dict()
                )

            if t > 0 and t % log_interval == 0:
                it += 1
                mean_reward = (
                    float(np.mean(episode_rewards[-20:])) if episode_rewards else None
                )

                self.writer.add_scalar("loss/q_loss", loss or 0, it)
                self.writer.add_scalar("policy/q_values", q_values or 0, it)
                if mean_reward is not None:
                    self.writer.add_scalar("rollout/mean_reward", mean_reward, it)

                if metrics_callback is not None:
                    metrics_callback(
                        {
                            "it": it,
                            "loss": loss,
                            "policy_loss": loss,
                            "value_loss": 0,
                            "entropy": 0,
                            "timesteps": t,
                            "mean_reward": mean_reward,
                            "episode_count": len(episode_rewards),
                        }
                    )

                if it % 100 == 0:
                    cp = self.save(it)
                    print(f"Saved Q-Learning checkpoint {cp}")

        self.writer.close()
