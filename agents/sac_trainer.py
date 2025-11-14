"""
Soft Actor-Critic (SAC) Trainer for discrete action spaces.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agents.q_learning_agent import ReplayMemory, Transition
from agents.sac_agent import SACAgent
from game.environment import GameEnv


class SACTrainer:
    def __init__(
        self,
        save_dir="checkpoints_sac",
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        batch_size=256,
        memory_size=100000,
        device=None,
    ):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.agent = SACAgent(device=self.device)

        self.agent.actor_optimizer = optim.Adam(self.agent.actor.parameters(), lr=lr)
        self.agent.critic_optimizer = optim.Adam(self.agent.critic.parameters(), lr=lr)
        self.agent.memory = ReplayMemory(memory_size)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
        self.steps_done = 0

    def sac_update(self):
        if len(self.agent.memory) < self.batch_size:
            return None, None, None

        transitions = self.agent.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # --- Critic Update ---
        with torch.no_grad():
            next_action_probs = self.agent.actor.get_action_probs(non_final_next_states)
            next_log_action_probs = torch.log(next_action_probs + 1e-6)
            q1_target, q2_target = self.agent.critic_target(non_final_next_states)
            min_q_target = torch.min(q1_target, q2_target)

            # Entropy-augmented target value
            next_value = (
                next_action_probs * (min_q_target - self.alpha * next_log_action_probs)
            ).sum(dim=1)

            next_q_values = torch.zeros(self.batch_size, device=self.device)
            next_q_values[non_final_mask] = next_value
            q_target = reward_batch + self.gamma * next_q_values

        q1, q2 = self.agent.critic(state_batch)
        q1 = q1.gather(1, action_batch)
        q2 = q2.gather(1, action_batch)

        critic_loss1 = F.mse_loss(q1, q_target.unsqueeze(1))
        critic_loss2 = F.mse_loss(q2, q_target.unsqueeze(1))
        critic_loss = critic_loss1 + critic_loss2

        self.agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.agent.critic_optimizer.step()

        # --- Actor Update ---
        action_probs = self.agent.actor.get_action_probs(state_batch)
        log_action_probs = torch.log(action_probs + 1e-6)

        q1_pi, q2_pi = self.agent.critic(state_batch)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (
            (action_probs * (self.alpha * log_action_probs - min_q_pi))
            .sum(dim=1)
            .mean()
        )

        self.agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.agent.actor_optimizer.step()

        # --- Soft update target networks ---
        for target_param, param in zip(
            self.agent.critic_target.parameters(), self.agent.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return critic_loss.item(), actor_loss.item(), log_action_probs.mean().item()

    def save(self, step):
        path = os.path.join(self.save_dir, f"checkpoint_{step}.pt")
        torch.save(
            {
                "actor_state": self.agent.actor.state_dict(),
                "critic_state": self.agent.critic.state_dict(),
                "actor_optimizer_state": self.agent.actor_optimizer.state_dict(),
                "critic_optimizer_state": self.agent.critic_optimizer.state_dict(),
            },
            path,
        )
        return path

    def train(
        self,
        total_timesteps=50000,
        env=None,
        log_interval=10,
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

            action, _, _ = self.agent.act(state, explore=True)
            next_state, reward, done, _ = env.step(action)
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
            self.steps_done += 1

            if self.steps_done > self.batch_size:
                critic_loss, actor_loss, entropy = self.sac_update()
            else:
                critic_loss, actor_loss, entropy = 0, 0, 0

            if done:
                episode_rewards.append(episode_reward)
                state = env.reset()
                episode_reward = 0

            if t > 0 and t % log_interval == 0:
                it += 1
                mean_reward = (
                    float(np.mean(episode_rewards[-20:])) if episode_rewards else None
                )

                self.writer.add_scalar("loss/critic_loss", critic_loss or 0, it)
                self.writer.add_scalar("loss/actor_loss", actor_loss or 0, it)
                self.writer.add_scalar("policy/entropy", entropy or 0, it)
                if mean_reward is not None:
                    self.writer.add_scalar("rollout/mean_reward", mean_reward, it)

                if metrics_callback is not None:
                    metrics_callback(
                        {
                            "it": it,
                            "loss": (critic_loss or 0) + (actor_loss or 0),
                            "policy_loss": actor_loss,
                            "value_loss": critic_loss,
                            "entropy": entropy,
                            "timesteps": t,
                            "mean_reward": mean_reward,
                            "episode_count": len(episode_rewards),
                        }
                    )

                if it % 100 == 0:
                    cp = self.save(it)
                    print(f"Saved SAC checkpoint {cp}")

        self.writer.close()
