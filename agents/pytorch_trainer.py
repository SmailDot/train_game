"""PPO trainer using PyTorch.

This file implements a compact, readable PPO training loop with checkpointing
and TensorBoard logging. It expects `agents.networks.ActorCritic` to be a
PyTorch nn.Module (the file provides a fallback but for training you must
install torch).

Notes:
- This implementation is intentionally clear rather than highly-optimized.
- For faster training use vectorized envs (multiprocessing) and larger batch sizes.
"""

import os

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter

    from agents.networks import ActorCritic
    from game.environment import GameEnv

    class PPOTrainer:
        def __init__(
            self,
            save_dir="checkpoints",
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            batch_size=64,
            ppo_epochs=4,
            device=None,
        ):
            self.device = device or (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.net = ActorCritic().to(self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.gamma = gamma
            self.lam = lam
            self.clip_eps = clip_eps
            self.vf_coef = vf_coef
            self.ent_coef = ent_coef
            self.batch_size = batch_size
            self.ppo_epochs = ppo_epochs
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))

        def collect_trajectory(self, env: GameEnv, horizon=2048):
            """Collect a trajectory of length `horizon`.

            Returns a tuple (batch, ep_rewards)
            where batch is the usual tensor batch used by ppo_update and
            ep_rewards is a list of episode total rewards encountered during
            the horizon (may be empty).
            """
            states, actions, rewards, dones, values, logps = [], [], [], [], [], []
            s = env.reset()
            ep_rewards = []
            cur_ep_reward = 0.0
            for t in range(horizon):
                s_t = torch.tensor(
                    s, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                logits, value = self.net(s_t)
                prob = torch.sigmoid(logits)
                # sample
                m = torch.distributions.Bernoulli(probs=prob)
                a = m.sample().item()
                logp = m.log_prob(
                    torch.tensor(a, device=self.device, dtype=torch.float32)
                )

                s_next, r, done, _ = env.step(int(a))

                states.append(s)
                actions.append(int(a))
                rewards.append(r)
                dones.append(done)
                values.append(value.item())
                logps.append(logp.item())

                # track episode reward for reporting
                cur_ep_reward += float(r)
                if done:
                    ep_rewards.append(cur_ep_reward)
                    cur_ep_reward = 0.0
                    s = env.reset()
                else:
                    s = s_next

            # compute last value
            s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_val = self.net(s_t)
            last_val = last_val.item()

            # compute advantages via GAE
            advs = []
            gae = 0.0
            for i in reversed(range(len(rewards))):
                delta = (
                    rewards[i]
                    + self.gamma
                    * (last_val if i == len(rewards) - 1 else values[i + 1])
                    * (1 - dones[i])
                    - values[i]
                )
                gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
                advs.insert(0, gae)

            returns = [a + v for a, v in zip(advs, values)]

            # convert to tensors
            batch = {
                "states": torch.tensor(
                    np.array(states), dtype=torch.float32, device=self.device
                ),
                "actions": torch.tensor(
                    actions, dtype=torch.float32, device=self.device
                ).unsqueeze(1),
                "logps": torch.tensor(
                    logps, dtype=torch.float32, device=self.device
                ).unsqueeze(1),
                "returns": torch.tensor(
                    returns, dtype=torch.float32, device=self.device
                ).unsqueeze(1),
                "advs": torch.tensor(
                    advs, dtype=torch.float32, device=self.device
                ).unsqueeze(1),
            }
            # normalize advantages
            batch["advs"] = (batch["advs"] - batch["advs"].mean()) / (
                batch["advs"].std() + 1e-8
            )
            return batch, ep_rewards

        def ppo_update(self, batch):
            N = batch["states"].size(0)
            idxs = np.arange(N)
            for _ in range(self.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, N, self.batch_size):
                    mb_idx = idxs[start : start + self.batch_size]
                    s = batch["states"][mb_idx]
                    a = batch["actions"][mb_idx]
                    old_logp = batch["logps"][mb_idx]
                    ret = batch["returns"][mb_idx]
                    adv = batch["advs"][mb_idx]

                    logits, value = self.net(s)
                    prob = torch.sigmoid(logits)
                    m = torch.distributions.Bernoulli(probs=prob)
                    new_logp = m.log_prob(a)
                    entropy = m.entropy().mean()

                    ratio = torch.exp(new_logp - old_logp)
                    surr1 = ratio * adv
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                        * adv
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(value, ret)

                    loss = (
                        policy_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy
                    )

                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.opt.step()

            return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

        def save(self, step):
            path = os.path.join(self.save_dir, f"checkpoint_{step}.pt")
            torch.save(
                {
                    "model_state": self.net.state_dict(),
                    "optimizer": self.opt.state_dict(),
                },
                path,
            )
            return path

        def train(self, total_timesteps=20000, env=None, log_interval=1, metrics_callback=None):
            """Main training loop.

            metrics_callback: optional callable(metrics: dict) called after each
            PPO update with keys: it, loss, policy_loss, value_loss, entropy,
            timesteps, mean_reward, episode_count
            """
            env = env or GameEnv()
            timesteps = 0
            it = 0
            while timesteps < total_timesteps:
                batch, ep_rewards = self.collect_trajectory(env)
                timesteps += batch["states"].size(0)
                loss, ploss, vloss, ent = self.ppo_update(batch)
                it += 1
                # log
                self.writer.add_scalar("loss/total", loss, it)
                self.writer.add_scalar("loss/policy", ploss, it)
                self.writer.add_scalar("loss/value", vloss, it)
                self.writer.add_scalar("policy/entropy", ent, it)

                mean_reward = float(np.mean(ep_rewards)) if ep_rewards else None
                episode_count = len(ep_rewards)

                # callback for UI or external monitor
                try:
                    if metrics_callback is not None:
                        metrics_callback(
                            {
                                "it": it,
                                "loss": float(loss),
                                "policy_loss": float(ploss),
                                "value_loss": float(vloss),
                                "entropy": float(ent),
                                "timesteps": int(timesteps),
                                "mean_reward": mean_reward,
                                "episode_count": episode_count,
                            }
                        )
                except Exception:
                    # metrics callback must not break training
                    pass

                if it % 10 == 0:
                    cp = self.save(it)
                    print(f"Saved checkpoint {cp}")

            self.writer.close()

except Exception:
    # Torch not available: keep file importable but trainer unavailable
    pass
