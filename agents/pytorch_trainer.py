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
    from agents.ppo_agent import PPOAgent
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
            ent_coef=0.05,
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

        def build_agent(self):
            agent = PPOAgent()
            agent.net = self.net
            agent.opt = self.opt
            agent.device = self.device
            return agent

        def collect_trajectory(self, envs=None, horizon=2048, stop_event=None):
            """Collect a `horizon`-length trajectory across one or more environments.

            Supports both list of environments (sequential) and vectorized environments.
            """
            from game.vec_env import SubprocVecEnv

            if envs is None:
                envs = [GameEnv()]
            elif isinstance(envs, GameEnv):
                envs = [envs]

            # 檢查是否為向量化環境
            is_vec_env = isinstance(envs, SubprocVecEnv)

            if is_vec_env:
                # 使用真正的並行環境
                return self._collect_trajectory_vectorized(envs, horizon, stop_event)
            else:
                # 使用串行環境（原有邏輯）
                envs = list(envs) or [GameEnv()]
                return self._collect_trajectory_sequential(envs, horizon, stop_event)

        def _collect_trajectory_vectorized(self, vec_env, horizon, stop_event=None):
            """使用向量化環境並行收集軌跡"""
            n_envs = len(vec_env)
            states = vec_env.reset()  # shape: (n_envs, state_dim)
            episode_returns = [0.0 for _ in range(n_envs)]

            batch_states = []
            actions, rewards, dones, values, logps, next_values = [], [], [], [], [], []
            ep_rewards = []

            steps = 0
            while steps < horizon:
                if (
                    stop_event is not None
                    and getattr(stop_event, "is_set", lambda: False)()
                ):
                    break

                # 批次處理所有環境的狀態
                s_batch = torch.tensor(
                    states, dtype=torch.float32, device=self.device
                )  # (n_envs, state_dim)

                with torch.no_grad():
                    logits, vals = self.net(s_batch)  # (n_envs, 1), (n_envs, 1)
                    probs = torch.sigmoid(logits)
                    dist = torch.distributions.Bernoulli(probs=probs)
                    action_tensors = dist.sample()  # (n_envs, 1)
                    logp = dist.log_prob(action_tensors)  # (n_envs, 1)

                actions_np = action_tensors.cpu().numpy().flatten().astype(int)

                # 並行執行所有環境
                next_states, rews, dones_arr, infos = vec_env.step(actions_np)

                # 記錄數據
                for i in range(n_envs):
                    batch_states.append(states[i])
                    actions.append(actions_np[i])
                    rewards.append(rews[i])
                    dones.append(dones_arr[i])
                    values.append(vals[i].item())
                    logps.append(logp[i].item())

                    episode_returns[i] += float(rews[i])

                    if dones_arr[i]:
                        ep_rewards.append(episode_returns[i])
                        episode_returns[i] = 0.0
                        # 計算 next_value (重置後為 0)
                        next_values.append(0.0)
                    else:
                        # 計算 next_value
                        with torch.no_grad():
                            s_next_t = torch.tensor(
                                next_states[i], dtype=torch.float32, device=self.device
                            ).unsqueeze(0)
                            _, next_value = self.net(s_next_t)
                            next_values.append(float(next_value.item()))

                states = next_states
                steps += n_envs

            if not batch_states:
                empty = torch.empty((0, 5), dtype=torch.float32, device=self.device)
                zero = torch.empty((0, 1), dtype=torch.float32, device=self.device)
                return (
                    {
                        "states": empty,
                        "actions": zero,
                        "logps": zero,
                        "returns": zero,
                        "advs": zero,
                    },
                    ep_rewards,
                )

            # 計算 GAE 優勢
            if len(next_values) < len(rewards):
                next_values.extend([0.0] * (len(rewards) - len(next_values)))

            advs = []
            gae = 0.0
            for i in reversed(range(len(rewards))):
                delta = (
                    rewards[i]
                    + self.gamma * next_values[i] * (1 - dones[i])
                    - values[i]
                )
                gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
                advs.insert(0, gae)

            returns = [adv + val for adv, val in zip(advs, values)]

            return (
                {
                    "states": torch.tensor(
                        batch_states, dtype=torch.float32, device=self.device
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
                },
                ep_rewards,
            )

        def _collect_trajectory_sequential(self, envs, horizon, stop_event=None):
            """使用串行環境收集軌跡（原有邏輯）"""

            states = [env.reset() for env in envs]
            episode_returns = [0.0 for _ in envs]

            batch_states = []
            actions, rewards, dones, values, logps, next_values = [], [], [], [], [], []
            ep_rewards = []

            for t in range(horizon):
                if (
                    stop_event is not None
                    and getattr(stop_event, "is_set", lambda: False)()
                ):
                    break

                env_idx = t % len(envs)
                env = envs[env_idx]
                s = states[env_idx]

                s_t = torch.tensor(
                    s, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                logits, value = self.net(s_t)
                prob = torch.sigmoid(logits)
                dist = torch.distributions.Bernoulli(probs=prob)
                action_tensor = dist.sample()
                action = int(action_tensor.item())
                logp = dist.log_prob(action_tensor)

                s_next, r, done, _ = env.step(action)

                batch_states.append(s)
                actions.append(action)
                rewards.append(r)
                dones.append(done)
                values.append(value.item())
                logps.append(logp.item())

                episode_returns[env_idx] += float(r)

                if done:
                    ep_rewards.append(episode_returns[env_idx])
                    episode_returns[env_idx] = 0.0
                    states[env_idx] = env.reset()
                    next_values.append(0.0)
                else:
                    states[env_idx] = s_next
                    with torch.no_grad():
                        s_next_t = torch.tensor(
                            s_next, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        _, next_value = self.net(s_next_t)
                        next_values.append(float(next_value.item()))

            if not batch_states:
                empty = torch.empty((0, 5), dtype=torch.float32, device=self.device)
                zero = torch.empty((0, 1), dtype=torch.float32, device=self.device)
                return (
                    {
                        "states": empty,
                        "actions": zero,
                        "logps": zero,
                        "returns": zero,
                        "advs": zero,
                    },
                    ep_rewards,
                )

            if len(next_values) < len(rewards):
                next_values.extend([0.0] * (len(rewards) - len(next_values)))

            advs = []
            gae = 0.0
            for i in reversed(range(len(rewards))):
                delta = (
                    rewards[i]
                    + self.gamma * next_values[i] * (1 - dones[i])
                    - values[i]
                )
                gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
                advs.insert(0, gae)

            returns = [adv + val for adv, val in zip(advs, values)]

            batch = {
                "states": torch.tensor(
                    np.array(batch_states), dtype=torch.float32, device=self.device
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
                    "optimizer_state": self.opt.state_dict(),
                },
                path,
            )
            return path

        def train(
            self,
            total_timesteps=None,
            env=None,
            envs=None,
            log_interval=1,
            metrics_callback=None,
            stop_event=None,
            initial_iteration=0,
        ):
            """Main training loop.

            metrics_callback: optional callable(metrics: dict) called after each
            PPO update with keys: it, loss, policy_loss, value_loss, entropy,
            timesteps, mean_reward, episode_count
            """
            if envs is not None:
                env_list = list(envs) or [GameEnv()]
            elif env is not None:
                env_list = env if isinstance(env, (list, tuple)) else [env]
            else:
                env_list = [GameEnv()]

            env_list = [e if isinstance(e, GameEnv) else GameEnv() for e in env_list]

            timesteps = 0
            it = initial_iteration

            while True:
                # honor external stop request
                if (
                    stop_event is not None
                    and getattr(stop_event, "is_set", lambda: False)()
                ):
                    break

                batch, ep_rewards = self.collect_trajectory(
                    env_list, stop_event=stop_event
                )
                if batch["states"].numel() == 0:
                    continue
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

                if total_timesteps is not None and timesteps >= total_timesteps:
                    break

                # allow stopping after update
                if (
                    stop_event is not None
                    and getattr(stop_event, "is_set", lambda: False)()
                ):
                    break

            self.writer.close()

except Exception:
    # Torch not available: keep file importable but trainer unavailable
    pass
