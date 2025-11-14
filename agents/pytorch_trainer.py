"""PPO trainer using PyTorch.

This file implements a compact, readable PPO training loop with checkpointing
and TensorBoard logging. It expects `agents.networks.ActorCritic` to be a
PyTorch nn.Module (the file provides a fallback but for training you must
install torch).

Notes:
- This implementation is intentionally clear rather than highly-optimized.
- For faster training use vectorized envs (multiprocessing) and larger batch sizes.
"""

import json
import os
from pathlib import Path

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

            # 存儲初始參數（用於動態更新）
            self.lr = lr
            self.gamma = gamma
            self.lam = lam
            self.clip_eps = clip_eps
            self.vf_coef = vf_coef
            self.ent_coef = ent_coef
            self.batch_size = batch_size
            self.ppo_epochs = ppo_epochs

            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

            # 配置文件路徑
            self.config_path = Path(__file__).parent.parent / "training_config.json"
            self._last_config_check = 0

            # 學習率調度器配置
            self.initial_lr = lr
            self.scheduler_config = self._load_scheduler_config()
            self._setup_lr_scheduler()

            print(f"💾 配置文件路徑: {self.config_path}")
            print("   可在訓練過程中修改此文件來調整參數")
            print(f"🎯 學習率調度器: {self.scheduler_config.get('type', 'none')}")
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))

        def _load_scheduler_config(self):
            """從配置文件加載學習率調度器設置"""
            try:
                if self.config_path.exists():
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    return config.get("lr_scheduler", {"type": "none"})
            except Exception:
                pass
            return {"type": "none"}

        def _setup_lr_scheduler(self):
            """設置學習率調度器"""
            scheduler_type = self.scheduler_config.get("type", "none")

            # 性能追蹤（用於自適應調度）
            self.best_reward = float("-inf")
            self.best_max_reward = float("-inf")  # 追蹤最高單回合分數
            self.best_min_reward = float("-inf")  # 追蹤最好的最低分（下限提升）
            self.patience_counter = 0
            self.lr_history = [self.initial_lr]

            if scheduler_type == "step":
                # 階梯式衰減：每 N 個迭代降低學習率
                step_size = self.scheduler_config.get("step_size", 100)
                gamma = self.scheduler_config.get("gamma", 0.9)
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.opt, step_size=step_size, gamma=gamma
                )
                print(f"   每 {step_size} 迭代學習率 ×{gamma} (階梯式衰減)")

            elif scheduler_type == "exponential":
                # 指數衰減：每個迭代都衰減
                gamma = self.scheduler_config.get("gamma", 0.999)
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.opt, gamma=gamma
                )
                print(f"   每迭代學習率 ×{gamma} (指數衰減)")

            elif scheduler_type == "reduce_on_plateau":
                # 基於性能：獎勵停滯時降低學習率
                patience = self.scheduler_config.get("patience", 20)
                factor = self.scheduler_config.get("factor", 0.5)
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt,
                    mode="max",
                    factor=factor,
                    patience=patience,
                    verbose=True,
                )
                print(f"   獎勵停滯 {patience} 次後學習率 ×{factor} (性能自適應)")

            elif scheduler_type == "cosine":
                # 餘弦退火：平滑衰減到最小值
                T_max = self.scheduler_config.get("T_max", 500)
                eta_min = self.scheduler_config.get("eta_min", 1e-6)
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, T_max=T_max, eta_min=eta_min
                )
                print(f"   {T_max} 迭代內餘弦衰減至 {eta_min} (餘弦退火)")

            elif scheduler_type == "adaptive":
                # 自定義自適應策略（不使用 PyTorch 內建）
                self.lr_scheduler = None
                patience = self.scheduler_config.get("patience", 30)
                factor = self.scheduler_config.get("factor", 0.5)
                min_lr = self.scheduler_config.get("min_lr", 1e-6)
                print(
                    f"   自適應調整：{patience}次無改善→學習率×{factor}，最低{min_lr}"
                )

            else:
                # 不使用調度器
                self.lr_scheduler = None
                print("   不使用學習率調度")

        def _update_lr_adaptive(self, mean_reward, max_reward, min_reward, iteration):
            """自定義自適應學習率更新邏輯（三指標系統）

            Args:
                mean_reward: 平均獎勵（評估整體穩定性）
                max_reward: 最高單回合獎勵（評估潛力上限）
                min_reward: 最低單回合獎勵（評估穩定性下限）
                iteration: 當前迭代數

            策略：
                - 平均分提升 → 整體進步，重置 patience
                - 最高分突破 → 發現潛力，減少 patience（鼓勵探索）
                - 最低分提升 → 下限改善，減少 patience（穩定性提升）
                - 最低分惡化 → 增加 patience（警告：策略不穩定）
            """
            if self.scheduler_config.get("type") != "adaptive":
                return

            if mean_reward is None:
                return

            patience = self.scheduler_config.get("patience", 30)
            factor = self.scheduler_config.get("factor", 0.5)
            min_lr = self.scheduler_config.get("min_lr", 1e-6)
            improvement_threshold = self.scheduler_config.get(
                "improvement_threshold", 0.01
            )

            # 檢查三個指標的改善情況
            mean_improved = mean_reward > self.best_reward * (1 + improvement_threshold)
            max_improved = (
                max_reward is not None
                and max_reward > self.best_max_reward * (1 + improvement_threshold / 2)
            )

            # 最低分改善：使用更寬鬆的閾值（0.5%），因為負分提升很困難
            min_improved = (
                min_reward is not None
                and min_reward > self.best_min_reward * (1 + improvement_threshold / 2)
            )

            # 最低分惡化檢測：如果最低分下降超過5%，說明策略變不穩定
            min_degraded = (
                min_reward is not None
                and self.best_min_reward > float("-inf")
                and min_reward < self.best_min_reward * (1 - improvement_threshold * 5)
            )

            # 更新最佳記錄
            if mean_improved:
                self.best_reward = mean_reward
                self.patience_counter = 0
                print(f"   📈 新最佳平均獎勵: {mean_reward:.2f}")

            if max_improved:
                self.best_max_reward = max_reward
                self.patience_counter = max(0, self.patience_counter - 5)
                print(f"   🌟 新最高單回合分數: {max_reward:.2f}（減少5次patience）")

            if min_improved:
                self.best_min_reward = min_reward
                self.patience_counter = max(0, self.patience_counter - 3)
                print(
                    f"   ⬆️ 最低分提升: {min_reward:.2f}（減少3次patience，穩定性改善）"
                )

            # 警告：最低分惡化
            if min_degraded:
                self.patience_counter += 2  # 增加2次patience，更快觸發LR降低
                print(
                    f"   ⚠️ 最低分惡化: {min_reward:.2f}（增加2次patience，策略不穩定）"
                )

            # 如果沒有任何改善
            if not mean_improved and not max_improved and not min_improved:
                self.patience_counter += 1

            # 如果停滯太久，降低學習率
            if self.patience_counter >= patience:
                current_lr = self.opt.param_groups[0]["lr"]
                new_lr = max(current_lr * factor, min_lr)

                if new_lr != current_lr:
                    for param_group in self.opt.param_groups:
                        param_group["lr"] = new_lr
                    self.lr = new_lr
                    self.lr_history.append(new_lr)
                    print(f"\n📉 學習率自適應調整: {current_lr:.6f} → {new_lr:.6f}")
                    print(f"   原因: {patience} 次迭代無顯著改善")
                    print(
                        f"   📊 當前最佳 - 平均: {self.best_reward:.2f} "
                        f"| 最高: {self.best_max_reward:.2f} "
                        f"| 最低: {self.best_min_reward:.2f}"
                    )
                    self.patience_counter = 0
                else:
                    print(f"\n⚠️ 學習率已達最小值 {min_lr:.6f}，無法再降低")

        def _check_performance_degradation(
            self, mean_reward, max_reward, min_reward, iteration
        ):
            """檢測性能嚴重退化並回檔到最佳檢查點"""
            # 只有在有足夠訓練歷史時才檢查（至少 100 次迭代）
            if iteration < 100:
                return False

            # 只有在所有獎勵都有效時才檢查
            if (
                mean_reward is None
                or max_reward is None
                or min_reward is None
                or self.best_reward <= 0
            ):
                return False

            # 計算各指標的下降比例
            mean_drop = (self.best_reward - mean_reward) / abs(self.best_reward)
            max_drop = (
                (self.best_max_reward - max_reward) / abs(self.best_max_reward)
                if self.best_max_reward > 0
                else 0
            )
            min_drop = (
                (self.best_min_reward - min_reward) / abs(self.best_min_reward)
                if self.best_min_reward > 0
                else 0
            )

            # 嚴格的退化閾值：任一指標下降超過 40% 即視為崩潰
            degradation_threshold = 0.40

            # 檢測崩潰條件（任一指標嚴重下降）
            is_catastrophic = (
                mean_drop > degradation_threshold
                or max_drop > degradation_threshold
                or (
                    min_drop > degradation_threshold and self.best_min_reward > 10
                )  # 最低分只有在原本較高時才關注
            )

            if is_catastrophic:
                print(f"\n{'='*60}")
                print("⚠️⚠️⚠️ 檢測到性能崩潰！⚠️⚠️⚠️")
                print(f"{'='*60}")
                print("📉 當前指標 vs 最佳記錄：")
                print(
                    f"   平均分: {mean_reward:.2f} (最佳: {self.best_reward:.2f}) "
                    f"↓ {mean_drop*100:.1f}%"
                )
                print(
                    f"   最高分: {max_reward:.2f} (最佳: {self.best_max_reward:.2f}) "
                    f"↓ {max_drop*100:.1f}%"
                )
                print(
                    f"   最低分: {min_reward:.2f} (最佳: {self.best_min_reward:.2f}) "
                    f"↓ {min_drop*100:.1f}%"
                )
                print("\n🔄 正在回檔到最佳檢查點...")

                # 執行回檔
                success = self._rollback_to_best_checkpoint()

                if success:
                    print("✅ 成功回檔！繼續訓練...")
                    print(f"{'='*60}\n")
                    return True
                else:
                    print("❌ 回檔失敗，繼續當前訓練...")
                    print(f"{'='*60}\n")
                    return False

            return False

        def _rollback_to_best_checkpoint(self):
            """回檔到最佳檢查點"""
            try:
                # 尋找最佳檢查點（基於迭代次數）
                checkpoints = []
                for file in os.listdir(self.save_dir):
                    if file.startswith("checkpoint_") and file.endswith(".pt"):
                        try:
                            step = int(
                                file.replace("checkpoint_", "").replace(".pt", "")
                            )
                            checkpoints.append((step, file))
                        except ValueError:
                            continue

                if not checkpoints:
                    print("   ⚠️ 找不到可用的檢查點")
                    return False

                # 按迭代次數排序，取最新的檢查點
                checkpoints.sort(reverse=True)

                # 嘗試載入最近的幾個檢查點（跳過當前迭代）
                for step, filename in checkpoints[:5]:  # 嘗試最近 5 個檢查點
                    checkpoint_path = os.path.join(self.save_dir, filename)

                    try:
                        print(f"   📂 嘗試載入檢查點: {filename}")
                        checkpoint = torch.load(
                            checkpoint_path, map_location=self.device
                        )

                        # 載入模型狀態
                        if "model_state" in checkpoint:
                            self.net.load_state_dict(checkpoint["model_state"])
                            print("      ✓ 模型參數已載入")
                        else:
                            print("      ✗ 檢查點格式錯誤")
                            continue

                        # 載入優化器狀態（重置學習動量）
                        if "optimizer_state" in checkpoint:
                            self.opt.load_state_dict(checkpoint["optimizer_state"])
                            print("      ✓ 優化器狀態已載入")

                        # 重置 patience 計數器
                        self.patience_counter = 0

                        # 重置學習率為初始值或略低的值
                        rollback_lr = self.initial_lr * 0.5  # 使用稍低的學習率
                        for param_group in self.opt.param_groups:
                            param_group["lr"] = rollback_lr
                        print(f"      ✓ 學習率重置為: {rollback_lr:.6f}")

                        print(f"\n   ✅ 成功從迭代 #{step} 回檔！")
                        return True

                    except Exception as e:
                        print(f"      ✗ 載入失敗: {e}")
                        continue

                print("   ❌ 所有檢查點都無法載入")
                return False

            except Exception as e:
                print(f"   ❌ 回檔過程發生錯誤: {e}")
                import traceback

                traceback.print_exc()
                return False

        def _load_dynamic_config(self, iteration):
            """每10個迭代檢查並加載配置文件更新"""
            if iteration % 10 != 0:
                return False

            if not self.config_path.exists():
                return False

            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # 根據設備類型選擇配置
                device_type = (
                    self.device.type
                    if hasattr(self.device, "type")
                    else str(self.device)
                )
                mode = "gpu_training" if device_type == "cuda" else "cpu_training"
                params = config.get(mode, {})

                updated = False
                updates = []

                # 檢查並更新學習率
                new_lr = params.get("learning_rate")
                if new_lr and abs(new_lr - self.lr) > 1e-9:
                    self.lr = new_lr
                    for param_group in self.opt.param_groups:
                        param_group["lr"] = new_lr
                    updates.append(f"學習率: {new_lr}")
                    updated = True

                # 更新其他參數
                if "gamma" in params and params["gamma"] != self.gamma:
                    self.gamma = params["gamma"]
                    updates.append(f"gamma: {self.gamma}")
                    updated = True

                if "gae_lambda" in params and params["gae_lambda"] != self.lam:
                    self.lam = params["gae_lambda"]
                    updates.append(f"lambda: {self.lam}")
                    updated = True

                if "clip_range" in params and params["clip_range"] != self.clip_eps:
                    self.clip_eps = params["clip_range"]
                    updates.append(f"clip: {self.clip_eps}")
                    updated = True

                if "vf_coef" in params and params["vf_coef"] != self.vf_coef:
                    self.vf_coef = params["vf_coef"]
                    updates.append(f"vf_coef: {self.vf_coef}")
                    updated = True

                if "ent_coef" in params and params["ent_coef"] != self.ent_coef:
                    self.ent_coef = params["ent_coef"]
                    updates.append(f"ent_coef: {self.ent_coef}")
                    updated = True

                if "batch_size" in params and params["batch_size"] != self.batch_size:
                    self.batch_size = params["batch_size"]
                    updates.append(f"batch_size: {self.batch_size}")
                    updated = True

                if "ppo_epochs" in params and params["ppo_epochs"] != self.ppo_epochs:
                    self.ppo_epochs = params["ppo_epochs"]
                    updates.append(f"ppo_epochs: {self.ppo_epochs}")
                    updated = True

                if updated:
                    print("\n⚙️ 參數已從配置文件更新:")
                    for update in updates:
                        print(f"   • {update}")
                    print()

                return updated

            except Exception as e:
                print(f"⚠️ 無法讀取配置文件: {e}")
                return False

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
            print(f"🚀 使用 {n_envs} 個並行環境收集數據...")
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
                # 檢查並更新配置（每10次迭代）
                self._load_dynamic_config(it)

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
                max_reward = float(np.max(ep_rewards)) if ep_rewards else None
                min_reward = float(np.min(ep_rewards)) if ep_rewards else None
                episode_count = len(ep_rewards)

                # 記錄獎勵統計到 TensorBoard
                if mean_reward is not None:
                    self.writer.add_scalar("reward/mean", mean_reward, it)
                    self.writer.add_scalar("reward/max", max_reward, it)
                    self.writer.add_scalar("reward/min", min_reward, it)

                    # 檢測性能退化（每10次迭代才檢查，避免過度敏感）
                    if it % 10 == 0:
                        self._check_performance_degradation(
                            mean_reward, max_reward, min_reward, it
                        )

                # 儲存歷史數據用於比較
                if not hasattr(self, "_history"):
                    self._history = {
                        "loss": [],
                        "policy_loss": [],
                        "value_loss": [],
                        "entropy": [],
                        "mean_reward": [],
                        "max_reward": [],
                        "min_reward": [],
                        "weight_mean": [],
                        "weight_std": [],
                        "grad_norm": [],
                    }

                self._history["loss"].append(loss)
                self._history["policy_loss"].append(ploss)
                self._history["value_loss"].append(vloss)
                self._history["entropy"].append(ent)
                if mean_reward is not None:
                    self._history["mean_reward"].append(mean_reward)
                    self._history["max_reward"].append(max_reward)
                    self._history["min_reward"].append(min_reward)

                # 打印詳細的訓練診斷信息（每10次迭代）
                if it % 10 == 0:
                    print(f"\n{'='*60}")
                    print(f"訓練迭代 #{it}")
                    print(f"{'='*60}")
                    print("📊 Loss 指標:")
                    print(f"  總損失: {loss:.4f}")
                    print(f"  策略損失: {ploss:.4f}")
                    print(f"  價值損失: {vloss:.4f}")
                    print(f"  熵值: {ent:.4f}")
                    print("\n🎮 訓練效果:")
                    if mean_reward is not None:
                        print(f"  平均獎勵: {mean_reward:.2f}")
                        print(f"  最高獎勵: {max_reward:.2f}")
                        print(f"  最低獎勵: {min_reward:.2f}")
                    else:
                        print("  平均獎勵: N/A (尚未完成任何回合)")
                    print(f"  完成回合數: {episode_count}")
                    print(f"  總時間步: {timesteps}")

                    # 顯示並行環境信息
                    if hasattr(env_list, "__len__") and len(env_list) > 1:
                        print("\n🔄 並行環境:")
                        print(f"  環境數量: {len(env_list)}")
                        print(f"  理論加速: {len(env_list)}x")

                    print("\n⚙️ 網路狀態:")
                    # 檢查網路權重是否在更新
                    current_w_mean = 0.0
                    current_w_std = 0.0
                    try:
                        w = self.net.get_weight_matrix()
                        if w is not None:
                            current_w_mean = float(np.mean(np.abs(w)))
                            current_w_std = float(np.std(w))
                            print(f"  權重平均值: {current_w_mean:.6f}")
                            print(f"  權重標準差: {current_w_std:.6f}")

                            # 儲存權重歷史
                            self._history["weight_mean"].append(current_w_mean)
                            self._history["weight_std"].append(current_w_std)
                        else:
                            print("  權重: 無法獲取")
                    except Exception as e:
                        print(f"  權重: 獲取失敗 ({e})")

                    # 檢查梯度
                    grad_norms = []
                    for param in self.net.parameters():
                        if param.grad is not None:
                            grad_norms.append(float(param.grad.norm().item()))
                    if grad_norms:
                        avg_grad = np.mean(grad_norms)
                        print(f"  平均梯度範數: {avg_grad:.6f}")
                        self._history["grad_norm"].append(avg_grad)

                        if avg_grad < 1e-6:
                            print("  ⚠️ 警告: 梯度過小，權重可能未正確更新！")
                        elif avg_grad > 0.001:
                            print("  ✅ 梯度正常，權重正在更新")
                    else:
                        print("  梯度: 無")

                    # 與上次迭代比較 (如果有歷史數據)
                    if len(self._history["loss"]) >= 2:
                        print(f"\n📈 與上次比較 (迭代 #{it-10}):")

                        loss_change = loss - self._history["loss"][-2]
                        loss_arrow = "📉" if loss_change < 0 else "📈"
                        print(f"  總損失: {loss_change:+.4f} {loss_arrow}")

                        ploss_change = ploss - self._history["policy_loss"][-2]
                        print(f"  策略損失: {ploss_change:+.4f}")

                        vloss_change = vloss - self._history["value_loss"][-2]
                        print(f"  價值損失: {vloss_change:+.4f}")

                        ent_change = ent - self._history["entropy"][-2]
                        print(f"  熵值: {ent_change:+.4f}")

                        if len(self._history["weight_mean"]) >= 2:
                            w_mean_change = (
                                current_w_mean - self._history["weight_mean"][-2]
                            )
                            w_std_change = (
                                current_w_std - self._history["weight_std"][-2]
                            )
                            print(f"  權重平均: {w_mean_change:+.6f}")
                            print(f"  權重標準差: {w_std_change:+.6f}")

                            if abs(w_mean_change) < 1e-6 and abs(w_std_change) < 1e-6:
                                print("  ⚠️ 權重幾乎沒有變化！")
                            else:
                                print("  ✅ 權重正在更新")

                        if len(self._history["mean_reward"]) >= 2:
                            reward_change = (
                                self._history["mean_reward"][-1]
                                - self._history["mean_reward"][-2]
                            )
                            reward_arrow = "📈" if reward_change > 0 else "📉"
                            print(f"  平均獎勵: {reward_change:+.2f} {reward_arrow}")

                    # 學習進度評估
                    if mean_reward is not None:
                        if mean_reward > 20:
                            print("\n✅ 學習進度: 優秀 (獎勵 > 20)")
                        elif mean_reward > 10:
                            print("\n📈 學習進度: 良好 (獎勵 > 10)")
                        elif mean_reward > 5:
                            print("\n⚡ 學習進度: 進步中 (獎勵 > 5)")
                        elif mean_reward > 0:
                            print("\n🔄 學習進度: 緩慢 (獎勵 > 0)")
                        else:
                            print("\n⚠️ 學習進度: 需要調整 (獎勵 < 0)")
                            print("   建議: 檢查獎勵函數、降低學習率或調整網路結構")
                    else:
                        # 即使沒有完成回合，也顯示學習狀態
                        print("\n🔄 學習狀態:")
                        if loss < 0.05:
                            print(f"  損失很低 ({loss:.4f})，但沒有完成回合")
                            print("  可能原因: 遊戲太難、獎勵函數問題")
                        elif ent < 0.05:
                            print(f"  熵值過低 ({ent:.4f})，策略可能過早收斂")
                            print("  建議: 增加 ent_coef 或重置訓練")
                        else:
                            print("  仍在學習中，繼續訓練...")

                    print(f"{'='*60}\n")

                # 更新學習率調度器
                if self.lr_scheduler is not None:
                    scheduler_type = self.scheduler_config.get("type", "none")
                    if (
                        scheduler_type == "reduce_on_plateau"
                        and mean_reward is not None
                    ):
                        # ReduceLROnPlateau 需要監控指標
                        self.lr_scheduler.step(mean_reward)
                    elif scheduler_type in ["step", "exponential", "cosine"]:
                        # 其他調度器基於迭代次數
                        self.lr_scheduler.step()

                # 自定義自適應學習率調整
                if it % 10 == 0:  # 每10次迭代檢查一次
                    self._update_lr_adaptive(mean_reward, max_reward, min_reward, it)

                    # 顯示當前學習率
                    current_lr = self.opt.param_groups[0]["lr"]
                    if abs(current_lr - self.initial_lr) > 1e-9:
                        print(
                            f"📊 當前學習率: {current_lr:.6f} "
                            f"(初始: {self.initial_lr:.6f})"
                        )

                # callback for UI or external monitor
                try:
                    if metrics_callback is not None:
                        # 獲取網路權重用於視覺化
                        weight_matrix = None
                        try:
                            weight_matrix = self.net.get_weight_matrix()
                        except Exception:
                            pass

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
                                "weights": weight_matrix,
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
