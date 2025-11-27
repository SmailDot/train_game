#!/usr/bin/env python3
"""
Game2048 SB3 訓練腳本

使用 Stable-Baselines3 訓練 PPO 代理，目標是達到 6666 分通關。
"""

import argparse
import os
import sys
from argparse import BooleanOptionalAction
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _import_env():
    from rl.game2048_env import Game2048Env as _Game2048Env

    return _Game2048Env


Game2048Env = _import_env()


def make_linear_schedule(start: float, end: float):
    """Create a linear schedule callable compatible with SB3."""

    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining

    return schedule


def apply_finetune_overrides(
    model: PPO,
    finetune_lr: Optional[float] = None,
    finetune_ent: Optional[float] = None,
) -> None:
    """Optionally override learning rate or entropy for fine-tuning runs."""

    if finetune_lr is not None:
        target_lr = float(finetune_lr)

        def _fixed_lr(_progress_remaining: float) -> float:
            return target_lr

        model.lr_schedule = _fixed_lr
        model.learning_rate = target_lr
        optimizer = getattr(model.policy, "optimizer", None)
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = target_lr
        print(f"⚙️ Fine-tune learning rate -> {target_lr:.2e}")

    if finetune_ent is not None:
        target_ent = float(finetune_ent)
        model.ent_coef = target_ent
        print(f"⚙️ Fine-tune entropy coef -> {target_ent:.5f}")


def get_curriculum_phases(name: str) -> List[Dict[str, dict]]:
    if name != "progressive":
        return []

    return [
        {
            "threshold": 0,
            "profile": {
                "ScrollIncreasePerPass": 0.012,
                "MaxScrollSpeed": 3.0,
                "GapShrinkPerPass": 0.4,
            },
        },
        {
            "threshold": 1_500_000,
            "profile": {
                "ScrollIncreasePerPass": 0.018,
                "MaxScrollSpeed": 3.6,
                "GapShrinkPerPass": 0.6,
            },
        },
        {
            "threshold": 3_000_000,
            "profile": {
                "ScrollIncreasePerPass": 0.025,
                "MaxScrollSpeed": 4.2,
                "GapShrinkPerPass": 0.8,
            },
        },
    ]


class WinCallback(BaseCallback):
    """
    自定義回調：監控通關事件
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.wins = 0
        self.best_score = 0

    def _on_step(self) -> bool:
        # 檢查 infos 中是否有通關
        if hasattr(self.locals, "infos"):
            for info in self.locals["infos"]:
                if info.get("win", False):
                    self.wins += 1
                    score = info.get("episode_score", 0)
                    if score > self.best_score:
                        self.best_score = score
                        if self.verbose > 0:
                            print(f"🎉 新紀錄！分數: {score}")

                    if self.verbose > 0:
                        print(f"🎯 通關 #{self.wins}！分數: {score}")

        return True


class EpisodeStatsCallback(BaseCallback):
    """Record custom environment metrics (e.g., passes, scroll speed)."""

    def __init__(self, prefix: str = "env", verbose: int = 0):
        super().__init__(verbose)
        self.prefix = prefix

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") if hasattr(self, "locals") else None
        if not infos:
            return True

        metrics = {}
        for info in infos:
            if not isinstance(info, dict):
                continue
            for key in ("passed_count", "scroll_speed", "alignment_score"):
                if key in info:
                    metrics.setdefault(key, []).append(info[key])

            # Track binary win flags to compute win rate during training
            if "win" in info:
                metrics.setdefault("win", []).append(float(bool(info["win"])))

        for key, values in metrics.items():
            if not values:
                continue

            if key == "win":
                self.logger.record(f"{self.prefix}/win_rate", float(np.mean(values)))
            else:
                self.logger.record(f"{self.prefix}/{key}", float(np.mean(values)))

        return True


class AdaptiveEntropyCallback(BaseCallback):
    """Dynamically adjust entropy coefficient based on recent win rate."""

    def __init__(
        self,
        window_size: int = 4096,
        low_threshold: float = 0.05,
        high_threshold: float = 0.25,
        increase_step: float = 5e-4,
        decrease_step: float = 3e-4,
        min_ent: float = 0.004,
        max_ent: float = 0.012,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.window_size = window_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.increase_step = increase_step
        self.decrease_step = decrease_step
        self.min_ent = min_ent
        self.max_ent = max_ent
        self._buffer: deque[float] = deque(maxlen=window_size)
        self._current_ent: Optional[float] = None

    def _on_training_start(self) -> None:
        # Capture the initial entropy coefficient from the model
        self._current_ent = float(getattr(self.model, "ent_coef", 0.01))
        if self.verbose:
            print(f"🔧 自適應熵啟動，初始 ent_coef = {self._current_ent:.5f}")

    def _set_entropy(self, value: float) -> None:
        if self._current_ent is None or abs(value - self._current_ent) < 1e-6:
            return

        self._current_ent = float(np.clip(value, self.min_ent, self.max_ent))
        self.model.ent_coef = self._current_ent
        # Log to TensorBoard for transparency
        self.logger.record("train/entropy_coef", self._current_ent)
        if self.verbose:
            print(f"⚙️ 調整 ent_coef -> {self._current_ent:.5f}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") if hasattr(self, "locals") else None
        if not infos:
            return True

        win_flag = 1.0 if any(info.get("win", False) for info in infos) else 0.0
        self._buffer.append(win_flag)

        if len(self._buffer) < self.window_size:
            return True

        win_rate = float(np.mean(self._buffer))

        if win_rate >= self.high_threshold:
            self._set_entropy(self._current_ent - self.decrease_step)
        elif win_rate <= self.low_threshold:
            self._set_entropy(self._current_ent + self.increase_step)

        return True


class CurriculumCallback(BaseCallback):
    """Gradually ramp environment difficulty according to predefined phases."""

    def __init__(self, phases: List[Dict[str, dict]], verbose: int = 0):
        super().__init__(verbose)
        self.phases = sorted(phases, key=lambda item: item["threshold"])
        self._current_phase = -1

    def _apply_phase(self, index: int) -> None:
        if index < 0 or index >= len(self.phases):
            return

        profile = self.phases[index]["profile"]
        threshold = self.phases[index]["threshold"]
        self._current_phase = index

        if self.verbose:
            print(
                "📈 課程階段"
                f" {index + 1}/{len(self.phases)} @ step {threshold:,}: {profile}"
            )

        env_method = getattr(self.training_env, "env_method", None)
        if env_method is not None:
            try:
                env_method("apply_difficulty_profile", profile)
            except AttributeError:
                if self.verbose:
                    print("⚠️ 無法套用課程設定，環境缺少 apply_difficulty_profile。")

    def _on_training_start(self) -> None:
        if self.phases:
            self._apply_phase(0)

    def _on_step(self) -> bool:
        if not self.phases:
            return True

        next_index = self._current_phase + 1
        if (
            next_index < len(self.phases)
            and self.num_timesteps >= self.phases[next_index]["threshold"]
        ):
            self._apply_phase(next_index)

        return True


def create_envs(
    n_envs: int = 32,
    normalize: bool = True,
    training: bool = True,
    norm_path: Optional[str] = None,
    seed: int = 42,
    render_mode: Optional[str] = None,
):
    """
    創建向量化環境

    Args:
        n_envs: 環境數量
        normalize: 是否使用 VecNormalize

    Returns:
        環境實例
    """
    print(f"🚀 創建 {n_envs} 個並行環境...")

    env_kwargs = {}
    if render_mode:
        env_kwargs["render_mode"] = render_mode

    vec_env = make_vec_env(Game2048Env, n_envs=n_envs, env_kwargs=env_kwargs, seed=seed)

    # 添加監控
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    vec_env = VecMonitor(vec_env, log_dir)

    # 可選：添加正規化
    if normalize:
        norm_reward = training
        if norm_path and os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env)
            print(f"📄 VecNormalize 統計載入: {norm_path}")
        else:
            if norm_path and not os.path.exists(norm_path):
                print(f"⚠️ 找不到 VecNormalize 檔案 {norm_path}，將重新初始化統計。")
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=norm_reward,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.995,
            )

        vec_env.training = training
        vec_env.norm_reward = norm_reward

    if render_mode:
        setattr(vec_env, "render_mode", render_mode)

    return vec_env


def create_callbacks(
    env,
    normalize: bool = False,
    eval_freq: int = 5000,
    save_freq: int = 10000,
    norm_path: Optional[str] = None,
    seed: int = 42,
    adaptive_entropy: bool = True,
    curriculum_phases: Optional[List[Dict[str, dict]]] = None,
):
    """建立訓練/評估所需的回調。"""

    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./checkpoints/",
        name_prefix="ppo_game2048",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    eval_env = create_envs(
        n_envs=4,
        normalize=normalize,
        training=False,
        norm_path=norm_path,
        seed=seed + 1,
        render_mode=None,
    )

    if (
        normalize
        and isinstance(env, VecNormalize)
        and isinstance(eval_env, VecNormalize)
    ):
        eval_env.obs_rms = env.obs_rms.copy()
        eval_env.ret_rms = env.ret_rms.copy()
        eval_env.training = False
        eval_env.norm_reward = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/eval/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks.append(eval_callback)

    callbacks.append(WinCallback(verbose=1))
    callbacks.append(EpisodeStatsCallback(verbose=0))
    if curriculum_phases:
        callbacks.append(CurriculumCallback(curriculum_phases, verbose=1))
    if adaptive_entropy:
        callbacks.append(AdaptiveEntropyCallback(verbose=0))

    return CallbackList(callbacks)


def create_model(env, config: dict):
    """
    創建 PPO 模型

    Args:
        env: 環境
        config: 配置字典

    Returns:
        PPO 模型
    """
    print("🧠 創建 PPO 模型...")

    learning_rate = config["learning_rate"]
    if isinstance(learning_rate, (tuple, list)) and len(learning_rate) == 2:
        learning_rate = make_linear_schedule(learning_rate[0], learning_rate[1])

    # 網絡架構配置
    policy_kwargs = dict(
        net_arch=dict(
            pi=[
                config["hidden_dim"],
                config["hidden_dim"],
                config["hidden_dim"],
            ],  # Actor 網絡
            vf=[
                config["hidden_dim"],
                config["hidden_dim"],
                config["hidden_dim"],
            ],  # Critic 網絡
        ),
        activation_fn=torch.nn.ReLU,
    )

    # 創建模型
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        # 學習參數
        learning_rate=learning_rate,
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        # PPO 參數
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        # 訓練效率
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        max_grad_norm=config["max_grad_norm"],
        # 日誌和設備
        verbose=config["verbose"],
        tensorboard_log=config["tensorboard_log"],
        device=config["device"],
    )

    return model


def get_training_config(target: str = "6666") -> dict:
    """
    獲取針對目標的訓練配置

    Args:
        target: 目標 ("6666" 或 "test")

    Returns:
        配置字典
    """
    base_config = {
        # 設備
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # 網絡架構
        "hidden_dim": 256,
        # 學習參數 (針對長期目標優化)
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        # PPO 參數
        "clip_range": 0.1,
        "ent_coef": 0.005,
        "vf_coef": 1.0,
        # 訓練效率
        "n_steps": 1024,
        "batch_size": 2048,
        "n_epochs": 10,
        "max_grad_norm": 0.3,
        # 日誌
        "verbose": 1,
        "tensorboard_log": "./logs/tensorboard/",
    }

    if target == "6666":
        # 針對 6666 分的配置
        config_6666 = base_config.copy()
        config_6666.update(
            {
                "learning_rate": (2e-4, 5e-5),  # 稍微提高初始學習率
                "ent_coef": 0.01,
                "vf_coef": 1.0,
                "n_steps": 4096,  # 增加 n_steps 讓每次更新看到更長軌跡 (2048 -> 4096)
                "batch_size": 8192,  # 增加 batch_size (4096 -> 8192)
                "n_epochs": 10,
                "hidden_dim": 512,  # 增加網絡容量 (256 -> 512)
            }
        )
        return config_6666

    elif target == "test":
        # 測試配置（快速驗證）
        config_test = base_config.copy()
        config_test.update(
            {
                "learning_rate": 2e-4,
                "ent_coef": 0.02,
                "n_steps": 512,
                "batch_size": 512,
                "n_epochs": 6,
                "verbose": 2,
            }
        )
        return config_test

    return base_config


def main():
    """主訓練函數"""
    parser = argparse.ArgumentParser(description="Game2048 SB3 訓練")
    parser.add_argument("--n-envs", type=int, default=32, help="並行環境數量")
    parser.add_argument(
        "--total-timesteps", type=int, default=5_000_000, help="總訓練步數"
    )
    parser.add_argument(
        "--target", type=str, default="6666", choices=["6666", "test"], help="訓練目標"
    )
    parser.add_argument(
        "--normalize",
        action=BooleanOptionalAction,
        default=True,
        help="啟用或停用 VecNormalize (預設啟用)",
    )
    parser.add_argument(
        "--norm-path",
        type=str,
        help="VecNormalize 統計檔案路徑（可用於載入/覆寫）",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="評估頻率（以 timesteps 為單位）",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=25_000,
        help="檢查點保存頻率",
    )
    parser.add_argument("--load", type=str, help="載入現有模型路徑")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument(
        "--finetune-lr",
        type=float,
        help="針對載入模型覆寫固定學習率 (僅在 --load 時生效)",
    )
    parser.add_argument(
        "--finetune-ent",
        type=float,
        help="針對載入模型覆寫熵係數 (僅在 --load 時生效)",
    )
    parser.add_argument(
        "--adaptive-entropy",
        action=BooleanOptionalAction,
        default=True,
        help="啟用自適應熵回調 (fine-tune 時可停用)",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default="none",
        choices=["none", "progressive"],
        help="指定訓練時期望使用的難度課程",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="自動嘗試載入最佳或最新的模型繼續訓練",
    )

    args = parser.parse_args()

    # 自動恢復邏輯
    if args.auto_resume and not args.load:
        candidates = [
            f"./models/ppo_game2048_{args.target}_final.zip",
            "./best_model/best_model.zip",
        ]
        for path in candidates:
            if os.path.exists(path):
                print(f"🔄 自動偵測到現有模型，準備恢復訓練: {path}")
                args.load = path
                # 嘗試尋找對應的正規化檔案
                norm_candidates = [
                    f"./models/vec_normalize_{args.target}.pkl",
                    path.replace(".zip", ".pkl"),
                    os.path.join(
                        os.path.dirname(path), f"vec_normalize_{args.target}.pkl"
                    ),
                ]
                for np_path in norm_candidates:
                    if os.path.exists(np_path):
                        print(f"   └── 發現正規化統計: {np_path}")
                        args.norm_path = np_path
                        break
                break

    # 設置隨機種子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("🎮 Game2048 SB3 訓練")
    print(f"🎯 目標: {args.target}")
    print(f"🚀 並行環境: {args.n_envs}")
    print(f"⏱️ 總步數: {args.total_timesteps:,}")
    print(f"🖥️ 設備: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    curriculum_phases = get_curriculum_phases(args.curriculum)
    if curriculum_phases:
        print(f"📈 啟用課程: {args.curriculum} ({len(curriculum_phases)} 階段)")

    if not args.normalize and args.norm_path:
        print("⚠️ 已停用 VecNormalize，忽略 --norm-path 參數。")

    # 創建環境
    env = create_envs(
        args.n_envs,
        normalize=args.normalize,
        training=True,
        norm_path=args.norm_path,
        seed=args.seed,
    )

    # 獲取配置
    config = get_training_config(args.target)

    norm_save_path = args.norm_path or f"./models/vec_normalize_{args.target}.pkl"

    # 創建或載入模型
    if args.load:
        print(f"📁 載入模型: {args.load}")
        model = PPO.load(args.load, env=env)
    else:
        model = create_model(env, config)

    apply_finetune_overrides(model, args.finetune_lr, args.finetune_ent)

    # 創建回調
    callbacks = create_callbacks(
        env,
        normalize=args.normalize,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        norm_path=args.norm_path,
        seed=args.seed,
        adaptive_entropy=args.adaptive_entropy,
        curriculum_phases=curriculum_phases,
    )

    # 開始訓練！
    print("🎯 開始訓練...")
    print("💡 提示: 開啟 TensorBoard 監控訓練進度")
    print("   tensorboard --logdir ./logs/tensorboard/")
    print("-" * 60)

    try:
        # 如果是載入模型，則不重置步數計數器，以保持 TensorBoard 曲線連續
        reset_timesteps = not bool(args.load)

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_timesteps,
        )

        # 保存最終模型
        final_path = f"./models/ppo_game2048_{args.target}_final.zip"
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        model.save(final_path)
        print(f"✅ 訓練完成！最終模型已保存到: {final_path}")

        # 如果使用 VecNormalize，保存正規化統計
        if args.normalize and hasattr(env, "save"):
            env.save(norm_save_path)
            print(f"✅ VecNormalize 統計已保存到: {norm_save_path}")

    except KeyboardInterrupt:
        print("\n⚠️ 訓練被中斷")
        # 保存中間結果
        interrupt_path = f"./models/ppo_game2048_{args.target}_interrupted.zip"
        os.makedirs(os.path.dirname(interrupt_path), exist_ok=True)
        model.save(interrupt_path)
        print(f"💾 中間結果已保存到: {interrupt_path}")
        if args.normalize and hasattr(env, "save"):
            env.save(norm_save_path)
            print(f"💾 VecNormalize 統計已保存到: {norm_save_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
