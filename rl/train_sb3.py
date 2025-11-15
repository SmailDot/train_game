#!/usr/bin/env python3
"""
Game2048 SB3 訓練腳本

使用 Stable-Baselines3 訓練 PPO 代理，目標是達到 6666 分通關。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

from rl.game2048_env import Game2048Env


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


def create_envs(n_envs: int = 32, normalize: bool = True):
    """
    創建向量化環境

    Args:
        n_envs: 環境數量
        normalize: 是否使用 VecNormalize

    Returns:
        環境實例
    """
    print(f"🚀 創建 {n_envs} 個並行環境...")

    # 創建基礎環境
    vec_env = make_vec_env(Game2048Env, n_envs=n_envs, env_kwargs={}, seed=42)

    # 添加監控
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    vec_env = VecMonitor(vec_env, log_dir)

    # 可選：添加正規化
    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.995,
        )

    return vec_env


def create_callbacks(eval_freq: int = 5000, save_freq: int = 10000):
    """
    創建訓練回調

    Args:
        eval_freq: 評估頻率
        save_freq: 保存頻率

    Returns:
        回調列表
    """
    callbacks = []

    # 檢查點回調
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./checkpoints/",
        name_prefix="ppo_game2048",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # 評估回調
    eval_env = make_vec_env(Game2048Env, n_envs=4)
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

    # 通關監控回調
    win_callback = WinCallback(verbose=1)
    callbacks.append(win_callback)

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
        learning_rate=config["learning_rate"],
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
        "learning_rate": 5e-5,  # 穩定但不太慢
        "gamma": 0.995,  # 高折扣因子（重視長期獎勵）
        "gae_lambda": 0.97,  # 高 GAE lambda
        # PPO 參數
        "clip_range": 0.15,  # 適中的 clip 範圍
        "ent_coef": 0.05,  # 高 entropy（探索）
        "vf_coef": 1.5,  # 強 critic 訓練
        # 訓練效率
        "n_steps": 2048,  # 每個環境收集 2048 步
        "batch_size": 512,  # 大 batch size
        "n_epochs": 15,  # 每次更新 15 輪
        "max_grad_norm": 0.5,
        # 日誌
        "verbose": 1,
        "tensorboard_log": "./logs/tensorboard/",
    }

    if target == "6666":
        # 針對 6666 分的配置
        config_6666 = base_config.copy()
        config_6666.update(
            {
                "learning_rate": 3e-5,  # 更慢但更穩定
                "ent_coef": 0.03,  # 稍微減少探索
                "vf_coef": 2.0,  # 更強的 critic
                "n_steps": 4096,  # 收集更多數據
                "batch_size": 1024,  # 更大的 batch
                "n_epochs": 20,  # 更多更新輪次
            }
        )
        return config_6666

    elif target == "test":
        # 測試配置（快速驗證）
        config_test = base_config.copy()
        config_test.update(
            {
                "learning_rate": 1e-4,  # 更快學習
                "ent_coef": 0.1,  # 更多探索
                "n_steps": 1024,  # 少量數據
                "batch_size": 256,  # 小 batch
                "n_epochs": 5,  # 少量更新
                "verbose": 2,  # 更多輸出
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
    parser.add_argument("--normalize", action="store_true", help="使用 VecNormalize")
    parser.add_argument("--load", type=str, help="載入現有模型路徑")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")

    args = parser.parse_args()

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

    # 創建環境
    env = create_envs(args.n_envs, args.normalize)

    # 獲取配置
    config = get_training_config(args.target)

    # 創建或載入模型
    if args.load:
        print(f"📁 載入模型: {args.load}")
        model = PPO.load(args.load, env=env)
    else:
        model = create_model(env, config)

    # 創建回調
    callbacks = create_callbacks()

    # 開始訓練！
    print("🎯 開始訓練...")
    print("💡 提示: 開啟 TensorBoard 監控訓練進度")
    print("   tensorboard --logdir ./logs/tensorboard/")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True
        )

        # 保存最終模型
        final_path = f"./models/ppo_game2048_{args.target}_final.zip"
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        model.save(final_path)
        print(f"✅ 訓練完成！最終模型已保存到: {final_path}")

        # 如果使用 VecNormalize，保存正規化統計
        if args.normalize and hasattr(env, "save"):
            norm_path = f"./models/vec_normalize_{args.target}.pkl"
            env.save(norm_path)
            print(f"✅ VecNormalize 統計已保存到: {norm_path}")

    except KeyboardInterrupt:
        print("\n⚠️ 訓練被中斷")
        # 保存中間結果
        interrupt_path = f"./models/ppo_game2048_{args.target}_interrupted.zip"
        os.makedirs(os.path.dirname(interrupt_path), exist_ok=True)
        model.save(interrupt_path)
        print(f"💾 中間結果已保存到: {interrupt_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
