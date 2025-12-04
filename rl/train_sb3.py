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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from stable_baselines3.common.logger import (
    CSVOutputFormat,
    KVWriter,
    Logger,
    TensorBoardOutputFormat,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _import_env():
    from rl.game2048_env import Game2048Env as _Game2048Env

    return _Game2048Env


Game2048Env = _import_env()


# --- Custom Logger for Chinese Support and Alignment ---

KEY_TRANSLATIONS = {
    # Environment
    "env/alignment_score": "env/alignment_score(對齊分數)",
    "env/passed_count": "env/passed_count(通過障礙物數量)",
    "env/scroll_speed": "env/scroll_speed(目前捲動速度)",
    "env/win_rate": "env/win_rate(通關率)",
    # Rollout
    "rollout/ep_len_mean": "rollout/ep_len_mean(平均回合長度)",
    "rollout/ep_rew_mean": "rollout/ep_rew_mean(平均回合獎勵)",
    # Time
    "time/fps": "time/fps(幀率)",
    "time/iterations": "time/iterations(迭代次數)",
    "time/time_elapsed": "time/time_elapsed(經過時間)",
    "time/total_timesteps": "time/total_timesteps(總步數)",
    # Train
    "train/approx_kl": "train/approx_kl(近似KL散度)",
    "train/clip_fraction": "train/clip_fraction(更新幅度過大比例)",
    "train/clip_range": "train/clip_range(更新幅度限制)",
    "train/entropy_coef": "train/entropy_coef(熵係數)",
    "train/entropy_loss": "train/entropy_loss(熵損失)",
    "train/explained_variance": "train/explained_variance(價值預測準確度)",
    "train/learning_rate": "train/learning_rate(學習率)",
    "train/loss": "train/loss(總損失)",
    "train/n_updates": "train/n_updates(更新次數)",
    "train/policy_gradient_loss": "train/policy_gradient_loss(策略梯度損失)",
    "train/value_loss": "train/value_loss(價值損失)",
}


def get_visual_width(s: str) -> int:
    """Calculate visual width of a string (Chinese chars = 2)."""
    width = 0
    for char in s:
        if "\u4e00" <= char <= "\u9fff" or "\uff00" <= char <= "\uffef":
            width += 2
        else:
            width += 1
    return width


class ChineseHumanOutputFormat(KVWriter):
    """Custom output format that handles Chinese character alignment correctly."""

    def __init__(self, file):
        self.file = file

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        # Create strings for printing
        kv_list = []
        for k, v in sorted(key_values.items()):
            # Ignore exclusion to ensure we print everything we have
            # if k in key_excluded: continue

            # Format value
            if isinstance(v, float):
                val_str = f"{v:.3g}"
            else:
                val_str = str(v)

            # Split key into English and Chinese parts
            # Assuming format: "english_key(chinese_translation)"
            if "(" in k and k.endswith(")"):
                parts = k.split("(")
                eng_part = parts[0]
                chn_part = "(" + parts[1]
            else:
                eng_part = k
                chn_part = ""

            kv_list.append((eng_part, chn_part, val_str))

        if not kv_list:
            return

        # Calculate max width for key column
        # We want: English Left <spaces> Chinese Right
        # So max_key_width = max(visual_width(eng) + visual_width(chn)) + padding
        max_key_width = 0
        for eng, chn, _ in kv_list:
            width = get_visual_width(eng) + get_visual_width(chn)
            if width > max_key_width:
                max_key_width = width

        # Add some minimum padding between English and Chinese
        max_key_width += 2

        max_val_len = max(get_visual_width(v) for _, _, v in kv_list)

        # Print separator
        # Format: | key_column | val |
        # Width: 2 + max_key_width + 3 + max_val_len + 2
        dash_len = max_key_width + max_val_len + 7
        self.file.write("-" * dash_len + "\n")

        for eng, chn, v in kv_list:
            # Key formatting: English + spaces + Chinese
            current_key_width = get_visual_width(eng) + get_visual_width(chn)
            padding_len = max_key_width - current_key_width
            key_str = f"{eng}{' ' * padding_len}{chn}"

            val_padding = " " * (max_val_len - get_visual_width(v))
            self.file.write(f"| {key_str} | {v}{val_padding} |\n")

        self.file.write("-" * dash_len + "\n")
        self.file.flush()


class ChineseLogger(Logger):
    """Logger that translates keys to Chinese before recording."""

    def record(
        self,
        key: str,
        value: Any,
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        # Translate key if possible
        translated_key = KEY_TRANSLATIONS.get(key, key)
        super().record(translated_key, value, exclude)


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

    def __init__(
        self,
        prefix: str = "env",
        verbose: int = 0,
        window_size: int = 100,
        target_win_rate: Optional[float] = None,
    ):
        super().__init__(verbose)
        self.prefix = prefix
        self.win_buffer = deque(maxlen=window_size)
        self.pass_buffer = deque(maxlen=window_size)
        self.target_win_rate = target_win_rate

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if not infos or dones is None:
            return True

        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                # Buffer the win status (True/False -> 1.0/0.0)
                self.win_buffer.append(float(info.get("win", False)))

                if "passed_count" in info:
                    self.pass_buffer.append(float(info["passed_count"]))

        # Record the mean of the buffers (Rolling Average)
        if self.win_buffer:
            current_win_rate = np.mean(self.win_buffer)
            self.logger.record(f"{self.prefix}/win_rate", current_win_rate)

            # Check for target win rate stop condition
            if (
                self.target_win_rate is not None
                and len(self.win_buffer) >= self.win_buffer.maxlen
                and current_win_rate >= self.target_win_rate
            ):
                if self.verbose > 0:
                    print(
                        f"🎉 達成目標通關率！當前: {current_win_rate:.2f} "
                        f">= 目標: {self.target_win_rate:.2f}"
                    )
                return False  # Stop training

        if self.pass_buffer:
            self.logger.record(f"{self.prefix}/passed_count", np.mean(self.pass_buffer))

        # Optional: Log instantaneous metrics for debugging if needed,
        # but for TensorBoard, the rolling average is much better.

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

    env_kwargs = {}
    if render_mode:
        env_kwargs["render_mode"] = render_mode
        # 如果啟用渲染，強制使用單一環境與 DummyVecEnv 以避免視窗衝突與崩潰
        n_envs = 1
        vec_env_cls = DummyVecEnv
        print(
            "⚠️ 啟用渲染模式：強制將環境數量設為 1 並使用 DummyVecEnv 以避免視窗衝突。"
        )
    else:
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    print(f"🚀 創建 {n_envs} 個並行環境 (Class: {vec_env_cls.__name__})...")

    vec_env = make_vec_env(
        Game2048Env,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        seed=seed,
        vec_env_cls=vec_env_cls,
    )

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
    target_win_rate: Optional[float] = None,
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
    callbacks.append(EpisodeStatsCallback(verbose=0, target_win_rate=target_win_rate))
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
        "gamma": 0.98,  # 稍微降低 gamma，讓 AI 更專注於眼前的閃避 (0.99 -> 0.98)
        "gae_lambda": 0.95,
        # PPO 參數
        "clip_range": 0.1,
        "ent_coef": 0.005,
        "vf_coef": 1.0,
        # 訓練效率
        "n_steps": 2048,  # 恢復標準步數
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
                "n_steps": 2048,
                "batch_size": 4096,
                "n_epochs": 10,
                "hidden_dim": 256,  # 縮小網絡容量 (512 -> 256) 以避免過擬合並加快收斂
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
    parser.add_argument(
        "--render",
        action="store_true",
        help="啟用渲染模式 (注意：這會開啟大量視窗，僅用於調試)",
    )
    parser.add_argument(
        "--target-win-rate",
        type=float,
        help="當通關率達到此值時停止訓練 (例如 0.9)",
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
        render_mode="human" if args.render else None,
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

    # 配置自定義 Logger (支援中文與對齊)
    log_dir = "./logs/tensorboard/"
    output_formats = [
        ChineseHumanOutputFormat(sys.stdout),
        TensorBoardOutputFormat(log_dir),
        CSVOutputFormat(os.path.join(log_dir, "progress.csv")),
    ]
    custom_logger = ChineseLogger(folder=log_dir, output_formats=output_formats)
    model.set_logger(custom_logger)

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
        target_win_rate=args.target_win_rate,
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
