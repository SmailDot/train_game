"""
Game2048 環境 - Stable-Baselines3 Gymnasium 兼容環境

將現有的 GameEnv 包裝成 Gymnasium 環境，讓 SB3 可以直接使用。
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game.environment import GameEnv


class Game2048Env(gym.Env):
    """
    Game2048 Gymnasium 環境包裝器

    將現有的 GameEnv 包裝成標準的 Gymnasium 環境接口，
    讓 Stable-Baselines3 可以直接使用。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        初始化環境

        Args:
            render_mode: 渲染模式 ("human" 或 "rgb_array")
            max_steps: 最大步數限制
            seed: 隨機種子
        """
        super().__init__()

        # 初始化遊戲環境
        self.game = GameEnv(seed=seed, max_steps=max_steps)
        self.render_mode = render_mode

        # 定義動作空間：離散動作 (0: 不跳, 1: 跳)
        self.action_space = spaces.Discrete(2)

        # 定義觀察空間：5 維連續狀態 (y, vy, x_obs, gap_top, gap_bottom)
        # 所有值都已經正規化到 [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # 追蹤資訊
        self.current_score = 0.0
        self.episode_length = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置環境到初始狀態

        Args:
            seed: 隨機種子
            options: 額外選項

        Returns:
            observation: 初始觀察
            info: 額外資訊
        """
        super().reset(seed=seed)

        if seed is not None:
            self.game.rng.seed(seed)

        # 重置遊戲
        obs = self.game.reset()
        self.current_score = 0.0
        self.episode_length = 0

        info = {"episode_score": 0.0, "episode_length": 0}

        return obs.astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        執行一個動作

        Args:
            action: 動作 (0: 不跳, 1: 跳)

        Returns:
            observation: 下一個觀察
            reward: 獎勵
            terminated: 是否結束 (死亡或通關)
            truncated: 是否被截斷 (到達最大步數)
            info: 額外資訊
        """
        # 確保動作是有效的
        action = int(action)

        # 執行動作
        obs, reward, terminated, info = self.game.step(action)

        # 更新追蹤資訊
        self.current_score += float(reward)
        self.episode_length += 1

        # 檢查是否通關
        win = info.get("win", False)

        # 更新 info
        info.update(
            {
                "episode_score": float(self.current_score),
                "episode_length": self.episode_length,
                "win": win,
            }
        )

        # terminated: 遊戲結束 (死亡或通關)
        # truncated: 從未使用 (我們的環境沒有步數限制)
        truncated = False

        return (obs.astype(np.float32), float(reward), terminated, truncated, info)

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Render using the requested mode (human or rgb_array)."""

        effective_mode = mode or self.render_mode or "human"
        return self.game.render(mode=effective_mode)

    def close(self):
        """關閉環境"""
        pass

    # Curriculum integration -------------------------------------------------
    def apply_difficulty_profile(self, profile: Dict[str, Any]) -> None:
        """Forward curriculum difficulty profiles to the underlying GameEnv."""

        apply_method = getattr(self.game, "apply_difficulty_profile", None)
        if callable(apply_method):
            apply_method(profile)

    # 兼容性方法
    def seed(self, seed: Optional[int] = None):
        """設置隨機種子 (向後兼容)"""
        if seed is not None:
            self.game.rng.seed(seed)
        return [seed]

    @property
    def unwrapped(self):
        """返回未包裝的環境"""
        return self

    def __str__(self):
        return f"Game2048Env(render_mode={self.render_mode})"

    def __repr__(self):
        return self.__str__()


# 創建向量化環境的輔助函數
def make_game2048_env(
    n_envs: int = 1,
    render_mode: Optional[str] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    創建 Game2048 環境 (單個或向量化)

    Args:
        n_envs: 環境數量
        render_mode: 渲染模式
        max_steps: 最大步數
        seed: 隨機種子

    Returns:
        環境實例
    """
    if n_envs == 1:
        return Game2048Env(render_mode=render_mode, max_steps=max_steps, seed=seed)
    else:
        # 對於多環境，使用 SubprocVecEnv
        from stable_baselines3.common.vec_env import SubprocVecEnv

        def make_env():
            def _init():
                env = Game2048Env(render_mode=render_mode, max_steps=max_steps)
                return env

            return _init

        # 使用 SubprocVecEnv 避免 Windows 問題
        return SubprocVecEnv([make_env() for _ in range(n_envs)])


if __name__ == "__main__":
    # 測試環境
    print("測試 Game2048Env...")

    # 創建環境
    env = Game2048Env()

    # 測試重置
    obs, info = env.reset()
    print(f"初始觀察形狀: {obs.shape}")
    print(f"初始觀察範圍: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"動作空間: {env.action_space}")
    print(f"觀察空間: {env.observation_space}")

    # 測試幾個步驟
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()  # 隨機動作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"步驟 {step + 1}: 獎勵={reward:.1f}, "
            f"累計={total_reward:.1f}, 結束={terminated}"
        )

        if terminated:
            print("遊戲結束！")
            break

    print("環境測試完成！")
    env.close()
