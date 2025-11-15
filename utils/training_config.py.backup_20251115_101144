"""
PPO 訓練優化配置
針對 RTX 3060 Ti 優化的超參數和訓練策略
"""

import os
from typing import Any, Dict, Optional

# RTX 3060 Ti 優化配置
RTX_3060TI_CONFIG = {
    "device": "cuda",  # 使用 GPU
    "batch_size": 256,  # 增大 batch size 利用 GPU
    "ppo_epochs": 10,  # 增加 PPO 更新次數
    "lr": 2.5e-4,  # 降低學習率確保穩定
    "gamma": 0.99,  # 折扣因子
    "lam": 0.95,  # GAE lambda
    "clip_eps": 0.2,  # PPO clip 範圍
    "vf_coef": 0.5,  # Value function 係數
    "ent_coef": 0.01,  # 降低 entropy 鼓勵更確定的策略
    "max_grad_norm": 0.5,  # 梯度裁剪
    "horizon": 4096,  # 增加 rollout 長度
}

# CPU 訓練配置（較保守）
CPU_CONFIG = {
    "device": "cpu",
    "batch_size": 64,
    "ppo_epochs": 4,
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.05,
    "max_grad_norm": 0.5,
    "horizon": 2048,
}

# 改進的獎勵塑造
REWARD_SHAPING_CONFIG = {
    "pass_obstacle": 10.0,  # 增加通過獎勵
    "collision": -10.0,  # 增加碰撞懲罰
    "survive_step": 0.1,  # 每步存活小獎勵
    "height_penalty": 0.05,  # 懲罰過高或過低
    "forward_progress": 0.2,  # 鼓勵前進
}


class TrainingConfig:
    """訓練配置管理"""

    def __init__(self, use_gpu: bool = True):
        self.config = RTX_3060TI_CONFIG if use_gpu else CPU_CONFIG
        self.reward_config = REWARD_SHAPING_CONFIG

    def get_ppo_kwargs(self) -> Dict[str, Any]:
        """獲取 PPO 訓練器參數"""
        return {
            "device": self.config["device"],
            "batch_size": self.config["batch_size"],
            "ppo_epochs": self.config["ppo_epochs"],
            "lr": self.config["lr"],
            "gamma": self.config["gamma"],
            "lam": self.config["lam"],
            "clip_eps": self.config["clip_eps"],
            "vf_coef": self.config["vf_coef"],
            "ent_coef": self.config["ent_coef"],
        }

    def get_training_params(self) -> Dict[str, Any]:
        """獲取訓練參數"""
        return {
            "horizon": self.config["horizon"],
            "max_grad_norm": self.config["max_grad_norm"],
        }

    def should_use_vectorized_env(self) -> bool:
        """是否應該使用向量化環境"""
        return self.config["device"] == "cuda"

    def get_recommended_n_envs(self) -> int:
        """推薦的並行環境數量"""
        if self.config["device"] == "cuda":
            return 8  # GPU 可以處理更多
        return 4  # CPU 較少


def list_available_checkpoints(algorithm: str = "ppo") -> list:
    """列出可用的 checkpoint 檔案"""
    if algorithm.lower() == "ppo":
        checkpoint_dir = "checkpoints"
    else:
        checkpoint_dir = f"checkpoints_{algorithm.lower()}"

    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_") and file.endswith(".pt"):
            try:
                iteration = int(file.replace("checkpoint_", "").replace(".pt", ""))
                full_path = os.path.join(checkpoint_dir, file)
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                checkpoints.append(
                    {
                        "file": file,
                        "path": full_path,
                        "iteration": iteration,
                        "size_mb": size_mb,
                    }
                )
            except ValueError:
                continue

    # 按迭代次數排序
    checkpoints.sort(key=lambda x: x["iteration"], reverse=True)
    return checkpoints


def get_latest_checkpoint(algorithm: str = "ppo") -> Optional[str]:
    """獲取最新的 checkpoint"""
    checkpoints = list_available_checkpoints(algorithm)
    if checkpoints:
        return checkpoints[0]["path"]
    return None


def print_training_summary(config: TrainingConfig):
    """打印訓練配置摘要"""
    print("\n" + "=" * 60)
    print("🚀 PPO 訓練配置")
    print("=" * 60)
    print(f"設備: {config.config['device'].upper()}")
    print(f"批次大小: {config.config['batch_size']}")
    print(f"PPO 更新次數: {config.config['ppo_epochs']}")
    print(f"學習率: {config.config['lr']}")
    print(f"Horizon: {config.config['horizon']}")
    print(f"推薦並行環境數: {config.get_recommended_n_envs()}")
    print("=" * 60)
    print("獎勵塑造:")
    for key, value in config.reward_config.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")
