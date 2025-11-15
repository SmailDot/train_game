#!/usr/bin/env python3
"""
Game2048 SB3 測試腳本

載入訓練好的模型並測試性能，驗證是否能達到 6666 分通關。
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

from rl.game2048_env import Game2048Env


def test_model(
    model_path: str,
    n_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
    seed: int = 42,
):
    """
    測試模型性能

    Args:
        model_path: 模型路徑
        n_episodes: 測試回合數
        render: 是否渲染
        deterministic: 是否使用確定性策略
        seed: 隨機種子
    """
    print(f"🧪 測試模型: {model_path}")
    print(f"🎮 測試回合: {n_episodes}")
    print(f"🎯 確定性: {deterministic}")
    print("-" * 50)

    # 載入模型
    try:
        model = PPO.load(model_path)
        print("✅ 模型載入成功")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    # 創建環境
    env = Game2048Env(render_mode="human" if render else None, seed=seed)

    # 統計數據
    scores = []
    lengths = []
    wins = 0
    max_score = 0

    print("開始測試...")
    print()

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_score = 0
        episode_length = 0
        done = False

        while not done:
            # 預測動作
            action, _ = model.predict(obs, deterministic=deterministic)

            # 執行動作
            obs, reward, terminated, truncated, info = env.step(action)

            episode_score += reward
            episode_length += 1
            done = terminated or truncated

            if render:
                env.render()

        # 記錄統計
        scores.append(episode_score)
        lengths.append(episode_length)
        max_score = max(max_score, episode_score)

        # 檢查是否通關
        if info.get("win", False):
            wins += 1
            print(f"🎉 回合 {episode + 1:2d}: {episode_score:6.0f} 分 (通關!)")
        else:
            print(f"   回合 {episode + 1:2d}: {episode_score:6.0f} 分")

    env.close()

    # 輸出統計結果
    print()
    print("=" * 50)
    print("📊 測試結果統計")
    print("=" * 50)

    scores = np.array(scores)
    lengths = np.array(lengths)

    print(f"總回合數: {n_episodes}")
    print(f"平均分數: {scores.mean():.1f} ± {scores.std():.1f}")
    print(f"最高分數: {max_score:.0f}")
    print(f"最低分數: {scores.min():.0f}")
    print(f"平均長度: {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"通關次數: {wins}/{n_episodes} ({wins/n_episodes*100:.1f}%)")

    # 評估等級
    avg_score = scores.mean()
    win_rate = wins / n_episodes

    print()
    print("🎯 性能評估:")
    if avg_score >= 6000 and win_rate >= 0.8:
        print("🏆 優秀！可以穩定通關")
    elif avg_score >= 4000 and win_rate >= 0.5:
        print("👍 不錯！有機會通關")
    elif avg_score >= 2000:
        print("👌 良好！繼續訓練可以提升")
    elif avg_score >= 1000:
        print("📈 進步中！需要更多訓練")
    else:
        print("🎓 學習中！繼續訓練")

    return {
        "scores": scores,
        "lengths": lengths,
        "wins": wins,
        "max_score": max_score,
        "avg_score": scores.mean(),
        "win_rate": win_rate,
    }


def compare_models(model_paths: list, n_episodes: int = 5):
    """
    比較多個模型的性能

    Args:
        model_paths: 模型路徑列表
        n_episodes: 每個模型測試的回合數
    """
    print("🔄 比較模型性能")
    print("=" * 60)

    results = {}
    for path in model_paths:
        if os.path.exists(path):
            print(f"\n測試模型: {Path(path).name}")
            result = test_model(path, n_episodes, render=False, deterministic=True)
            if result:
                results[path] = result
        else:
            print(f"⚠️ 模型不存在: {path}")

    # 比較結果
    if results:
        print("\n" + "=" * 60)
        print("📊 模型比較結果")
        print("=" * 60)
        print(f"{'模型':<15} {'平均分':<8} {'最高分':<6} {'通關率':<8}")
        print("-" * 60)

        for path, result in results.items():
            name = Path(path).name
            win_rate_pct = result['win_rate'] * 100
            print(f"{name:<15} {result['avg_score']:<8.1f} {result['max_score']:<6.0f} {win_rate_pct:<8.1f}%")

    return results


def find_best_model(directory: str = "./best_model"):
    """
    找到最佳模型

    Args:
        directory: 模型目錄

    Returns:
        最佳模型路徑
    """
    if not os.path.exists(directory):
        print(f"⚠️ 目錄不存在: {directory}")
        return None

    # 查找 best_model.zip
    best_path = os.path.join(directory, "best_model.zip")
    if os.path.exists(best_path):
        return best_path

    # 查找其他模型文件
    model_files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    if model_files:
        # 按修改時間排序，取最新的
        model_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True
        )
        return os.path.join(directory, model_files[0])

    print(f"⚠️ 在 {directory} 中找不到模型文件")
    return None


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="Game2048 SB3 模型測試")
    parser.add_argument("--model", type=str, help="模型路徑")
    parser.add_argument("--episodes", type=int, default=10, help="測試回合數")
    parser.add_argument("--render", action="store_true", help="顯示遊戲畫面")
    parser.add_argument(
        "--stochastic", action="store_true", help="使用隨機策略（非確定性）"
    )
    parser.add_argument("--compare", nargs="+", help="比較多個模型")
    parser.add_argument("--find-best", action="store_true", help="自動查找最佳模型")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")

    args = parser.parse_args()

    # 設置隨機種子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("🎮 Game2048 SB3 模型測試")
    print("=" * 40)

    if args.compare:
        # 比較多個模型
        compare_models(args.compare, args.episodes)

    elif args.find_best:
        # 自動查找最佳模型
        best_model = find_best_model()
        if best_model:
            print(f"🎯 找到最佳模型: {best_model}")
            test_model(
                best_model, args.episodes, args.render, not args.stochastic, args.seed
            )
        else:
            print("❌ 找不到最佳模型")

    elif args.model:
        # 測試指定模型
        if os.path.exists(args.model):
            test_model(
                args.model, args.episodes, args.render, not args.stochastic, args.seed
            )
        else:
            print(f"❌ 模型不存在: {args.model}")

    else:
        # 預設行為：查找並測試最佳模型
        print("🔍 查找最佳模型...")
        best_model = find_best_model()
        if best_model:
            print(f"🎯 測試最佳模型: {best_model}")
            test_model(
                best_model, args.episodes, args.render, not args.stochastic, args.seed
            )
        else:
            print("❌ 找不到模型，請使用 --model 指定路徑")


if __name__ == "__main__":
    main()
