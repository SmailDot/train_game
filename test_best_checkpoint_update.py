"""
测试最佳检查点自动更新机制
"""

import json
import sys
from pathlib import Path

import torch

# 设置输出编码
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def test_best_checkpoint_logic():
    """测试检查点更新逻辑"""

    checkpoint_dir = Path("checkpoints")
    scores_file = checkpoint_dir / "scores.json"
    best_checkpoint = checkpoint_dir / "checkpoint_best.pt"

    print("=" * 80)
    print("📊 测试最佳检查点更新逻辑")
    print("=" * 80)

    # 1. 读取 scores.json 中的所有分数
    if not scores_file.exists():
        print("❌ scores.json 不存在")
        return

    with open(scores_file, "r", encoding="utf-8") as f:
        scores_data = json.load(f)

    if not isinstance(scores_data, list):
        print("❌ scores.json 格式错误")
        return

    print(f"\n📁 找到 {len(scores_data)} 个迭代的分数记录")

    # 2. 找出历史最高分
    best_iter = None
    best_score = float("-inf")

    for entry in scores_data:
        score = entry.get("score", 0)
        iteration = entry.get("iteration", 0)
        if score > best_score:
            best_score = score
            best_iter = iteration

    if best_iter is None:
        print("❌ 找不到有效的分数记录")
        return

    print("\n🏆 历史最佳记录:")
    print(f"   迭代: #{best_iter}")
    print(f"   分数: {best_score}")

    # 3. 检查最佳检查点文件是否存在
    best_checkpoint_file = checkpoint_dir / f"checkpoint_{best_iter}.pt"
    if not best_checkpoint_file.exists():
        print(f"\n⚠️  最佳检查点文件不存在: checkpoint_{best_iter}.pt")
        print("   建议使用 rollback_tool.py 或 checkpoint_manager.py 恢复")
        return

    # 4. 检查 checkpoint_best.pt 是否存在且是否是最佳版本
    if best_checkpoint.exists():
        try:
            checkpoint = torch.load(best_checkpoint, map_location="cpu")
            current_best_iter = checkpoint.get("iteration", "unknown")
            current_best_score = checkpoint.get("mean_reward", "unknown")

            print("\n📦 当前 checkpoint_best.pt:")
            print(f"   迭代: #{current_best_iter}")
            print(f"   平均奖励: {current_best_score}")

            if str(current_best_iter) != str(best_iter):
                print("\n⚠️  checkpoint_best.pt 不是最新最佳版本！")
                print(f"   应该更新为迭代 #{best_iter} (分数: {best_score})")
        except Exception as e:
            print(f"\n❌ 读取 checkpoint_best.pt 失败: {e}")
    else:
        print("\n⚠️  checkpoint_best.pt 不存在")
        print(f"   应该创建并指向迭代 #{best_iter} (分数: {best_score})")

    # 5. 显示前 10 名检查点
    print("\n📊 前 10 名检查点:")
    print(f"{'迭代':>10} | {'分数':>10} | {'文件状态':>15}")
    print("-" * 40)

    # 按分数排序
    sorted_scores = sorted(scores_data, key=lambda x: x.get("score", 0), reverse=True)
    for i, entry in enumerate(sorted_scores[:10], 1):
        iteration = entry.get("iteration", "unknown")
        score = entry.get("score", 0)
        checkpoint_file = checkpoint_dir / f"checkpoint_{iteration}.pt"
        file_status = "✅ 存在" if checkpoint_file.exists() else "❌ 已删除"
        print(f"{iteration:>10} | {score:>10} | {file_status:>15}")

    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_best_checkpoint_logic()
