"""
创建最佳检查点 (checkpoint_best.pt) 从现有最好的检查点
"""

import json
import shutil
import sys
from pathlib import Path

import torch

# 设置输出编码
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def create_best_checkpoint():
    """从现有检查点中创建 checkpoint_best.pt"""

    checkpoint_dir = Path("checkpoints")
    scores_file = checkpoint_dir / "scores.json"
    best_checkpoint = checkpoint_dir / "checkpoint_best.pt"

    print("=" * 80)
    print("💎 创建最佳检查点 (checkpoint_best.pt)")
    print("=" * 80)

    # 1. 读取 scores.json
    if not scores_file.exists():
        print("❌ scores.json 不存在")
        return

    with open(scores_file, "r", encoding="utf-8") as f:
        scores_data = json.load(f)

    if not isinstance(scores_data, list):
        print("❌ scores.json 格式错误")
        return

    print(f"\n📁 找到 {len(scores_data)} 个迭代的分数记录")

    # 2. 找出历史最高分并检查文件是否存在
    candidates = []
    for entry in scores_data:
        score = entry.get("score", 0)
        iteration = entry.get("iteration", 0)
        checkpoint_file = checkpoint_dir / f"checkpoint_{iteration}.pt"
        if checkpoint_file.exists():
            candidates.append((iteration, score, checkpoint_file))

    if not candidates:
        print("\n❌ 没有找到任何可用的检查点文件")
        return

    # 按分数排序，取最高的
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_iter, best_score, best_file = candidates[0]

    print("\n🏆 现存最佳检查点:")
    print(f"   迭代: #{best_iter}")
    print(f"   分数: {best_score}")
    print(f"   文件: {best_file.name}")

    # 3. 复制为 checkpoint_best.pt
    try:
        shutil.copy2(best_file, best_checkpoint)
        print("\n✅ 成功创建 checkpoint_best.pt")
        print(f"   来源: checkpoint_{best_iter}.pt (分数: {best_score})")

        # 4. 验证文件
        checkpoint = torch.load(best_checkpoint, map_location="cpu")
        print("\n📦 验证检查点内容:")
        print(f"   模型参数: {'✅' if 'model_state' in checkpoint else '❌'}")
        print(f"   优化器状态: {'✅' if 'optimizer_state' in checkpoint else '❌'}")
        if "iteration" in checkpoint:
            print(f"   记录迭代: #{checkpoint['iteration']}")

    except Exception as e:
        print(f"\n❌ 创建失败: {e}")
        return

    # 5. 显示前 5 名（现存文件）
    print("\n📊 现存前 5 名检查点:")
    print(f"{'迭代':>10} | {'分数':>10} | {'状态':>15}")
    print("-" * 40)

    for i, (iteration, score, file) in enumerate(candidates[:5], 1):
        status = "💎 当前best" if iteration == best_iter else "✅ 可用"
        print(f"{iteration:>10} | {score:>10} | {status:>15}")

    print("\n" + "=" * 80)
    print("✅ 完成")
    print("=" * 80)
    print("\n提示:")
    print("  - checkpoint_best.pt 会在每次打破记录时自动更新")
    print("  - 性能崩溃时会优先回档到 checkpoint_best.pt")
    print("  - UI 载入模型时也会优先尝试 checkpoint_best.pt")


if __name__ == "__main__":
    create_best_checkpoint()
