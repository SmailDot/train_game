"""
测试最佳检查点更新逻辑（模拟实际场景）
"""

import json
import sys
from pathlib import Path

# 设置输出编码
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def test_checkpoint_update_timing():
    """测试检查点更新的时序问题"""

    checkpoint_dir = Path("checkpoints")
    scores_file = checkpoint_dir / "scores.json"

    print("=" * 80)
    print("📊 测试检查点更新时序")
    print("=" * 80)

    if not scores_file.exists():
        print("❌ scores.json 不存在")
        return

    with open(scores_file, "r", encoding="utf-8") as f:
        scores_data = json.load(f)

    print(f"\n📁 找到 {len(scores_data)} 个分数记录")

    # 模拟场景：检查点在5280次迭代保存，但游戏分数在5283次
    test_checkpoint_iter = 5280

    print("\n🎯 模拟场景:")
    print(f"   检查点保存: 第 {test_checkpoint_iter} 次迭代")
    print("   (系统每10次迭代保存一次)")

    # 查找5280附近的分数（±20范围内）
    print(
        f"\n🔍 查找第 {test_checkpoint_iter-20} ~ {test_checkpoint_iter} 次迭代的分数:"
    )
    print("-" * 60)

    historical_best = float("-inf")
    recent_scores = []

    for entry in scores_data:
        score = entry.get("score", 0)
        iteration = entry.get("iteration", 0)

        # 历史最高分
        if score > historical_best:
            historical_best = score

        # 最近20次迭代内的分数
        if test_checkpoint_iter - 20 <= iteration <= test_checkpoint_iter:
            recent_scores.append((iteration, score))

    # 按迭代次数排序
    recent_scores.sort()

    if recent_scores:
        for iteration, score in recent_scores[-10:]:  # 显示最近10个
            print(f"   迭代 #{iteration}: {score} 分")

        recent_best_score = max(s[1] for s in recent_scores)
        recent_best_iter = [s[0] for s in recent_scores if s[1] == recent_best_score][0]

        print("\n📊 分析结果:")
        print(f"   历史最高分: {historical_best}")
        print(f"   最近最高分: {recent_best_score} (第 {recent_best_iter} 次)")

        if recent_best_score >= historical_best:
            print(
                f"\n✅ 应该更新 checkpoint_best.pt:"
                f"\n   - 使用检查点: checkpoint_{test_checkpoint_iter}.pt"
                f"\n   - 对应游戏回合: 第 {recent_best_iter} 次"
                f"\n   - 分数: {recent_best_score}"
            )
        else:
            print(
                f"\n⚠️  最近分数 ({recent_best_score}) "
                f"未超过历史最高 ({historical_best})，不需要更新"
            )
    else:
        print("   (无记录)")

    # 额外测试：查找1192分的记录
    print("\n🎮 查找1192分的记录:")
    print("-" * 60)
    found_1192 = False
    for entry in scores_data:
        if entry.get("score") == 1192:
            iteration = entry.get("iteration")
            print(f"   找到：第 {iteration} 次迭代")
            print(f"   最近的检查点: checkpoint_{(iteration // 10) * 10}.pt")
            found_1192 = True

    if not found_1192:
        print("   未找到1192分的记录")

    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_checkpoint_update_timing()
