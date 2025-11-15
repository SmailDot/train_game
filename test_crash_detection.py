"""測試崩潰檢測邏輯的修復"""

import json

import numpy as np


def test_crash_detection_logic():
    """測試崩潰檢測是否正確使用時間順序"""

    # 讀取 scores.json
    with open("checkpoints/scores.json", "r", encoding="utf-8") as f:
        scores_data = json.load(f)

    print("=" * 70)
    print("🔍 測試崩潰檢測邏輯")
    print("=" * 70)

    # 1. 顯示原始數據（按分數排序）
    print("\n📊 原始 scores.json（按分數排序）：")
    print("   前 10 個條目：")
    for i, entry in enumerate(scores_data[:10], 1):
        print(f"   {i}. 迭代 #{entry['iteration']:5d} - 分數: {entry['score']:4d}")

    # 2. 按迭代次數重新排序（時間順序）
    scores_by_iteration = sorted(
        scores_data, key=lambda x: x.get("iteration", 0), reverse=True
    )

    print("\n⏰ 按時間排序（迭代次數從大到小）：")
    print("   最近 10 個條目：")
    for i, entry in enumerate(scores_by_iteration[:10], 1):
        print(f"   {i}. 迭代 #{entry['iteration']:5d} - 分數: {entry['score']:4d}")

    # 3. 快速檢測測試
    print("\n🚨 快速檢測（最近 10 局）：")
    recent_10_scores = [entry.get("score", 0) for entry in scores_by_iteration[:10]]
    recent_10_mean = np.mean(recent_10_scores)
    recent_10_max = np.max(recent_10_scores)
    recent_10_min = np.min(recent_10_scores)

    print(f"   平均: {recent_10_mean:.1f}")
    print(f"   最高: {recent_10_max}")
    print(f"   最低: {recent_10_min}")
    print(f"   觸發極端崩潰? {recent_10_max < 200} (閾值: <200)")

    # 4. 趨勢分析測試
    if len(scores_by_iteration) >= 20:
        print("\n📈 趨勢分析（最近 20 局）：")
        recent_20_scores = [entry.get("score", 0) for entry in scores_by_iteration[:20]]

        recent_10 = np.mean(recent_20_scores[:10])  # 最近 10 局
        previous_10 = np.mean(recent_20_scores[10:20])  # 之前 10 局

        print(f"   最近10局平均: {recent_10:.1f}")
        print(f"   之前10局平均: {previous_10:.1f}")

        if previous_10 > 0:
            ratio = recent_10 / previous_10
            drop = (previous_10 - recent_10) / previous_10
            print(f"   比例: {ratio:.2f} ({drop*100:+.1f}%)")
            print(f"   觸發趨勢警告? {ratio < 0.67} (閾值: <0.67，下降>33%)")

    # 5. 深度檢測測試
    print("\n🔍 深度檢測（最近 50 局 vs 歷史最佳）：")
    recent_window = min(50, len(scores_by_iteration))
    recent_scores = [
        entry.get("score", 0) for entry in scores_by_iteration[:recent_window]
    ]

    recent_mean = np.mean(recent_scores)
    recent_max = np.max(recent_scores)

    # 歷史最佳（使用原始按分數排序的數據）
    top_20_percent = max(10, len(scores_data) // 5)
    historical_best_scores = [
        entry.get("score", 0) for entry in scores_data[:top_20_percent]
    ]
    historical_mean = np.mean(historical_best_scores)
    historical_max = np.max(historical_best_scores)

    mean_drop = (
        (historical_mean - recent_mean) / historical_mean if historical_mean > 0 else 0
    )
    max_drop = (
        (historical_max - recent_max) / historical_max if historical_max > 0 else 0
    )

    print(f"   最近{recent_window}局平均: {recent_mean:.1f}")
    print(f"   歷史最佳平均: {historical_mean:.1f}")
    print(f"   平均分下降: {mean_drop*100:.1f}%")
    print(f"   最高分下降: {max_drop*100:.1f}%")
    print(f"   最近平均: {recent_mean:.1f}")

    is_catastrophic = mean_drop > 0.60 and max_drop > 0.50 and recent_mean < 500
    print(f"\n   觸發崩潰回檔? {is_catastrophic}")
    print(f"   條件: mean_drop > 60% AND max_drop > 50% AND recent_mean < 500")
    print(
        f"   實際: {mean_drop*100:.1f}% > 60% AND {max_drop*100:.1f}% > 50% AND {recent_mean:.1f} < 500"
    )

    print("\n" + "=" * 70)
    print("✅ 測試完成")
    print("=" * 70)


if __name__ == "__main__":
    test_crash_detection_logic()
