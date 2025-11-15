"""檢查訓練後的詳細狀態"""

import json
import os
from datetime import datetime

print("=" * 80)
print("🔍 訓練後詳細分析")
print("=" * 80)

# 讀取 scores.json
with open("checkpoints/scores.json", "r", encoding="utf-8") as f:
    scores_data = json.load(f)

print(f"\n📊 總記錄數: {len(scores_data)}")

# 按時間排序（最近的在前）
scores_by_time = sorted(scores_data, key=lambda x: x.get("iteration", 0), reverse=True)

# 按分數排序（最高的在前）
scores_by_score = sorted(scores_data, key=lambda x: x.get("score", 0), reverse=True)

print("\n🏆 歷史最高分 TOP 5:")
for i, entry in enumerate(scores_by_score[:5], 1):
    print(f"   {i}. 迭代 #{entry['iteration']:5d} - {entry['score']:4d}分")

print("\n⏰ 最近 20 局表現:")
recent_20 = scores_by_time[:20]
for i, entry in enumerate(recent_20, 1):
    print(f"   {i:2d}. 迭代 #{entry['iteration']:5d} - {entry['score']:4d}分")

# 統計分析
recent_scores = [e["score"] for e in recent_20]
import numpy as np

print(f"\n📈 最近 20 局統計:")
print(f"   平均: {np.mean(recent_scores):.1f}")
print(f"   中位數: {np.median(recent_scores):.1f}")
print(f"   標準差: {np.std(recent_scores):.1f}")
print(f"   最高: {np.max(recent_scores)}")
print(f"   最低: {np.min(recent_scores)}")

# 分段統計
count_1000_plus = sum(1 for s in recent_scores if s >= 1000)
count_500_999 = sum(1 for s in recent_scores if 500 <= s < 1000)
count_below_500 = sum(1 for s in recent_scores if s < 500)

print(f"\n📊 分數分布:")
print(f"   ≥1000分: {count_1000_plus} 局 ({count_1000_plus/20*100:.0f}%)")
print(f"   500-999分: {count_500_999} 局 ({count_500_999/20*100:.0f}%)")
print(f"   <500分: {count_below_500} 局 ({count_below_500/20*100:.0f}%)")

# 檢查最新迭代
latest_iteration = scores_by_time[0]["iteration"]
print(f"\n🔢 訓練進度:")
print(f"   最新迭代: #{latest_iteration}")
print(f"   歷史最高分迭代: #{scores_by_score[0]['iteration']}")

# 檢查是否有 0 分
zero_scores = [e for e in scores_data if e["score"] == 0]
if zero_scores:
    print(f"\n⚠️ 發現 {len(zero_scores)} 個 0 分記錄:")
    for entry in zero_scores[:10]:  # 只顯示前 10 個
        print(f"   迭代 #{entry['iteration']} - 0分")
else:
    print(f"\n✅ 無 0 分記錄")

# 檢查最近 50 局的趨勢
if len(scores_by_time) >= 50:
    recent_50 = [e["score"] for e in scores_by_time[:50]]
    first_25 = np.mean(recent_50[:25])
    second_25 = np.mean(recent_50[25:50])

    print(f"\n📈 趨勢分析（最近 50 局）:")
    print(f"   最近25局平均: {first_25:.1f}")
    print(f"   之前25局平均: {second_25:.1f}")

    if first_25 > second_25:
        change = (first_25 - second_25) / second_25 * 100
        print(f"   趨勢: ⬆️ 上升 {change:.1f}%")
    elif first_25 < second_25:
        change = (second_25 - first_25) / second_25 * 100
        print(f"   趨勢: ⬇️ 下降 {change:.1f}%")
    else:
        print(f"   趨勢: ➡️ 持平")

# 檢查檢查點文件
print(f"\n💾 檢查點狀態:")
checkpoint_best = "checkpoints/checkpoint_best.pt"
if os.path.exists(checkpoint_best):
    size = os.path.getsize(checkpoint_best) / 1024
    mtime = datetime.fromtimestamp(os.path.getmtime(checkpoint_best))
    print(f"   checkpoint_best.pt: ✅ 存在 ({size:.1f} KB, {mtime})")
else:
    print(f"   checkpoint_best.pt: ❌ 不存在")

# 檢查最新的檢查點
checkpoints = sorted(
    [
        f
        for f in os.listdir("checkpoints")
        if f.startswith("checkpoint_")
        and f.endswith(".pt")
        and f != "checkpoint_best.pt"
    ]
)
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    latest_path = os.path.join("checkpoints", latest_checkpoint)
    size = os.path.getsize(latest_path) / 1024
    mtime = datetime.fromtimestamp(os.path.getmtime(latest_path))
    print(f"   最新檢查點: {latest_checkpoint} ({size:.1f} KB, {mtime})")

print("\n" + "=" * 80)
print("✅ 分析完成")
print("=" * 80)
