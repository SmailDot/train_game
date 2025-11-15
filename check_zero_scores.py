"""檢查 0 分問題"""

import json
import os

# 讀取 scores.json
with open("checkpoints/scores.json", "r", encoding="utf-8") as f:
    scores_data = json.load(f)

# 按迭代排序
by_iter = sorted(scores_data, key=lambda x: x["iteration"], reverse=True)

print("=" * 80)
print("🚨 檢查 0 分崩潰")
print("=" * 80)

print(f"\n總記錄: {len(scores_data)}")
print(f"最新迭代: #{by_iter[0]['iteration']}")
print(f"最新分數: {by_iter[0]['score']}")

# 檢查最近 50 局
print("\n最近 50 局:")
for i, entry in enumerate(by_iter[:50], 1):
    score = entry["score"]
    symbol = "❌" if score <= 0 else "✅"
    print(f"{i:2d}. #{entry['iteration']:5d}: {score:4d}分 {symbol}")

# 統計 0 分和負分
zero_or_negative = [e for e in scores_data if e["score"] <= 0]
print(f"\n⚠️ 0 分或負分記錄數: {len(zero_or_negative)}")

if zero_or_negative:
    print("\n所有 0 分/負分記錄:")
    for entry in sorted(zero_or_negative, key=lambda x: x["iteration"]):
        print(f"  #{entry['iteration']:5d}: {entry['score']:4d}分")

# 檢查什麼時候開始出現問題
print("\n分析:")
for i in range(len(by_iter) - 1):
    if by_iter[i]["score"] <= 100 and by_iter[i + 1]["score"] > 500:
        print(
            f"性能崩潰點: 迭代 #{by_iter[i+1]['iteration']} ({by_iter[i+1]['score']}分) → #{by_iter[i]['iteration']} ({by_iter[i]['score']}分)"
        )
        break
