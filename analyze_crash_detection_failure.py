"""分析為什麼崩潰檢測沒有觸發"""

import json

import numpy as np

with open("checkpoints/scores.json", "r", encoding="utf-8") as f:
    scores_data = json.load(f)

# 按時間排序
scores_by_iteration = sorted(
    scores_data, key=lambda x: x.get("iteration", 0), reverse=True
)

print("=" * 80)
print("🔍 分析崩潰檢測為什麼沒有觸發")
print("=" * 80)

print(f"\nscores.json 中的記錄數: {len(scores_data)}")
print(f"最新迭代: #{scores_by_iteration[0]['iteration']}")
print(f"最新分數: {scores_by_iteration[0]['score']}")

# 關鍵問題：scores.json 是否包含最近的 0 分記錄？
print(f"\n⚠️ 問題分析:")
print(f"   你看到的: 迭代 #14400+, 回合 #59612+ 都是 0 分")
print(f"   scores.json 最新: 迭代 #{scores_by_iteration[0]['iteration']}")
print(f"   差距: {14400 - scores_by_iteration[0]['iteration']} 次迭代")

print(f"\n🚨 結論:")
print(f"   scores.json 只保留最高的 50 個分數！")
print(f"   0 分的記錄不會被加入 scores.json（因為分數太低）")
print(f"   所以崩潰檢測根本看不到這些 0 分！")

print(f"\n📊 scores.json 中最低分:")
lowest = min(e["score"] for e in scores_data)
print(f"   {lowest} 分")

print(f"\n💡 這就是為什麼:")
print(f"   1. AI 從迭代 7436 開始崩潰")
print(f"   2. 一直訓練到迭代 14400+")
print(f"   3. 但 scores.json 還停在 7436（最後一個好成績）")
print(f"   4. 崩潰檢測讀取 scores.json，看到的都是 1000+ 的好成績")
print(f"   5. 所以判斷「一切正常」，沒有觸發回檔！")

print(f"\n❌ 致命缺陷:")
print(f"   scores.json 的排行榜機制（只保留 TOP 50）")
print(f"   導致崩潰後的低分被完全忽略！")
