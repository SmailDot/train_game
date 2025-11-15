"""找出最佳回檔點 - 基於 scores.json 的實際數據"""

import json
import os
from datetime import datetime

print("=" * 80)
print("🔍 尋找最佳回檔點")
print("=" * 80)

# 讀取 scores.json（TOP 50 高分）
with open("checkpoints/scores.json", "r", encoding="utf-8") as f:
    scores_data = json.load(f)

# 按迭代排序找出時間線
by_time = sorted(scores_data, key=lambda x: x["iteration"], reverse=True)

print(f"\n📊 scores.json 中的數據:")
print(f"   總記錄: {len(scores_data)}")
print(f"   最高分: {max(e['score'] for e in scores_data)}")
print(f"   最低分: {min(e['score'] for e in scores_data)}")
print(f"   最新迭代: #{by_time[0]['iteration']}")
print(f"   最舊迭代: #{by_time[-1]['iteration']}")

# 找出最高分
best_score_entry = max(scores_data, key=lambda x: x["score"])
print(f"\n🏆 歷史最高分:")
print(f"   分數: {best_score_entry['score']}")
print(f"   迭代: #{best_score_entry['iteration']}")
print(f"   檔案: checkpoint_{(best_score_entry['iteration']//10)*10}.pt")

# 分析最近的表現（找出崩潰點）
print(f"\n⏰ 最近 20 局表現:")
for i, entry in enumerate(by_time[:20], 1):
    print(f"   {i:2d}. 迭代 #{entry['iteration']:5d}: {entry['score']:4d}分")

# 找出崩潰點（最後一個好成績）
last_good = by_time[0]
print(f"\n🎯 最後的好成績:")
print(f"   迭代: #{last_good['iteration']}")
print(f"   分數: {last_good['score']}")
print(f"   檢查點: checkpoint_{(last_good['iteration']//10)*10}.pt")

# 檢查是否存在
recommended_checkpoint = f"checkpoints/checkpoint_{(last_good['iteration']//10)*10}.pt"
if os.path.exists(recommended_checkpoint):
    size = os.path.getsize(recommended_checkpoint) / 1024
    mtime = datetime.fromtimestamp(os.path.getmtime(recommended_checkpoint))
    print(f"   狀態: ✅ 存在 ({size:.1f} KB, {mtime})")
else:
    print(f"   狀態: ❌ 不存在")

# 推薦策略
print(f"\n💡 回檔建議:")
print(f"\n選項 A（保守 - 推薦）:")
print(f"   回檔到: checkpoint_{(best_score_entry['iteration']//10)*10}.pt")
print(
    f"   理由: 歷史最高分 {best_score_entry['score']} 的迭代 #{best_score_entry['iteration']}"
)
print(f"   風險: 低，已證明能達到最高分")

print(f"\n選項 B（激進）:")
print(f"   回檔到: checkpoint_{(last_good['iteration']//10)*10}.pt")
print(f"   理由: 最新的好成績 {last_good['score']} 在迭代 #{last_good['iteration']}")
print(f"   風險: 中，可能接近崩潰點")

# 計算浪費
current_iter = 14464  # 當前最新
if last_good["iteration"] < current_iter:
    wasted = current_iter - last_good["iteration"]
    print(f"\n⚠️ 浪費分析:")
    print(f"   最後好成績: #{last_good['iteration']}")
    print(f"   當前迭代: #{current_iter}")
    print(f"   浪費了: {wasted} 次迭代")
    print(f"   浪費時間: 約 {wasted * 0.03:.0f} 分鐘 (~{wasted * 0.03 / 60:.1f} 小時)")

print(f"\n" + "=" * 80)
