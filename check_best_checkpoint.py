"""查詢當前最佳檢查點信息"""

import json
import os

import torch

print("=" * 60)
print("🏆 當前最佳檢查點信息")
print("=" * 60)

# 1. 檢查 checkpoint_best.pt
best_checkpoint_path = "checkpoints/checkpoint_best.pt"
if os.path.exists(best_checkpoint_path):
    try:
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        print("\n📦 checkpoint_best.pt:")
        print(f"   訓練迭代: #{checkpoint.get('iteration', 'unknown')}")
        print(f"   平均獎勵: {checkpoint.get('mean_reward', 'N/A')}")
        print(f"   最高獎勵: {checkpoint.get('max_reward', 'N/A')}")
        print(f"   最低獎勵: {checkpoint.get('min_reward', 'N/A')}")

        # 檢查檔案時間
        import time

        mtime = os.path.getmtime(best_checkpoint_path)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        print(f"   更新時間: {time_str}")
    except Exception as e:
        print(f"   ❌ 讀取失敗: {e}")
else:
    print("\n❌ checkpoint_best.pt 不存在")

# 2. 查詢 scores.json 中的最高分
scores_path = "checkpoints/scores.json"
if os.path.exists(scores_path):
    try:
        with open(scores_path, "r", encoding="utf-8") as f:
            scores_data = json.load(f)

        if scores_data:
            # 找最高分
            best_score_entry = max(scores_data, key=lambda x: x.get("score", 0))

            print("\n🎮 遊戲最高分記錄 (scores.json):")
            print(f"   分數: {best_score_entry['score']}")
            print(f"   迭代: #{best_score_entry['iteration']}")
            print(f"   備註: {best_score_entry.get('note', 'N/A')}")

            # 找對應的 checkpoint
            best_iter = best_score_entry["iteration"]
            nearest_checkpoint_iter = (best_iter // 10) * 10
            checkpoint_file = f"checkpoints/checkpoint_{nearest_checkpoint_iter}.pt"

            print(f"\n💎 建議使用的檢查點:")
            print(f"   檔案: checkpoint_{nearest_checkpoint_iter}.pt")
            if os.path.exists(checkpoint_file):
                print(f"   狀態: ✅ 存在")
            else:
                print(f"   狀態: ❌ 不存在")

            # 統計最近表現
            print("\n📊 最近 20 局統計:")
            recent_20 = [entry["score"] for entry in scores_data[:20]]
            print(f"   平均: {sum(recent_20)/len(recent_20):.1f}")
            print(f"   最高: {max(recent_20)}")
            print(f"   最低: {min(recent_20)}")
            print(f"   ≥1000分: {len([s for s in recent_20 if s >= 1000])} 局")
            print(f"   500-999分: {len([s for s in recent_20 if 500 <= s < 1000])} 局")
            print(f"   <500分: {len([s for s in recent_20 if s < 500])} 局")

    except Exception as e:
        print(f"   ❌ 讀取失敗: {e}")
else:
    print("\n❌ scores.json 不存在")

print("\n" + "=" * 60)
print("✅ 查詢完成")
print("=" * 60)
