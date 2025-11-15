"""
根據分析創建完整的修復和回檔方案
"""

import json
import os
import shutil
from datetime import datetime

print("=" * 80)
print("🔧 執行完整修復和回檔")
print("=" * 80)

# === 第一步：確定回檔目標 ===
print("\n第一步：確定回檔目標")
print("-" * 80)

# 讀取 scores.json 找出最佳點
with open("checkpoints/scores.json", "r", encoding="utf-8") as f:
    scores_data = json.load(f)

best_entry = max(scores_data, key=lambda x: x["score"])
best_checkpoint_iter = (best_entry["iteration"] // 10) * 10

print(f"🏆 選擇回檔到歷史最高分:")
print(f"   分數: {best_entry['score']}")
print(f"   迭代: #{best_entry['iteration']}")
print(f"   檢查點: checkpoint_{best_checkpoint_iter}.pt")

source_file = f"checkpoints/checkpoint_{best_checkpoint_iter}.pt"
if not os.path.exists(source_file):
    print(f"❌ 錯誤：找不到 {source_file}")
    exit(1)

# === 第二步：備份當前狀態 ===
print(f"\n第二步：備份當前狀態")
print("-" * 80)

backup_dir = (
    f"checkpoints/backup/crash_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
os.makedirs(backup_dir, exist_ok=True)

# 備份 checkpoint_best.pt
if os.path.exists("checkpoints/checkpoint_best.pt"):
    shutil.copy2("checkpoints/checkpoint_best.pt", f"{backup_dir}/checkpoint_best.pt")
    print(f"✅ 備份 checkpoint_best.pt")

# 備份 scores.json
shutil.copy2("checkpoints/scores.json", f"{backup_dir}/scores.json")
print(f"✅ 備份 scores.json")

# 備份 training_meta.json（如果存在）
if os.path.exists("checkpoints/training_meta.json"):
    shutil.copy2("checkpoints/training_meta.json", f"{backup_dir}/training_meta.json")
    print(f"✅ 備份 training_meta.json")

# === 第三步：清理崩潰後的檢查點 ===
print(f"\n第三步：清理崩潰後的檢查點")
print("-" * 80)

# 刪除迭代 > 7500 的檢查點（崩潰後的）
deleted_count = 0
deleted_size = 0

for filename in os.listdir("checkpoints"):
    if (
        filename.startswith("checkpoint_")
        and filename.endswith(".pt")
        and filename != "checkpoint_best.pt"
    ):
        try:
            iter_num = int(filename.replace("checkpoint_", "").replace(".pt", ""))
            if iter_num > 7500:
                filepath = os.path.join("checkpoints", filename)
                size = os.path.getsize(filepath)
                os.remove(filepath)
                deleted_count += 1
                deleted_size += size
        except:
            pass

print(f"✅ 刪除 {deleted_count} 個崩潰後的檢查點")
print(f"   釋放空間: {deleted_size / (1024*1024):.1f} MB")

# === 第四步：設置新的 checkpoint_best.pt ===
print(f"\n第四步：設置新的 checkpoint_best.pt")
print("-" * 80)

shutil.copy2(source_file, "checkpoints/checkpoint_best.pt")
print(f"✅ 設置 checkpoint_best.pt = checkpoint_{best_checkpoint_iter}.pt")

# === 第五步：重置訓練元數據 ===
print(f"\n第五步：重置訓練元數據")
print("-" * 80)

meta_file = "checkpoints/training_meta.json"
if os.path.exists(meta_file):
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 更新迭代次數
    meta["iteration"] = best_checkpoint_iter
    meta["last_rollback"] = datetime.now().isoformat()
    meta["rollback_reason"] = f"Manual rollback to best score {best_entry['score']}"

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 更新 training_meta.json")

# === 總結 ===
print(f"\n" + "=" * 80)
print(f"✅ 修復完成")
print(f"=" * 80)

print(f"\n📊 統計:")
print(f"   回檔到迭代: #{best_checkpoint_iter}")
print(f"   預期分數: ~{best_entry['score']}")
print(f"   刪除檢查點: {deleted_count} 個")
print(f"   備份位置: {backup_dir}")

print(f"\n下一步:")
print(f"   1. ✅ 修復已完成")
print(f"   2. 重新啟動訓練: python run_game.py")
print(f"   3. 監控 training_history.json（新增的完整歷史）")
print(f"   4. 崩潰檢測現在會看到所有分數（包括0分）")

print(f"\n⚠️ 重要:")
print(f"   - 已修復 scores.json TOP 50 截斷問題")
print(f"   - 新增 training_history.json 保存完整歷史")
print(f"   - 崩潰檢測優先讀取完整歷史")
print(f"   - 下次崩潰會在 10-50 局內被檢測到")

print(f"\n" + "=" * 80)
