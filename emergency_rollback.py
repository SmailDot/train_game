"""緊急回檔腳本"""

import os
import shutil
from datetime import datetime

print("=" * 80)
print("🚨 緊急回檔操作")
print("=" * 80)

# 找到最佳檢查點（5936 是最高分 1418）
best_iter = 5930  # 最接近 5936 的檢查點
source_checkpoint = f"checkpoints/checkpoint_{best_iter}.pt"
best_checkpoint = "checkpoints/checkpoint_best.pt"

if os.path.exists(source_checkpoint):
    # 備份當前的 checkpoint_best.pt
    if os.path.exists(best_checkpoint):
        backup = f"checkpoints/checkpoint_best_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        shutil.copy2(best_checkpoint, backup)
        print(f"✅ 備份當前 checkpoint_best.pt → {backup}")

    # 複製最佳檢查點
    shutil.copy2(source_checkpoint, best_checkpoint)
    print(f"✅ 回檔到 checkpoint_{best_iter}.pt")
    print(f"   （對應最高分 1418 的迭代 #5936）")

    # 刪除崩潰後的檢查點（7500+ 到 14460）
    print("\n🗑️ 清理崩潰後的檢查點...")
    deleted = 0
    for f in os.listdir("checkpoints"):
        if (
            f.startswith("checkpoint_")
            and f.endswith(".pt")
            and f != "checkpoint_best.pt"
        ):
            try:
                iter_num = int(f.replace("checkpoint_", "").replace(".pt", ""))
                if iter_num > 7500:  # 崩潰點
                    os.remove(os.path.join("checkpoints", f))
                    deleted += 1
                    if deleted <= 5:  # 只顯示前 5 個
                        print(f"   刪除: {f}")
            except:
                pass

    if deleted > 5:
        print(f"   ... 共刪除 {deleted} 個檔案")

    print("\n" + "=" * 80)
    print("✅ 回檔完成！")
    print("=" * 80)
    print("\n下次訓練將從迭代 #5930 開始")
    print("使用修復後的崩潰檢測系統")

else:
    print(f"❌ 找不到 {source_checkpoint}")
    print("請檢查檢查點文件")
