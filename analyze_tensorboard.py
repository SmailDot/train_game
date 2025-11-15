"""分析 TensorBoard 日誌找出崩潰原因"""

import os
import struct
from pathlib import Path

print("=" * 80)
print("🔍 分析 TensorBoard 日誌")
print("=" * 80)

tb_dir = Path("checkpoints/tb")

if not tb_dir.exists():
    print("❌ TensorBoard 目錄不存在")
    exit(1)

# 查找事件文件
event_files = list(tb_dir.glob("events.out.tfevents.*"))

print(f"\n📁 找到 {len(event_files)} 個事件文件")

for ef in event_files[:5]:  # 只顯示前5個
    size_mb = ef.stat().st_size / (1024 * 1024)
    print(f"   {ef.name}: {size_mb:.2f} MB")

# 檢查是否有最近的數據
if event_files:
    latest = max(event_files, key=lambda x: x.stat().st_mtime)
    size_mb = latest.stat().st_size / (1024 * 1024)
    print(f"\n最新文件: {latest.name}")
    print(f"大小: {size_mb:.2f} MB")
    print(f"修改時間: {latest.stat().st_mtime}")

    # 嘗試讀取一些統計數據
    print(f"\n💡 建議:")
    print(f"   1. 使用 TensorBoard 查看: tensorboard --logdir=checkpoints/tb")
    print(f"   2. 查找 reward/mean, loss/total 的趨勢")
    print(f"   3. 尋找迭代 #7436 附近的異常")

print("\n" + "=" * 80)
