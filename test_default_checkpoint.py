"""
測試預設 checkpoint 選擇功能
"""

import os


# 測試 _get_default_checkpoint 邏輯
def test_default_checkpoint():
    # 優先使用最佳檢查點
    best_checkpoint = os.path.join("checkpoints", "checkpoint_best.pt")
    if os.path.exists(best_checkpoint):
        print(f"✓ 找到最佳檢查點: {best_checkpoint}")
        return best_checkpoint

    # 如果沒有最佳檢查點，找最新的
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("✗ checkpoints 目錄不存在")
        return None

    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_") and filename.endswith(".pt"):
            try:
                # 提取迭代數字
                iter_num = int(filename.replace("checkpoint_", "").replace(".pt", ""))
                checkpoints.append((iter_num, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                continue

    if checkpoints:
        # 返回最新的 checkpoint
        checkpoints.sort(reverse=True)
        print(f"✓ 找到最新檢查點: {checkpoints[0][1]} (迭代 {checkpoints[0][0]})")
        return checkpoints[0][1]

    print("✗ 沒有找到任何檢查點")
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("測試預設 Checkpoint 選擇")
    print("=" * 60)

    result = test_default_checkpoint()

    print("\n" + "=" * 60)
    if result:
        print(f"預設將使用: {os.path.basename(result)}")
        print("用戶仍可透過「瀏覽」按鈕選擇其他檔案")
    else:
        print("將從頭開始訓練")
    print("=" * 60)
