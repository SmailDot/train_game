"""
檢查點管理工具

功能：
1. 查看所有檢查點及其對應分數
2. 清理舊的低分檢查點
3. 保留最佳檢查點
"""

import json
import os
from datetime import datetime


def analyze_checkpoints():
    """分析檢查點和分數記錄"""
    checkpoint_dir = "checkpoints"
    scores_file = os.path.join(checkpoint_dir, "scores.json")

    # 獲取所有檢查點檔案
    checkpoints = {}
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            try:
                if f == "checkpoint_best.pt":
                    checkpoints["best"] = os.path.join(checkpoint_dir, f)
                else:
                    iteration = int(f.replace("checkpoint_", "").replace(".pt", ""))
                    checkpoints[iteration] = os.path.join(checkpoint_dir, f)
            except ValueError:
                continue

    # 讀取分數記錄
    scores_by_iteration = {}
    if os.path.exists(scores_file):
        try:
            with open(scores_file, "r", encoding="utf-8") as f:
                scores = json.load(f)
                for record in scores:
                    it = record.get("iteration")
                    score = record.get("score", 0)
                    if it:
                        if (
                            it not in scores_by_iteration
                            or score > scores_by_iteration[it]
                        ):
                            scores_by_iteration[it] = score
        except Exception:
            pass

    return checkpoints, scores_by_iteration


def display_checkpoint_status():
    """顯示檢查點狀態"""
    checkpoints, scores = analyze_checkpoints()

    print("=" * 80)
    print("📊 檢查點狀態分析")
    print("=" * 80)

    # 合併檢查點和分數信息
    data = []
    for it, path in checkpoints.items():
        if it == "best":
            score = "N/A (最佳)"
            file_size = os.path.getsize(path) / 1024  # KB
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            data.append(
                {
                    "iteration": "BEST",
                    "score": score,
                    "size_kb": file_size,
                    "modified": mtime,
                    "path": path,
                }
            )
        else:
            score = scores.get(it, "未知")
            file_size = os.path.getsize(path) / 1024  # KB
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            data.append(
                {
                    "iteration": it,
                    "score": score,
                    "size_kb": file_size,
                    "modified": mtime,
                    "path": path,
                }
            )

    # 按分數排序（未知分數放最後）
    def sort_key(x):
        if x["iteration"] == "BEST":
            return (0, 99999)
        score = x["score"]
        if isinstance(score, str):
            return (2, 0)  # 未知分數
        return (1, -score)  # 已知分數，降序

    data.sort(key=sort_key)

    print(f"\n共 {len(data)} 個檢查點:\n")
    print(f"{'迭代':>8} | {'分數':>8} | {'大小(KB)':>10} | {'修改時間':>20}")
    print("-" * 80)

    for item in data:
        it = item["iteration"]
        score = item["score"]
        size = item["size_kb"]
        mtime = item["modified"].strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(score, int):
            score_str = f"{score:,}"
        else:
            score_str = str(score)

        print(f"{str(it):>8} | {score_str:>8} | {size:>10.1f} | {mtime:>20}")

    return data


def clean_low_score_checkpoints(threshold=300, keep_count=10):
    """清理低分檢查點"""
    checkpoints, scores = analyze_checkpoints()

    print(f"\n{'='*80}")
    print(f"🗑️  清理低分檢查點 (閾值: {threshold} 分)")
    print(f"{'='*80}")

    # 找出要刪除的檢查點
    to_delete = []
    to_keep = []

    for it, path in checkpoints.items():
        if it == "best":
            to_keep.append((it, path, "最佳"))
            continue

        score = scores.get(it, 0)

        # 保留最近的 N 個檢查點（無論分數）
        recent_iterations = sorted(
            [i for i in checkpoints.keys() if isinstance(i, int)], reverse=True
        )[:keep_count]

        if it in recent_iterations:
            to_keep.append((it, path, f"{score} (最近)"))
        elif score < threshold:
            to_delete.append((it, path, score))
        else:
            to_keep.append((it, path, score))

    if not to_delete:
        print("\n✅ 沒有需要清理的低分檢查點")
        return

    print(f"\n將刪除 {len(to_delete)} 個低分檢查點:")
    for it, path, score in to_delete:
        print(f"   ❌ 迭代 {it:5d} | 分數 {score:4d}")

    print(f"\n將保留 {len(to_keep)} 個檢查點:")
    for it, path, reason in to_keep[:10]:
        print(f"   ✅ 迭代 {str(it):>5} | {reason}")

    confirm = (
        input(f"\n確認刪除這 {len(to_delete)} 個檢查點? (yes/NO): ").strip().lower()
    )

    if confirm != "yes":
        print("\n❌ 取消清理")
        return

    # 執行刪除
    deleted_count = 0
    for it, path, score in to_delete:
        try:
            os.remove(path)
            print(f"   ✅ 已刪除: checkpoint_{it}.pt")
            deleted_count += 1
        except Exception as e:
            print(f"   ❌ 刪除失敗: checkpoint_{it}.pt - {e}")

    print(f"\n✅ 清理完成！刪除了 {deleted_count} 個檢查點")


def create_best_checkpoint_from_existing():
    """從現有檢查點中找出最佳的，複製為 checkpoint_best.pt"""
    checkpoints, scores = analyze_checkpoints()

    if not scores:
        print("\n⚠️  找不到分數記錄")
        return

    # 找出最高分
    best_iteration = max(scores.items(), key=lambda x: x[1])
    best_it, best_score = best_iteration

    print(f"\n{'='*80}")
    print("💎 創建最佳檢查點")
    print(f"{'='*80}")
    print(f"\n最佳表現: 迭代 {best_it}, 分數 {best_score}")

    # 檢查該檢查點是否存在
    if best_it not in checkpoints:
        print(f"\n❌ 檢查點檔案不存在: checkpoint_{best_it}.pt")
        return

    source_path = checkpoints[best_it]
    dest_path = "checkpoints/checkpoint_best.pt"

    try:
        import shutil

        shutil.copy2(source_path, dest_path)
        print(f"\n✅ 已創建最佳檢查點: {dest_path}")
        print(f"   來源: checkpoint_{best_it}.pt (分數: {best_score})")
    except Exception as e:
        print(f"\n❌ 創建失敗: {e}")


def main():
    print("=" * 80)
    print("🔧 檢查點管理工具")
    print("=" * 80)

    while True:
        print("\n選項:")
        print("  1. 查看檢查點狀態")
        print("  2. 清理低分檢查點")
        print("  3. 創建最佳檢查點 (checkpoint_best.pt)")
        print("  0. 退出")

        choice = input("\n請選擇 (0-3): ").strip()

        if choice == "0":
            print("\n再見！")
            break
        elif choice == "1":
            display_checkpoint_status()
        elif choice == "2":
            display_checkpoint_status()
            clean_low_score_checkpoints()
        elif choice == "3":
            create_best_checkpoint_from_existing()
        else:
            print("\n❌ 無效的選擇")


if __name__ == "__main__":
    main()
