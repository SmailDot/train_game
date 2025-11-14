"""
手動回檔到最佳檢查點

當訓練崩潰後，使用此工具手動回檔到歷史最佳表現的檢查點
"""

import json
import os
import shutil
from datetime import datetime


def find_best_checkpoint():
    """從 scores.json 找出最佳檢查點"""
    scores_file = "checkpoints/scores.json"

    if not os.path.exists(scores_file):
        print("❌ 找不到 scores.json 檔案")
        return None

    try:
        with open(scores_file, "r", encoding="utf-8") as f:
            scores = json.load(f)

        if not scores:
            print("❌ scores.json 是空的")
            return None

        # 排序找出最高分
        best = max(scores, key=lambda x: x.get("score", 0))

        return best

    except Exception as e:
        print(f"❌ 讀取 scores.json 失敗: {e}")
        return None


def list_recent_best_checkpoints():
    """列出近期表現最好的檢查點"""
    scores_file = "checkpoints/scores.json"

    if not os.path.exists(scores_file):
        return []

    try:
        with open(scores_file, "r", encoding="utf-8") as f:
            scores = json.load(f)

        # 找出分數 > 500 的檢查點，並按分數排序
        good_checkpoints = [s for s in scores if s.get("score", 0) > 500]
        good_checkpoints.sort(key=lambda x: x.get("score", 0), reverse=True)

        return good_checkpoints[:15]  # 返回前 15 個

    except Exception:
        return []


def rollback_to_checkpoint(iteration):
    """回檔到指定的檢查點"""
    checkpoint_path = f"checkpoints/checkpoint_{iteration}.pt"

    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到檢查點: {checkpoint_path}")
        return False

    # 備份當前最新的檢查點
    checkpoint_dir = "checkpoints"
    all_checkpoints = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
    )

    if all_checkpoints:
        latest = all_checkpoints[-1]
        latest_path = os.path.join(checkpoint_dir, latest)

        # 創建備份目錄
        backup_dir = "checkpoints/backup"
        os.makedirs(backup_dir, exist_ok=True)

        # 備份最新檢查點
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{latest}.backup_{timestamp}")

        try:
            shutil.copy2(latest_path, backup_path)
            print(f"✅ 已備份最新檢查點到: {backup_path}")
        except Exception as e:
            print(f"⚠️  備份失敗: {e}")

    # 複製目標檢查點為最新
    try:
        # 找出當前最大的迭代次數
        max_iteration = 0
        for f in all_checkpoints:
            try:
                it = int(f.replace("checkpoint_", "").replace(".pt", ""))
                max_iteration = max(max_iteration, it)
            except ValueError:
                continue

        # 創建新的檢查點（迭代次數 +10）
        new_iteration = max_iteration + 10
        new_checkpoint_path = f"checkpoints/checkpoint_{new_iteration}.pt"

        shutil.copy2(checkpoint_path, new_checkpoint_path)
        print(f"✅ 已回檔到迭代 #{iteration}")
        print(f"✅ 新檢查點: checkpoint_{new_iteration}.pt")

        return True

    except Exception as e:
        print(f"❌ 回檔失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("🔄 手動回檔工具")
    print("=" * 70)

    # 顯示最佳檢查點
    best = find_best_checkpoint()

    if best:
        print("\n🏆 歷史最佳表現:")
        print(f"   分數: {best['score']}")
        print(f"   迭代: {best['iteration']}")
        print(f"   備註: {best.get('note', 'N/A')}")

    # 列出近期表現好的檢查點
    print("\n📊 表現優秀的檢查點 (分數 > 500):")
    print(f"{'='*70}")

    good_checkpoints = list_recent_best_checkpoints()

    if not good_checkpoints:
        print("⚠️  找不到表現優秀的檢查點")
    else:
        for i, cp in enumerate(good_checkpoints[:10], 1):
            iteration = cp["iteration"]
            score = cp["score"]
            checkpoint_exists = os.path.exists(f"checkpoints/checkpoint_{iteration}.pt")
            status = "✅" if checkpoint_exists else "❌ (檔案不存在)"
            print(f"{i:2d}. 迭代 {iteration:5d} | 分數 {score:4d} {status}")

    # 詢問是否回檔
    print(f"\n{'='*70}")
    print("⚠️  回檔操作說明:")
    print("   1. 會備份當前最新的檢查點")
    print("   2. 複製指定的歷史檢查點為新的檢查點")
    print("   3. 下次啟動訓練時會自動載入新檢查點")
    print(f"{'='*70}\n")

    if best:
        default_iteration = best["iteration"]
        user_input = input(
            f"請輸入要回檔的迭代次數 (Enter=使用最佳 {default_iteration}，0=取消): "
        ).strip()

        if user_input == "0":
            print("\n❌ 取消回檔")
            return

        if user_input == "":
            target_iteration = default_iteration
        else:
            try:
                target_iteration = int(user_input)
            except ValueError:
                print("❌ 無效的輸入")
                return

        # 執行回檔
        print(f"\n🔄 開始回檔到迭代 #{target_iteration}...")
        success = rollback_to_checkpoint(target_iteration)

        if success:
            print(f"\n{'='*70}")
            print("✅ 回檔完成！")
            print(f"{'='*70}")
            print("\n下一步:")
            print("1. 啟動遊戲: python run_game.py")
            print("2. 選擇 AI 模式開始訓練")
            print("3. 系統會自動載入回檔後的檢查點")
            print("\n💡 提示:")
            print("   - 回檔後學習率會重置")
            print("   - 建議檢查 training_config.json 的參數設定")
            print("   - 觀察訓練曲線，確認沒有再次崩潰")
        else:
            print(f"\n{'='*70}")
            print("❌ 回檔失敗")
            print(f"{'='*70}")


if __name__ == "__main__":
    main()
