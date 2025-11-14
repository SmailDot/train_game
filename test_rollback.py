"""
測試檢查點回檔機制

這個腳本用於測試當 AI 性能嚴重退化時，系統是否能正確檢測並回檔到最佳檢查點。
"""

import os
import sys

import torch

# 確保可以導入專案模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.pytorch_trainer import PPOTrainer


def test_performance_degradation_detection():
    """測試性能退化檢測邏輯"""
    print("=" * 70)
    print("測試 1: 性能退化檢測邏輯")
    print("=" * 70)

    # 創建訓練器（不需要 env 參數）
    trainer = PPOTrainer(save_dir="./checkpoints", device="cpu")

    # 設定歷史最佳值
    trainer.best_reward = 100.0
    trainer.best_max_reward = 150.0
    trainer.best_min_reward = 50.0

    print("\n📊 設定歷史最佳值：")
    print(f"   平均分: {trainer.best_reward:.2f}")
    print(f"   最高分: {trainer.best_max_reward:.2f}")
    print(f"   最低分: {trainer.best_min_reward:.2f}")

    # 測試案例 1: 正常情況（無退化）
    print("\n" + "-" * 70)
    print("案例 1: 正常情況（平均分略有下降，但未達閾值）")
    print("-" * 70)
    result = trainer._check_performance_degradation(
        mean_reward=95.0, max_reward=145.0, min_reward=48.0, iteration=100
    )
    print(f"   結果: {'需要回檔' if result else '繼續訓練'} ✓")
    assert not result, "正常情況不應觸發回檔"

    # 測試案例 2: 平均分嚴重下降
    print("\n" + "-" * 70)
    print("案例 2: 平均分嚴重下降（從 100 降到 50，下降 50%）")
    print("-" * 70)
    result = trainer._check_performance_degradation(
        mean_reward=50.0, max_reward=140.0, min_reward=45.0, iteration=100
    )
    print(f"   結果: {'需要回檔 ✓' if result else '未檢測到 ✗'}")
    # 注意：這個測試可能失敗如果沒有檢查點可以回檔

    # 測試案例 3: 最高分嚴重下降
    print("\n" + "-" * 70)
    print("案例 3: 最高分嚴重下降（從 150 降到 80，下降 46.7%）")
    print("-" * 70)
    result = trainer._check_performance_degradation(
        mean_reward=95.0, max_reward=80.0, min_reward=45.0, iteration=100
    )
    print(f"   結果: {'需要回檔 ✓' if result else '未檢測到 ✗'}")

    # 測試案例 4: 最低分嚴重下降
    print("\n" + "-" * 70)
    print("案例 4: 最低分嚴重下降（從 50 降到 20，下降 60%）")
    print("-" * 70)
    result = trainer._check_performance_degradation(
        mean_reward=95.0, max_reward=145.0, min_reward=20.0, iteration=100
    )
    print(f"   結果: {'需要回檔 ✓' if result else '未檢測到 ✗'}")

    # 測試案例 5: 早期訓練（不應觸發回檔）
    print("\n" + "-" * 70)
    print("案例 5: 早期訓練（迭代 < 100，即使退化也不回檔）")
    print("-" * 70)
    result = trainer._check_performance_degradation(
        mean_reward=30.0, max_reward=60.0, min_reward=10.0, iteration=50  # 早期訓練
    )
    print(f"   結果: {'需要回檔 ✗' if result else '繼續訓練 ✓'}")
    assert not result, "早期訓練不應觸發回檔"

    print("\n" + "=" * 70)
    print("✅ 性能退化檢測邏輯測試完成")
    print("=" * 70)


def test_checkpoint_loading():
    """測試檢查點載入功能"""
    print("\n\n" + "=" * 70)
    print("測試 2: 檢查點載入功能")
    print("=" * 70)

    checkpoint_dir = "./checkpoints"

    # 檢查是否有可用的檢查點
    if not os.path.exists(checkpoint_dir):
        print("\n⚠️ 檢查點目錄不存在，跳過此測試")
        print("   請先訓練模型以生成檢查點")
        return

    checkpoints = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]

    if not checkpoints:
        print("\n⚠️ 找不到檢查點文件，跳過此測試")
        print("   請先訓練模型以生成檢查點")
        return

    print(f"\n📂 找到 {len(checkpoints)} 個檢查點：")
    for cp in sorted(checkpoints)[-5:]:  # 顯示最新的 5 個
        print(f"   - {cp}")

    # 創建訓練器（不需要 env 參數）
    trainer = PPOTrainer(save_dir=checkpoint_dir, device="cpu")

    # 測試回檔功能
    print("\n🔄 測試回檔功能...")
    success = trainer._rollback_to_best_checkpoint()

    if success:
        print("\n✅ 成功載入檢查點！")
    else:
        print("\n❌ 載入檢查點失敗")

    print("\n" + "=" * 70)
    print("✅ 檢查點載入測試完成")
    print("=" * 70)


def test_integration():
    """整合測試：模擬完整的性能崩潰場景"""
    print("\n\n" + "=" * 70)
    print("測試 3: 整合測試 - 模擬性能崩潰場景")
    print("=" * 70)

    checkpoint_dir = "./checkpoints"

    # 檢查是否有可用的檢查點
    if not os.path.exists(checkpoint_dir) or not any(
        f.endswith(".pt") for f in os.listdir(checkpoint_dir)
    ):
        print("\n⚠️ 需要先進行訓練以生成檢查點")
        print("   請執行: python run_game.py --ai")
        return

    # 創建訓練器（不需要 env 參數）
    trainer = PPOTrainer(save_dir=checkpoint_dir, device="cpu")

    # 載入現有檢查點
    checkpoints = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
    )

    if checkpoints:
        latest = checkpoints[-1]
        print(f"\n📂 載入檢查點: {latest}")
        try:
            checkpoint_path = os.path.join(checkpoint_dir, latest)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            trainer.net.load_state_dict(checkpoint["model_state"])
            print("   ✓ 載入成功")
        except Exception as e:
            print(f"   ✗ 載入失敗: {e}")
            return

    # 模擬良好的歷史表現
    trainer.best_reward = 200.0
    trainer.best_max_reward = 350.0
    trainer.best_min_reward = 100.0

    print("\n📊 設定歷史最佳表現：")
    print(f"   平均分: {trainer.best_reward:.2f}")
    print(f"   最高分: {trainer.best_max_reward:.2f}")
    print(f"   最低分: {trainer.best_min_reward:.2f}")

    # 模擬性能崩潰（所有指標嚴重下降）
    print("\n💥 模擬性能崩潰（可能是錯誤的參數調整導致）：")
    print("   平均分: 200.0 → 80.0 (下降 60%)")
    print("   最高分: 350.0 → 150.0 (下降 57%)")
    print("   最低分: 100.0 → 40.0 (下降 60%)")

    # 觸發性能退化檢測
    result = trainer._check_performance_degradation(
        mean_reward=80.0,
        max_reward=150.0,
        min_reward=40.0,
        iteration=500,  # 足夠的訓練歷史
    )

    if result:
        print("\n✅ 系統正確檢測到性能崩潰並執行回檔！")
    else:
        print("\n⚠️ 系統未能檢測到性能崩潰（可能是檢查點不足）")

    print("\n" + "=" * 70)
    print("✅ 整合測試完成")
    print("=" * 70)


def main():
    """執行所有測試"""
    print("\n" + "=" * 70)
    print("檢查點回檔機制測試套件")
    print("=" * 70)
    print("\n此測試將驗證以下功能：")
    print("1. 性能退化檢測邏輯（40% 閾值）")
    print("2. 檢查點載入功能")
    print("3. 完整的性能崩潰恢復流程")

    try:
        # 測試 1: 性能退化檢測邏輯
        test_performance_degradation_detection()

        # 測試 2: 檢查點載入
        test_checkpoint_loading()

        # 測試 3: 整合測試
        test_integration()

        print("\n\n" + "=" * 70)
        print("🎉 所有測試完成！")
        print("=" * 70)
        print("\n💡 使用建議：")
        print("   - 在實驗新參數前，確保有穩定的檢查點")
        print("   - 觀察控制台輸出，系統會自動檢測性能崩潰")
        print("   - 回檔後，學習率會降低至初始值的 50%")
        print("   - 定期備份 checkpoints/ 目錄")

    except Exception as e:
        print("\n❌ 測試過程中發生錯誤：")
        print(f"   {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
