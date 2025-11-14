"""
測試權重載入功能
驗證 .pt 檔案是否正確載入到模型中
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.pytorch_trainer import PPOTrainer


def test_weight_loading():
    """測試權重載入是否正確"""
    print("=" * 70)
    print("測試權重載入功能")
    print("=" * 70)

    # 檢查檢查點目錄
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("\n❌ 檢查點目錄不存在")
        return False

    # 尋找最新的檢查點
    checkpoints = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
    )

    if not checkpoints:
        print("\n❌ 找不到檢查點檔案")
        return False

    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    print(f"\n📂 測試檔案: {latest_checkpoint}")

    # 創建兩個訓練器進行比較
    print("\n1️⃣ 創建第一個訓練器（載入前）...")
    # trainer1 用於展示多實例場景
    _ = PPOTrainer(save_dir=checkpoint_dir, device="cpu")

    print("2️⃣ 創建第二個訓練器（載入後）...")
    trainer2 = PPOTrainer(save_dir=checkpoint_dir, device="cpu")

    # 獲取載入前的權重
    print("\n3️⃣ 記錄載入前的權重...")
    weights_before = {}
    for name, param in trainer2.net.named_parameters():
        weights_before[name] = param.data.clone()
        print(f"   {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")

    # 載入檢查點
    print(f"\n4️⃣ 載入檢查點: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("   ✓ 檢查點載入成功")
        print(f"   ✓ 檢查點包含的鍵: {list(checkpoint.keys())}")

        if "model_state" in checkpoint:
            print("\n5️⃣ 載入模型權重...")
            trainer2.net.load_state_dict(checkpoint["model_state"])
            print("   ✓ 模型權重載入成功")
        else:
            print("   ❌ 檢查點中沒有 'model_state'")
            return False

    except Exception as e:
        print(f"   ❌ 載入失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 驗證權重是否改變
    print("\n6️⃣ 驗證權重是否改變...")
    weights_changed = False
    total_params = 0
    changed_params = 0

    for name, param in trainer2.net.named_parameters():
        total_params += 1
        before = weights_before[name]
        after = param.data

        # 計算差異
        diff = torch.abs(after - before).sum().item()

        if diff > 1e-6:
            weights_changed = True
            changed_params += 1
            print(f"   ✓ {name}: 權重已改變 (差異: {diff:.6f})")
            print(f"      載入前: mean={before.mean():.6f}, std={before.std():.6f}")
            print(f"      載入後: mean={after.mean():.6f}, std={after.std():.6f}")
        else:
            print(f"   ⚠️  {name}: 權重未改變")

    print("\n" + "=" * 70)
    if weights_changed:
        print("✅ 權重載入成功！")
        print(f"   {changed_params}/{total_params} 個參數層的權重已改變")
        return True
    else:
        print("❌ 權重載入失敗！所有權重都未改變")
        print("   這可能是因為：")
        print("   1. 檢查點檔案損壞")
        print("   2. 載入的是初始化的權重")
        print("   3. load_state_dict() 沒有正確執行")
        return False


def test_ui_loading_logic():
    """測試 UI 中的載入邏輯"""
    print("\n\n" + "=" * 70)
    print("測試 UI 載入邏輯")
    print("=" * 70)

    checkpoint_dir = "./checkpoints"

    # 模擬 UI 中的載入邏輯
    print("\n📋 模擬 UI 載入流程...")

    trainer = PPOTrainer(save_dir=checkpoint_dir, device="cpu")

    def _load_model(path: str) -> bool:
        """模擬 UI 中的 _load_model 函數"""
        try:
            state = torch.load(path, map_location=trainer.device)
            print(f"   ✓ 載入檔案: {path}")
            print(f"   ✓ 檔案類型: {type(state)}")

            if isinstance(state, dict):
                print(f"   ✓ 檔案是字典，鍵: {list(state.keys())}")

                model_state = state.get("model_state", state)
                print(f"   ✓ 取得 model_state (類型: {type(model_state)})")

                # 記錄載入前的權重
                first_param_before = next(iter(trainer.net.parameters())).data.clone()

                trainer.net.load_state_dict(model_state)
                print("   ✓ 執行 load_state_dict()")

                # 檢查載入後的權重
                first_param_after = next(iter(trainer.net.parameters())).data

                diff = torch.abs(first_param_after - first_param_before).sum().item()
                if diff > 1e-6:
                    print(f"   ✅ 權重已改變 (差異: {diff:.6f})")
                else:
                    print("   ⚠️  權重未改變")

                opt_state = state.get("optimizer_state")
                if opt_state is not None:
                    try:
                        trainer.opt.load_state_dict(opt_state)
                        print("   ✓ 優化器狀態已載入")
                    except Exception:
                        print("   ⚠️  無法載入 optimizer_state")
                return True
            else:
                print("   ❌ 檔案不是字典格式")
        except Exception as load_err:
            print(f"   ❌ 載入模型失敗: {load_err}")
            import traceback

            traceback.print_exc()
        return False

    # 尋找最新的檢查點
    checkpoints = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
    )

    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"\n🔄 嘗試載入: {latest_checkpoint}")
        success = _load_model(checkpoint_path)

        if success:
            print("\n✅ UI 載入邏輯測試通過")
        else:
            print("\n❌ UI 載入邏輯測試失敗")
    else:
        print("\n⚠️  找不到檢查點檔案")


def main():
    """執行所有測試"""
    print("\n" + "=" * 70)
    print("權重載入診斷工具")
    print("=" * 70)
    print("\n此工具將幫助您診斷權重載入問題\n")

    # 測試 1: 基本權重載入
    test1_passed = test_weight_loading()

    # 測試 2: UI 載入邏輯
    test_ui_loading_logic()

    print("\n\n" + "=" * 70)
    print("診斷總結")
    print("=" * 70)

    if test1_passed:
        print("\n✅ 權重載入功能正常")
        print("\n如果您在 UI 中看不到權重變化，可能的原因：")
        print("1. 每次啟動 AI 模式都會創建新的訓練器（重新初始化）")
        print("2. 需要確保在啟動 AI 時正確載入檢查點")
        print("3. 檢查 UI 的 _setup_ppo_trainer() 是否正確調用載入邏輯")
    else:
        print("\n❌ 權重載入異常")
        print("\n可能的解決方案：")
        print("1. 檢查檢查點檔案是否完整")
        print("2. 嘗試重新訓練並保存新的檢查點")
        print("3. 檢查 PyTorch 版本兼容性")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
