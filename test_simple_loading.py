"""
測試完整的權重載入流程

這個腳本會：
1. 檢查現有的檢查點
2. 創建新的訓練器
3. 模擬 UI 的載入流程
4. 驗證權重是否正確載入
"""

import os

import torch

from agents.pytorch_trainer import PPOTrainer


def main():
    print("=" * 70)
    print("完整權重載入流程測試")
    print("=" * 70)

    checkpoint_dir = "./checkpoints"

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
        return

    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    print(f"\n📂 最新檢查點: {latest_checkpoint}")
    print(f"📂 完整路徑: {checkpoint_path}")

    # 創建訓練器（模擬 UI 啟動 AI 時的流程）
    print("\n" + "=" * 70)
    print("步驟 1: 創建新的訓練器（模擬 UI 啟動）")
    print("=" * 70)
    trainer = PPOTrainer(save_dir=checkpoint_dir, device="cpu")

    # 記錄初始權重
    print("\n初始權重（隨機初始化）:")
    initial_weights = {}
    for name, param in trainer.net.named_parameters():
        initial_weights[name] = param.data.clone()
        print(f"   {name}: mean={param.data.mean():.6f}")

    # 載入檢查點（模擬 _prepare_ppo_resume）
    print("\n" + "=" * 70)
    print("步驟 2: 載入檢查點")
    print("=" * 70)

    try:
        print(f"🔄 正在載入: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=trainer.device)

        if isinstance(state, dict):
            print("   ✓ 檢查點格式正確")
            print(f"   ✓ 包含鍵: {list(state.keys())}")

            model_state = state.get("model_state", state)

            # 載入前記錄第一個參數
            first_param_before = next(iter(trainer.net.parameters())).data.clone()

            # 載入權重
            trainer.net.load_state_dict(model_state)
            print("   ✓ 執行 load_state_dict() 完成")

            # 載入後檢查
            first_param_after = next(iter(trainer.net.parameters())).data
            diff = torch.abs(first_param_after - first_param_before).sum().item()

            if diff > 1e-6:
                print(f"   ✅ 權重已成功載入！(差異: {diff:.2f})")
            else:
                print(f"   ❌ 警告: 權重似乎未改變 (差異: {diff:.6f})")

        else:
            print("   ❌ 檢查點格式錯誤")
            return

    except Exception as e:
        print(f"   ❌ 載入失敗: {e}")
        import traceback

        traceback.print_exc()
        return

    # 驗證所有層的權重
    print("\n" + "=" * 70)
    print("步驟 3: 驗證所有層的權重變化")
    print("=" * 70)

    all_changed = True
    for name, param in trainer.net.named_parameters():
        initial = initial_weights[name]
        current = param.data

        diff = torch.abs(current - initial).sum().item()

        if diff > 1e-6:
            print(f"   ✅ {name}: 權重已改變 (差異: {diff:.2f})")
            print(f"      初始: mean={initial.mean():.6f}")
            print(f"      載入: mean={current.mean():.6f}")
        else:
            print(f"   ❌ {name}: 權重未改變")
            all_changed = False

    # 最終結論
    print("\n" + "=" * 70)
    print("測試結論")
    print("=" * 70)

    if all_changed:
        print("\n✅ 權重載入完全成功！")
        print("\n如果您在 UI 中仍然看不到權重變化，請檢查：")
        print("1. 是否每次啟動都創建了新的訓練器實例")
        print("2. 檢查控制台是否有載入成功的訊息")
        print("3. 確認您選擇的 .pt 檔案路徑正確")
    else:
        print("\n❌ 權重載入失敗！")
        print("\n可能的原因：")
        print("1. 檢查點檔案可能已損壞")
        print("2. PyTorch 版本不兼容")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
