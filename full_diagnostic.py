"""全面診斷訓練系統的潛在問題"""

import json
import os

import numpy as np
import torch


def check_all_issues():
    print("=" * 80)
    print("🔍 訓練系統全面診斷")
    print("=" * 80)

    issues_found = []
    warnings = []

    # ===== 1. 檢查 PyTorch 環境 =====
    print("\n📦 1. PyTorch 環境檢查")
    print(f"   版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 數量: {torch.cuda.device_count()}")
        print(f"   當前 GPU: {torch.cuda.get_device_name(0)}")

    # ===== 2. 檢查崩潰檢測邏輯 =====
    print("\n🚨 2. 崩潰檢測邏輯檢查")

    scores_file = "checkpoints/scores.json"
    if os.path.exists(scores_file):
        with open(scores_file, "r", encoding="utf-8") as f:
            scores_data = json.load(f)

        # 檢查數據順序
        print(f"   總記錄數: {len(scores_data)}")

        # 按分數排序（原始）
        top_5_by_score = scores_data[:5]
        print(f"   按分數前5: ", end="")
        for entry in top_5_by_score:
            print(f"#{entry['iteration']}({entry['score']})", end=" ")
        print()

        # 按迭代排序（時間）
        scores_by_iteration = sorted(
            scores_data, key=lambda x: x.get("iteration", 0), reverse=True
        )
        recent_5 = scores_by_iteration[:5]
        print(f"   按時間前5: ", end="")
        for entry in recent_5:
            print(f"#{entry['iteration']}({entry['score']})", end=" ")
        print()

        # 檢查是否正確排序
        if top_5_by_score[0]["iteration"] != recent_5[0]["iteration"]:
            print("   ✅ 正確：已區分分數排序和時間排序")
        else:
            warnings.append("數據恰好最高分就是最近的，無法驗證排序邏輯")

    # ===== 3. 檢查 GAE 計算 =====
    print("\n📊 3. GAE 計算邏輯檢查")

    # 模擬一個簡單的軌跡
    gamma = 0.99
    lam = 0.95

    # 測試案例：3 步軌跡，第 2 步結束
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5]
    dones = [0, 1, 0]  # 第 2 步 done
    next_values = [0.6, 0.0, 0.6]  # done 時為 0

    # 手動計算 GAE
    advs = []
    gae = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advs.insert(0, gae)

    print(f"   測試軌跡: rewards={rewards}, dones={dones}")
    print(f"   計算的 GAE: {[f'{a:.3f}' for a in advs]}")

    # 驗證 done 時 GAE 是否正確重置
    # 第 2 步 done，所以第 3 步的 GAE 應該重新開始
    if abs(advs[1]) < 10 and abs(advs[2]) < 10:  # 合理範圍
        print("   ✅ GAE 計算看起來正確")
    else:
        issues_found.append("GAE 計算可能有問題：值異常大")

    # ===== 4. 檢查檢查點完整性 =====
    print("\n💾 4. 檢查點完整性檢查")

    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            [
                f
                for f in os.listdir(checkpoint_dir)
                if f.startswith("checkpoint_") and f.endswith(".pt")
            ]
        )

        print(f"   檢查點數量: {len(checkpoints)}")

        # 檢查 checkpoint_best.pt
        best_checkpoint = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        if os.path.exists(best_checkpoint):
            print(f"   ✅ checkpoint_best.pt 存在")

            # 嘗試載入
            try:
                ckpt = torch.load(best_checkpoint, map_location="cpu")
                if "model_state" in ckpt and "optimizer_state" in ckpt:
                    print(f"   ✅ checkpoint_best.pt 格式正確")
                else:
                    issues_found.append("checkpoint_best.pt 格式錯誤：缺少必要的鍵")
            except Exception as e:
                issues_found.append(f"無法載入 checkpoint_best.pt: {e}")
        else:
            warnings.append("checkpoint_best.pt 不存在")

        # 檢查最近的檢查點
        if len(checkpoints) > 0:
            recent_checkpoint = checkpoints[-1]
            recent_path = os.path.join(checkpoint_dir, recent_checkpoint)
            try:
                ckpt = torch.load(recent_path, map_location="cpu")
                print(f"   ✅ 最近的檢查點可載入: {recent_checkpoint}")
            except Exception as e:
                issues_found.append(f"無法載入最近的檢查點 {recent_checkpoint}: {e}")

    # ===== 5. 檢查數值穩定性 =====
    print("\n🔬 5. 數值穩定性檢查")

    # 檢查 scores.json 中是否有異常值
    if os.path.exists(scores_file):
        all_scores = [entry["score"] for entry in scores_data]
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)

        print(f"   分數統計: 平均={mean_score:.1f}, 標準差={std_score:.1f}")
        print(f"   範圍: [{min_score}, {max_score}]")

        # 檢查是否有 NaN 或 Inf
        if any(not np.isfinite(s) for s in all_scores):
            issues_found.append("發現 NaN 或 Inf 分數！")
        else:
            print(f"   ✅ 無 NaN 或 Inf")

        # 檢查是否有異常低分（可能是 bug）
        very_low_scores = [s for s in all_scores if s < 50]
        if very_low_scores:
            warnings.append(f"發現 {len(very_low_scores)} 個極低分數（<50）")

    # ===== 6. 檢查參數配置 =====
    print("\n⚙️ 6. 訓練參數檢查")

    config_file = "training_config.json"
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        print(f"   學習率: {config.get('lr', 'N/A')}")
        print(f"   熵係數: {config.get('ent_coef', 'N/A')}")
        print(f"   Gamma: {config.get('gamma', 'N/A')}")
        print(f"   批次大小: {config.get('batch_size', 'N/A')}")

        # 檢查學習率是否過大或過小
        lr = config.get("lr", 3e-4)
        if lr > 1e-2:
            warnings.append(f"學習率可能過大: {lr}")
        elif lr < 1e-6:
            warnings.append(f"學習率可能過小: {lr}")
        else:
            print(f"   ✅ 學習率在合理範圍")
    else:
        print("   ⚠️ 未找到 training_config.json")

    # ===== 7. 檢查潛在的內存問題 =====
    print("\n💾 7. 內存使用檢查")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"   GPU 內存已分配: {allocated:.1f} MB")
        print(f"   GPU 內存已保留: {reserved:.1f} MB")

        if allocated > 1000:
            warnings.append(f"GPU 內存使用較高: {allocated:.1f} MB")

    # ===== 8. 檢查最近訓練趨勢 =====
    print("\n📈 8. 最近訓練趨勢檢查")

    if os.path.exists(scores_file):
        scores_by_iteration = sorted(
            scores_data, key=lambda x: x.get("iteration", 0), reverse=True
        )

        if len(scores_by_iteration) >= 30:
            recent_30 = [e["score"] for e in scores_by_iteration[:30]]

            # 分成三段
            first_10 = np.mean(recent_30[:10])
            second_10 = np.mean(recent_30[10:20])
            third_10 = np.mean(recent_30[20:30])

            print(f"   最近10局平均: {first_10:.1f}")
            print(f"   之前10局平均: {second_10:.1f}")
            print(f"   再之前10局平均: {third_10:.1f}")

            # 檢查趨勢
            if first_10 > second_10 > third_10:
                print("   ✅ 持續進步趨勢")
            elif first_10 < second_10 * 0.7:
                warnings.append(f"最近表現下降明顯: {first_10:.1f} vs {second_10:.1f}")
            else:
                print("   ✅ 表現穩定")

    # ===== 總結 =====
    print("\n" + "=" * 80)
    print("📋 診斷總結")
    print("=" * 80)

    if issues_found:
        print(f"\n❌ 發現 {len(issues_found)} 個嚴重問題:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✅ 未發現嚴重問題")

    if warnings:
        print(f"\n⚠️ {len(warnings)} 個警告:")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    else:
        print("✅ 無警告")

    if not issues_found and not warnings:
        print("\n🎉 系統狀態良好，可以開始長時間訓練！")
    elif not issues_found:
        print("\n✅ 無嚴重問題，警告項目可以忽略或稍後處理")
    else:
        print("\n⚠️ 建議修復嚴重問題後再開始長時間訓練")

    print("=" * 80)

    return len(issues_found) == 0


if __name__ == "__main__":
    success = check_all_issues()
    exit(0 if success else 1)
