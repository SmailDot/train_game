"""
根據參數分析結果，生成訓練改進建議
"""

import json

import numpy as np

print("=" * 80)
print("🎯 基於參數分析的訓練改進方案")
print("=" * 80)

# 讀取分析報告
with open("checkpoints/detailed_parameter_analysis.json", "r", encoding="utf-8") as f:
    report = json.load(f)

print(f"\n📊 分析概要:")
print(f"   分析範圍: #{report['iteration_range'][0]} → #{report['iteration_range'][1]}")
print(f"   崩潰點: #{report['crash_iteration']}")
print(f"   檢查點數量: {report['checkpoints_analyzed']}")
print(f"   追蹤參數: {report['parameters_tracked']}")

# === 分析不穩定參數 ===
print(f"\n" + "=" * 80)
print(f"問題 1: Critic Bias 極度不穩定（變異係數 41.5%）")
print(f"=" * 80)

critic_bias = report["unstable_parameters"][0]
print(f"\n詳細數據:")
print(f"   參數名稱: {critic_bias['param']}")
print(f"   變異係數: {critic_bias['cv']:.1f}%")
print(f"   總變化: {critic_bias['total_change_pct']:.1f}%")
print(f"   平均 Norm: {critic_bias['mean_norm']:.6f}")

print(f"\n🔍 問題分析:")
print(f"   Critic bias 的變異係數達到 41.5%，遠高於其他參數")
print(f"   這表示 critic 網絡在評估狀態價值時非常不穩定")
print(f"   在 PPO 算法中，critic 不穩定會導致 advantage 估計錯誤")
print(f"   錯誤的 advantage → 錯誤的 policy 更新 → 性能崩潰")

print(f"\n✅ 解決方案:")
print(f"   1. 降低 Critic 學習率:")
print(f"      目前 actor 和 critic 使用相同學習率 0.00025")
print(f"      建議 critic_lr = 0.0001 (actor_lr 的 40%)")
print(f"")
print(f"   2. 增加 Critic Loss 的權重:")
print(f"      目前可能 critic 訓練不足")
print(f"      建議 critic_loss_coef = 1.0 (或更高)")
print(f"")
print(f"   3. 使用 Huber Loss 代替 MSE:")
print(f"      Huber loss 對離群值更魯棒")
print(f"      可以減少 critic 的劇烈震盪")

# === 分析 Actor Bias 變化 ===
print(f"\n" + "=" * 80)
print(f"問題 2: Actor Bias 在崩潰前後有明顯跳變")
print(f"=" * 80)

actor_change = report["crash_impact"][0]
print(f"\n詳細數據:")
print(f"   參數名稱: {actor_change['param']}")
print(f"   崩潰前 (#{actor_change['iter_before']}): {actor_change['norm_before']:.6f}")
print(f"   崩潰後 (#{actor_change['iter_after']}): {actor_change['norm_after']:.6f}")
print(f"   變化幅度: {actor_change['change_pct']:+.1f}%")

print(f"\n🔍 問題分析:")
print(f"   Actor bias 在崩潰前後增加了 5%")
print(f"   雖然 5% 不算巨大，但可能導致動作分布偏移")
print(f"   在 2048 遊戲中，動作分布的微小變化可能導致:")
print(f"   - 選擇錯誤的移動方向")
print(f"   - 過度偏好某個方向")
print(f"   - 失去探索能力")

print(f"\n✅ 解決方案:")
print(f"   1. 增加 Entropy Bonus:")
print(f"      鼓勵探索，防止動作分布過早收斂")
print(f"      建議 entropy_coef = 0.01 → 0.02")
print(f"")
print(f"   2. 使用 Action Smoothing:")
print(f"      在訓練時添加小量噪音到動作")
print(f"      防止 policy 過度確定")
print(f"")
print(f"   3. 限制 Policy Update 大小:")
print(f"      降低 PPO clip range")
print(f"      建議 clip_range = 0.2 → 0.1")

# === 整體參數趨勢分析 ===
print(f"\n" + "=" * 80)
print(f"問題 3: 多個參數在整個訓練期間持續增長")
print(f"=" * 80)

growing_params = [p for p in report["unstable_parameters"] if p["total_change_pct"] > 5]
print(f"\n持續增長的參數:")
for p in growing_params:
    print(f"   {p['param']:<20} 總變化: {p['total_change_pct']:+.1f}%")

print(f"\n🔍 問題分析:")
print(f"   多個權重矩陣持續增長（+5% 到 +13.5%）")
print(f"   這表示模型在不斷放大輸入信號")
print(f"   可能原因:")
print(f"   - 梯度累積效應")
print(f"   - 缺乏權重衰減")
print(f"   - 獎勵尺度問題")

print(f"\n✅ 解決方案:")
print(f"   1. 增加 Weight Decay:")
print(f"      目前 weight_decay = 0 或很小")
print(f"      建議 weight_decay = 1e-4")
print(f"")
print(f"   2. 使用 Layer Normalization:")
print(f"      在 fc1 和 fc2 之後添加 LayerNorm")
print(f"      穩定激活值的尺度")
print(f"")
print(f"   3. 降低整體學習率:")
print(f"      目前 0.00025 可能稍高")
print(f"      建議 0.0001")

# === 生成具體的配置建議 ===
print(f"\n" + "=" * 80)
print(f"🔧 具體配置修改建議")
print(f"=" * 80)

config_suggestions = {
    "learning_rate": {
        "current": 0.00025,
        "suggested": 0.0001,
        "reason": "降低學習率以減少參數震盪",
    },
    "critic_learning_rate": {
        "current": "same as actor",
        "suggested": 0.00005,
        "reason": "Critic 需要更保守的更新（actor_lr 的 50%）",
    },
    "weight_decay": {"current": 0.0, "suggested": 0.0001, "reason": "防止權重持續增長"},
    "clip_range": {"current": 0.2, "suggested": 0.1, "reason": "限制 policy 更新幅度"},
    "entropy_coef": {
        "current": 0.01,
        "suggested": 0.02,
        "reason": "增加探索，防止過早收斂",
    },
    "critic_loss_coef": {
        "current": 0.5,
        "suggested": 1.0,
        "reason": "加強 critic 訓練",
    },
    "max_grad_norm": {"current": 0.5, "suggested": 0.3, "reason": "更強的梯度裁剪"},
}

print(f"\n在 agents/pytorch_trainer.py 中修改:")
print(f"")
for param, info in config_suggestions.items():
    print(f"# {param}")
    print(f"# 原因: {info['reason']}")
    print(f"# 目前: {info['current']}")
    print(f"# 建議: {info['suggested']}")
    print(f"")

# === 保存配置建議 ===
with open("checkpoints/training_config_suggestions.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "analysis_summary": {
                "main_issue": "Critic bias instability (CV 41.5%)",
                "secondary_issues": [
                    "Actor bias jumps at crash point (+5%)",
                    "Multiple parameters continuously growing (+5% to +13%)",
                ],
            },
            "config_suggestions": config_suggestions,
            "implementation_notes": [
                "修改 agents/pytorch_trainer.py 中的超參數",
                "考慮添加 Layer Normalization",
                "使用 Huber Loss 代替 MSE for critic",
                "實施 separate learning rates for actor and critic",
            ],
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"✅ 配置建議已保存到: checkpoints/training_config_suggestions.json")

# === 最終建議 ===
print(f"\n" + "=" * 80)
print(f"📝 實施步驟")
print(f"=" * 80)

print(
    f"""
1. 立即修改（優先級：高）
   ✓ 降低學習率: 0.00025 → 0.0001
   ✓ 增加權重衰減: 0.0 → 0.0001
   ✓ 降低 clip range: 0.2 → 0.1
   
2. 重要修改（優先級：中）
   ✓ 設置獨立的 critic_lr: 0.00005
   ✓ 增加 entropy_coef: 0.01 → 0.02
   ✓ 增加 critic_loss_coef: 0.5 → 1.0
   
3. 進階修改（優先級：低，如果問題持續）
   ✓ 添加 Layer Normalization
   ✓ 使用 Huber Loss
   ✓ 實施 learning rate scheduling

4. 測試流程
   ✓ 使用修改後的配置從 checkpoint_5930.pt 重新開始
   ✓ 密切監控前 1000 次迭代
   ✓ 檢查 critic.bias 的變異係數是否降低
   ✓ 確認分數穩定增長，沒有突然崩潰

5. 驗證指標
   ✓ Critic bias CV 應該 < 20%
   ✓ 崩潰前後參數變化 < 3%
   ✓ 分數不應該連續 10 局 < 200
"""
)

print(f"=" * 80)
