"""
自動應用參數分析建議，修改訓練配置
"""

import json
import os
from datetime import datetime

print("=" * 80)
print("🔧 應用訓練參數改進")
print("=" * 80)

# === 讀取建議 ===
with open("checkpoints/training_config_suggestions.json", "r", encoding="utf-8") as f:
    suggestions = json.load(f)

print("\n📊 將應用以下改進:")
for param, info in suggestions["config_suggestions"].items():
    print(f"\n{param}:")
    print(f"   目前: {info['current']}")
    print(f"   建議: {info['suggested']}")
    print(f"   原因: {info['reason']}")

# === 創建新的配置文件 ===
print("\n" + "=" * 80)
print("創建改進後的配置...")
print("=" * 80)

improved_config = {
    "_description": "根據參數分析自動生成的改進配置",
    "_generated_at": datetime.now().isoformat(),
    "_analysis_results": {
        "main_issue": "Critic bias instability (CV 41.5%)",
        "crash_iteration": 7436,
        "analyzed_checkpoints": 171,
    },
    # 優化器配置
    "learning_rate": 0.0001,  # 降低 from 0.00025
    "weight_decay": 0.0001,  # 增加 from 0.0
    # PPO 配置
    "clip_range": 0.1,  # 降低 from 0.2
    "entropy_coef": 0.02,  # 增加 from 0.01
    "vf_coef": 1.0,  # 增加 from 0.5 (critic_loss_coef)
    "max_grad_norm": 0.3,  # 降低 from 0.5
    # 訓練配置
    "gamma": 0.99,
    "lam": 0.95,
    "batch_size": 64,
    "ppo_epochs": 4,
    # 學習率調度（如果需要）
    "lr_scheduler": {"type": "none", "enabled": False},  # 先保持固定學習率
    # 模型配置
    "use_layer_norm": False,  # 可選：稍後可以啟用
    "use_huber_loss": False,  # 可選：稍後可以啟用
}

# 保存配置
config_file = "training_config_improved.json"
with open(config_file, "w", encoding="utf-8") as f:
    json.dump(improved_config, f, ensure_ascii=False, indent=2)

print(f"✅ 改進配置已保存到: {config_file}")

# === 創建配置對比 ===
print("\n" + "=" * 80)
print("📊 配置對比")
print("=" * 80)

comparison = {
    "Parameter": [
        "learning_rate",
        "weight_decay",
        "clip_range",
        "entropy_coef",
        "vf_coef",
        "max_grad_norm",
    ],
    "Original": [0.00025, 0.0, 0.2, 0.01, 0.5, 0.5],
    "Improved": [0.0001, 0.0001, 0.1, 0.02, 1.0, 0.3],
    "Change": ["-60%", "+∞", "-50%", "+100%", "+100%", "-40%"],
}

print(f"\n{'參數':<20} {'原始值':>12} {'改進值':>12} {'變化':>12}")
print("-" * 60)
for i in range(len(comparison["Parameter"])):
    print(
        f"{comparison['Parameter'][i]:<20} {comparison['Original'][i]:>12.6f} "
        f"{comparison['Improved'][i]:>12.6f} {comparison['Change'][i]:>12}"
    )

# === 創建使用說明 ===
print("\n" + "=" * 80)
print("📝 使用說明")
print("=" * 80)

usage_instructions = f"""
使用改進配置的方法：

方法 1：直接修改 run_game.py
---------------------------------
打開 run_game.py，找到 PPOTrainer 初始化部分，修改為:

trainer = PPOTrainer(
    save_dir="checkpoints",
    lr=0.0001,              # 原 0.00025
    clip_eps=0.1,           # 原 0.2
    ent_coef=0.02,          # 原 0.01
    vf_coef=1.0,            # 原 0.5
    gamma=0.99,
    lam=0.95,
    batch_size=64,
    ppo_epochs=4,
)

並修改優化器初始化（在 pytorch_trainer.py 中）:
self.opt = torch.optim.Adam(
    self.net.parameters(), 
    lr=lr,
    weight_decay=0.0001     # 新增
)

方法 2：使用配置文件（如果已實現）
---------------------------------
如果訓練器支持配置文件，複製:
cp {config_file} training_config.json

然後正常啟動訓練:
python run_game.py

方法 3：動態配置（推薦）
---------------------------------
訓練器已支持動態配置更新，創建 training_config.json:

{{
    "learning_rate": 0.0001,
    "clip_range": 0.1,
    "entropy_coef": 0.02,
    "vf_coef": 1.0,
    "max_grad_norm": 0.3
}}

啟動訓練後，配置會自動應用。

重要提醒
---------------------------------
1. 先執行回檔到 checkpoint_5930.pt:
   python execute_complete_fix.py

2. 然後應用新配置並啟動訓練:
   python run_game.py

3. 密切監控前 1000 次迭代:
   - 檢查分數是否穩定
   - 確認沒有崩潰到 0 分
   - 觀察 training_history.json

4. 驗證改進效果:
   - Critic bias 變異係數應該 < 20%
   - 參數變化應該更平穩
   - 分數應該穩定增長
"""

print(usage_instructions)

# 保存使用說明
with open("PARAMETER_IMPROVEMENT_GUIDE.md", "w", encoding="utf-8") as f:
    f.write(f"# 參數改進應用指南\n\n")
    f.write(f"生成時間: {datetime.now().isoformat()}\n\n")
    f.write(f"## 分析結果\n\n")
    f.write(f"- 主要問題: Critic bias 極度不穩定（變異係數 41.5%）\n")
    f.write(f"- 崩潰迭代: #7436\n")
    f.write(f"- 分析檢查點: 171 個 (#5940 → #14460)\n\n")
    f.write(f"## 配置對比\n\n")
    f.write(f"| 參數 | 原始值 | 改進值 | 變化 |\n")
    f.write(f"|------|--------|--------|------|\n")
    for i in range(len(comparison["Parameter"])):
        f.write(
            f"| {comparison['Parameter'][i]} | {comparison['Original'][i]:.6f} | "
            f"{comparison['Improved'][i]:.6f} | {comparison['Change'][i]} |\n"
        )
    f.write(f"\n{usage_instructions}\n")

print(f"\n✅ 使用說明已保存到: PARAMETER_IMPROVEMENT_GUIDE.md")

# === 總結 ===
print("\n" + "=" * 80)
print("✅ 完成")
print("=" * 80)

print(
    f"""
已生成的文件:
1. {config_file} - 改進後的訓練配置
2. PARAMETER_IMPROVEMENT_GUIDE.md - 詳細應用指南

下一步:
1. 執行回檔: python execute_complete_fix.py
2. 應用新配置（參考上述說明）
3. 啟動訓練: python run_game.py
4. 監控效果

預期改進:
✓ Critic bias 穩定性提升（CV 從 41.5% 降到 <20%）
✓ 參數變化更平穩
✓ 不再出現突然崩潰到 0 分
✓ 訓練更穩定，性能更好
"""
)

print("=" * 80)
