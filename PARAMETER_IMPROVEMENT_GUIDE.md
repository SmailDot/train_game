# 參數改進應用指南

生成時間: 2025-11-15T10:10:15.167393

## 分析結果

- 主要問題: Critic bias 極度不穩定（變異係數 41.5%）
- 崩潰迭代: #7436
- 分析檢查點: 171 個 (#5940 → #14460)

## 配置對比

| 參數 | 原始值 | 改進值 | 變化 |
|------|--------|--------|------|
| learning_rate | 0.000250 | 0.000100 | -60% |
| weight_decay | 0.000000 | 0.000100 | +∞ |
| clip_range | 0.200000 | 0.100000 | -50% |
| entropy_coef | 0.010000 | 0.020000 | +100% |
| vf_coef | 0.500000 | 1.000000 | +100% |
| max_grad_norm | 0.500000 | 0.300000 | -40% |


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
cp training_config_improved.json training_config.json

然後正常啟動訓練:
python run_game.py

方法 3：動態配置（推薦）
---------------------------------
訓練器已支持動態配置更新，創建 training_config.json:

{
    "learning_rate": 0.0001,
    "clip_range": 0.1,
    "entropy_coef": 0.02,
    "vf_coef": 1.0,
    "max_grad_norm": 0.3
}

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

