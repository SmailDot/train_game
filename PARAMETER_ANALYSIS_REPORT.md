# 🔬 訓練崩潰深度分析與修復方案

**分析時間**: 2025-11-15
**問題**: 訓練從迭代 #5940 到 #14460 期間性能崩潰至 0 分
**分析範圍**: 171 個檢查點（每 50 次迭代採樣）

---

## 📊 發現的關鍵問題

### 1. **Critic Bias 極度不穩定** ⚠️⚠️⚠️
```
變異係數（CV）: 41.5%
總變化: -44.7%
平均 Norm: 0.167258
```

**影響**:
- Critic 網絡無法穩定評估狀態價值
- 導致 Advantage 估計錯誤
- 錯誤的 Advantage → 錯誤的 Policy 更新 → 性能崩潰

**證據**:
- 其他參數 CV 都 < 3%，只有 critic.bias 異常高
- 在參數趨勢圖中可見劇烈震盪（見 `checkpoints/param_trends.png`）

---

### 2. **Actor Bias 在崩潰點跳變** ⚠️
```
崩潰前（#7390）: 0.533719
崩潰後（#7440）: 0.560377
變化幅度: +5.0%
```

**影響**:
- 動作分布突然偏移
- 可能導致選擇錯誤的移動方向
- 失去探索能力，陷入局部最優

---

### 3. **多個參數持續增長** ⚠️
```
fc1.bias:     +11.2%
fc2.bias:     +10.9%
actor.weight: +13.5%
fc1.weight:   +8.6%
critic.weight: +5.4%
```

**影響**:
- 模型不斷放大輸入信號
- 梯度累積效應
- 缺乏權重正則化

---

## 🔧 應用的改進方案

### 修改的參數

| 參數 | 原始值 | 改進值 | 變化 | 原因 |
|------|--------|--------|------|------|
| **learning_rate** | 0.00025 | 0.0001 | -60% | 減少參數震盪 |
| **weight_decay** | 0.0 | 0.0001 | +∞ | 防止權重持續增長 |
| **clip_range** | 0.2 | 0.1 | -50% | 限制 policy 更新幅度 |
| **entropy_coef** | 0.01 | 0.02 | +100% | 增加探索 |
| **vf_coef** | 0.5 | 1.0 | +100% | 加強 critic 訓練 |
| **max_grad_norm** | 0.5 | 0.3 | -40% | 更強梯度裁剪 |

### 修改的文件

1. ✅ **utils/training_config.py**
   - 更新 RTX_3060TI_CONFIG 和 CPU_CONFIG
   - 添加改進歷史註釋
   - 備份: `utils/training_config.py.backup_20251115_101144`

2. ✅ **agents/pytorch_trainer.py**
   - optimizer 添加 `weight_decay=1e-4`
   - 備份: `agents/pytorch_trainer.py.backup_20251115_101144`

3. ✅ **game/ui.py** (之前的修復)
   - 添加 `training_history.json` 完整歷史記錄
   - 修復 scores.json TOP 50 截斷問題

4. ✅ **agents/pytorch_trainer.py** (之前的修復)
   - 崩潰檢測優先讀取完整歷史

---

## 📈 預期效果

### 穩定性指標
```
✓ Critic bias CV: 41.5% → <20%
✓ 參數變化: >5% → <3%
✓ 崩潰檢測: 無效 → 10-50 局內檢測
```

### 訓練效果
```
✓ 更穩定的分數增長
✓ 減少突然崩潰風險
✓ 更好的探索-利用平衡
✓ 參數更新更平穩
```

---

## 🚀 執行計劃

### 步驟 1: 回檔到最佳檢查點
```bash
python execute_complete_fix.py
```

**效果**:
- 回檔到 checkpoint_5930.pt（歷史最高分 1418）
- 清理崩潰後的檢查點（#7500+）
- 釋放約 450 MB 磁碟空間
- 備份當前狀態

### 步驟 2: 啟動改進後的訓練
```bash
python run_game.py
```

**自動應用**:
- 新的學習率（0.0001）
- 更強的梯度裁剪（0.3）
- 加強的 critic 訓練（vf_coef=1.0）
- 權重衰減（weight_decay=1e-4）

### 步驟 3: 監控訓練效果

#### 3.1 檢查 training_history.json
```bash
# 查看最近 20 條記錄
python -c "import json; data = json.load(open('checkpoints/training_history.json')); print('\n'.join([f\"#{d['iteration']}: {d['score']}\" for d in data[:20]]))"
```

#### 3.2 使用 TensorBoard 監控
```bash
tensorboard --logdir=checkpoints/tb
```

**關注指標**:
- `loss/critic`: 應該平穩下降
- `loss/actor`: 應該震盪但有下降趨勢
- `metrics/entropy`: 應該維持在合理範圍（0.5-1.5）

#### 3.3 檢查參數穩定性（每 1000 次迭代）
```bash
python analyze_parameter_trends.py
```

**驗證**:
- Critic bias CV < 20%
- 所有參數變化 < 3% per 1000 iterations

---

## 📁 生成的分析文件

### 詳細分析
1. **checkpoints/param_trends.png** - 參數變化趨勢圖（8 個關鍵參數）
2. **checkpoints/detailed_parameter_analysis.json** - 完整參數統計
3. **checkpoints/training_config_suggestions.json** - 配置建議
4. **checkpoints/parameter_analysis_report.json** - 異常檢測報告

### 配置文件
5. **training_config_improved.json** - 改進後的完整配置
6. **PARAMETER_IMPROVEMENT_GUIDE.md** - 詳細應用指南

### 備份文件
7. **utils/training_config.py.backup_20251115_101144**
8. **agents/pytorch_trainer.py.backup_20251115_101144**

---

## 🎯 驗證清單

### 立即驗證（前 100 次迭代）
- [ ] 訓練正常啟動
- [ ] training_history.json 開始填充
- [ ] 分數 > 200（不應該立即崩潰）
- [ ] 無錯誤信息

### 短期驗證（前 1000 次迭代）
- [ ] 分數穩定在 500-1000 範圍
- [ ] 沒有連續 10 局 < 200 的情況
- [ ] Critic bias CV < 30%
- [ ] 崩潰檢測未觸發

### 中期驗證（前 3000 次迭代）
- [ ] 分數接近或超過 1000
- [ ] Critic bias CV < 20%
- [ ] 參數變化趨於平穩
- [ ] 達到或超過之前的最高分 1418

---

## 📝 理論基礎

### 為什麼這些改進有效？

#### 1. 降低學習率
```
理論: η ↓ → ΔW ↓ → 參數更新更平穩
效果: 減少震盪，增加穩定性
```

#### 2. 增加 Weight Decay
```
理論: L = L_original + λ||W||²
效果: 防止權重無限增長，提高泛化能力
```

#### 3. 降低 Clip Range
```
理論: π_new/π_old ∈ [1-ε, 1+ε], ε ↓ → 更新更保守
效果: 減少 policy 劇烈變化
```

#### 4. 增加 Entropy Bonus
```
理論: L = L_policy - α·H(π)
效果: 鼓勵探索，防止過早收斂
```

#### 5. 加強 Critic 訓練
```
理論: L = L_policy + β·L_critic, β ↑ → critic 更重要
效果: 更準確的價值估計 → 更好的 advantage
```

---

## 🔮 未來可選的進階改進

如果問題持續，可以考慮：

### 1. Layer Normalization
```python
# 在 networks.py 中
self.fc1 = nn.Linear(obs_dim, 256)
self.ln1 = nn.LayerNorm(256)  # 新增
```

### 2. Huber Loss for Critic
```python
# 在 pytorch_trainer.py 中
# critic_loss = F.mse_loss(values, returns)  # 舊
critic_loss = F.smooth_l1_loss(values, returns)  # 新（Huber）
```

### 3. 學習率調度
```python
# 在 pytorch_trainer.py 中
self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    self.opt, T_max=10000
)
```

### 4. 獨立的 Actor/Critic 學習率
```python
self.opt_actor = Adam(self.net.actor.parameters(), lr=1e-4)
self.opt_critic = Adam(self.net.critic.parameters(), lr=5e-5)
```

---

## 📞 問題排查

### 如果分數仍然崩潰

1. **檢查 training_history.json**
   ```bash
   # 查看是否正確記錄 0 分
   python -c "import json; d=json.load(open('checkpoints/training_history.json')); print([x for x in d if x['score']<100][:10])"
   ```

2. **檢查崩潰檢測**
   ```bash
   # 應該在日誌中看到回檔信息
   grep "檢測到性能崩潰" logs/*.log
   ```

3. **重新分析參數**
   ```bash
   python analyze_parameter_trends.py
   ```

4. **嘗試更保守的配置**
   - learning_rate: 0.0001 → 0.00005
   - clip_range: 0.1 → 0.05

### 如果訓練太慢

1. **增加 batch_size**
   ```python
   # utils/training_config.py
   "batch_size": 256 → 512  # 如果 GPU 記憶體足夠
   ```

2. **減少 ppo_epochs**
   ```python
   "ppo_epochs": 10 → 6
   ```

---

## 🏆 成功標準

### 最終目標
```
✓ 穩定訓練 10,000+ 次迭代無崩潰
✓ 平均分數 > 1200
✓ 最高分數 > 1500
✓ Critic bias CV < 15%
✓ 成功達成 2048 tile（最終目標）
```

---

## 📚 相關文檔

- **參數趨勢圖**: `checkpoints/param_trends.png`
- **詳細分析**: `checkpoints/detailed_parameter_analysis.json`
- **應用指南**: `PARAMETER_IMPROVEMENT_GUIDE.md`
- **PPO 論文**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

---

**報告生成時間**: 2025-11-15 10:11:44
**分析工具**: analyze_checkpoint_parameters.py, analyze_parameter_trends.py
**應用工具**: modify_training_config.py, execute_complete_fix.py

祝訓練順利！🚀
