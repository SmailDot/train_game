# 🚀 快速執行指南

## 📋 現在就開始！

### 第一步：回檔 + 清理（2 分鐘）
```powershell
python execute_complete_fix.py
```
**做什麼**:
- ✅ 回檔到 checkpoint_5930.pt（最高分 1418）
- ✅ 刪除崩潰後的 7,000+ 個檢查點
- ✅ 釋放 ~450 MB 空間
- ✅ 備份當前狀態

---

### 第二步：啟動改進後的訓練（立即）
```powershell
python run_game.py
```
**自動應用的改進**:
- ✅ 學習率降低 60%（0.00025 → 0.0001）
- ✅ Clip range 降低 50%（0.2 → 0.1）
- ✅ Critic 訓練加強 100%（vf_coef 0.5 → 1.0）
- ✅ 探索增加 100%（entropy_coef 0.01 → 0.02）
- ✅ 梯度裁剪加強 40%（0.5 → 0.3）
- ✅ 權重衰減啟用（weight_decay 0 → 0.0001）

---

### 第三步：監控（選擇性）

#### 方法 1: 查看分數記錄
```powershell
# 查看最近 20 局
python -c "import json; d=json.load(open('checkpoints/training_history.json')); [print(f\"#{x['iteration']:5d}: {x['score']:4d}\") for x in d[:20]]"
```

#### 方法 2: TensorBoard（推薦）
```powershell
tensorboard --logdir=checkpoints/tb
# 在瀏覽器打開 http://localhost:6006
```

#### 方法 3: 每 1000 次迭代檢查參數
```powershell
python analyze_parameter_trends.py
```

---

## ✅ 驗證清單

### 前 100 次迭代（~3 分鐘）
- [ ] 訓練正常啟動
- [ ] 分數 > 200
- [ ] training_history.json 正在填充

### 前 1000 次迭代（~30 分鐘）
- [ ] 分數穩定在 500-1000
- [ ] 沒有連續 10 局 < 200
- [ ] 無崩潰警告

### 前 3000 次迭代（~1.5 小時）
- [ ] 分數接近或超過 1000
- [ ] 達到或超過最高分 1418
- [ ] 參數穩定

---

## 🔧 已應用的改進

### 根本原因
```
問題: Critic bias 變異係數 41.5%（極度不穩定）
結果: 錯誤的價值估計 → 錯誤的 policy 更新 → 崩潰
```

### 解決方案
| 改進 | 效果 |
|------|------|
| 降低學習率 | 減少參數震盪 |
| 增加 weight decay | 防止權重無限增長 |
| 降低 clip range | 限制 policy 變化 |
| 加強 critic 訓練 | 更準確的價值估計 |
| 增加探索 | 防止過早收斂 |
| 更強梯度裁剪 | 防止梯度爆炸 |

### 預期結果
```
✓ Critic bias CV: 41.5% → <20%
✓ 參數變化: >5% → <3%
✓ 崩潰檢測: 無效 → 10-50 局內
✓ 訓練穩定性: ⭐⭐ → ⭐⭐⭐⭐⭐
```

---

## 📊 如何看 TensorBoard

### 關鍵指標
1. **loss/critic** - 應該平穩下降
   - 好: 逐漸下降，小震盪
   - 壞: 劇烈震盪，突然暴漲

2. **loss/actor** - 震盪但有趨勢
   - 好: 整體下降趨勢
   - 壞: 無趨勢，純噪音

3. **metrics/mean_reward** - 遊戲分數
   - 好: 穩定增長
   - 壞: 突然歸零

4. **metrics/entropy** - 探索程度
   - 好: 0.5-1.5 之間
   - 壞: < 0.1（沒探索）

---

## 🆘 如果出問題

### 分數又崩潰了？
```powershell
# 1. 檢查是否正確記錄
python -c "import json; d=json.load(open('checkpoints/training_history.json')); print('Zero scores:', len([x for x in d if x['score']==0]))"

# 2. 檢查崩潰檢測是否啟動
# 應該看到自動回檔信息

# 3. 重新分析參數
python analyze_parameter_trends.py
```

### 訓練太慢？
```
GPU 使用率低？
→ 增加 batch_size (256 → 512)
→ 減少 ppo_epochs (10 → 6)
```

### 還是不穩定？
```
嘗試更保守的配置:
- learning_rate: 0.0001 → 0.00005
- clip_range: 0.1 → 0.05
```

---

## 📁 生成的文件

### 分析報告
- `PARAMETER_ANALYSIS_REPORT.md` - 完整深度分析
- `PARAMETER_IMPROVEMENT_GUIDE.md` - 詳細應用指南
- `checkpoints/param_trends.png` - 參數趨勢圖

### 配置文件
- `training_config_improved.json` - 改進後配置
- `checkpoints/detailed_parameter_analysis.json` - 統計數據

### 備份文件
- `utils/training_config.py.backup_20251115_101144`
- `agents/pytorch_trainer.py.backup_20251115_101144`

---

## 🎯 目標

### 短期（今天）
- ✅ 應用所有改進
- ✅ 訓練 3000 次迭代無崩潰
- ✅ 分數超過 1000

### 中期（本週）
- ✅ 訓練 10,000 次迭代
- ✅ 平均分數 > 1200
- ✅ 最高分數 > 1500

### 長期（最終）
- ✅ 達成 2048 tile
- ✅ 穩定的訓練系統
- ✅ 可復現的結果

---

**就是這樣！直接執行第一步吧！** 🚀
