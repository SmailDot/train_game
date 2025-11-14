# PPO 訓練優化和改進文檔

## 更新日期
2025-11-14

## 主要改進

### 1. ✅ 修復 UI 重疊問題
**問題描述**: 遊戲 UI 與左邊的演算法控制面板重疊在一起

**解決方案**:
- 修改 `draw_playfield()` 方法，將所有座標從絕對座標轉換為相對於 `play_area` 的座標
- 球的位置計算：`ball_x = self.play_area.left + int(self.play_area.width * 0.2)`
- 障礙物位置計算：使用 `play_area.left` 和 `play_area.top` 作為偏移
- 高度映射：從 `env.ScreenHeight` 正確縮放到 `play_area.height`

**測試結果**:
- ✅ UI 初始化成功
- ✅ 遊戲區域正確定位：`<rect(460, 0, 620, 840)>`
- ✅ 沒有與演算法面板重疊

---

### 2. ✅ 訓練模型選擇對話框
**新增功能**: 點擊 "AI 訓練" 按鈕時顯示配置對話框

**對話框功能**:
1. **算法選擇** (單選按鈕):
   - PPO - Proximal Policy Optimization - 穩定高效
   - SAC - Soft Actor-Critic - 連續控制
   - DQN - Deep Q-Network - 經典算法
   - DDQN - Double DQN - 改進的 DQN
   - TD3 - Twin Delayed DDPG - 高級算法

2. **Checkpoint 選項**:
   - Checkbox: "從 Checkpoint 繼續訓練"
   - "瀏覽..." 按鈕：打開文件選擇對話框
   - 自動根據選擇的算法定位到對應的 checkpoint 目錄

3. **操作按鈕**:
   - "開始訓練" - 根據選擇啟動訓練
   - "取消" - 返回主選單

**實現文件**:
- `game/training_dialog.py` - 對話框類
- `game/ui.py` - 整合對話框到主 UI

**用戶體驗**:
- 半透明背景遮罩
- 懸停效果
- 鍵盤熱鍵保留（1-5 快速選擇算法）

---

### 3. ✅ RTX 3060 Ti 優化配置
**新增配置文件**: `utils/training_config.py`

**GPU 優化參數**:
```python
RTX_3060TI_CONFIG = {
    "device": "cuda",              # 使用 GPU
    "batch_size": 256,             # 增大以利用 GPU
    "ppo_epochs": 10,              # 增加 PPO 更新次數
    "lr": 2.5e-4,                  # 降低學習率確保穩定
    "gamma": 0.99,                 # 折扣因子
    "lam": 0.95,                   # GAE lambda
    "clip_eps": 0.2,               # PPO clip 範圍
    "vf_coef": 0.5,                # Value function 係數
    "ent_coef": 0.01,              # 降低 entropy
    "max_grad_norm": 0.5,          # 梯度裁剪
    "horizon": 4096,               # 增加 rollout 長度
}
```

**CPU 配置**（較保守）:
- batch_size: 64
- ppo_epochs: 4
- horizon: 2048

**配置管理類**:
```python
config = TrainingConfig(use_gpu=True)
ppo_kwargs = config.get_ppo_kwargs()
should_use_vec_env = config.should_use_vectorized_env()
n_envs = config.get_recommended_n_envs()  # GPU: 8, CPU: 4
```

---

### 4. ✅ 改進的獎勵塑造
**新的獎勵配置**:
```python
REWARD_SHAPING_CONFIG = {
    "pass_obstacle": 10.0,      # 增加通過獎勵
    "collision": -10.0,         # 增加碰撞懲罰
    "survive_step": 0.1,        # 每步存活小獎勵
    "height_penalty": 0.05,     # 懲罰過高或過低
    "forward_progress": 0.2,    # 鼓勵前進
}
```

**目的**:
- 更明確的學習信號
- 鼓勵穩定飛行
- 減少抖動行為

---

### 5. ✅ Checkpoint 管理功能
**新增工具函數**:
```python
# 列出可用的 checkpoint
checkpoints = list_available_checkpoints(algorithm="ppo")
# 返回：[{"file": "...", "path": "...", "iteration": 1000, "size_mb": 2.5}, ...]

# 獲取最新的 checkpoint
latest = get_latest_checkpoint(algorithm="ppo")
```

**文件選擇對話框**:
- 使用 tkinter 文件對話框
- 自動定位到算法對應的目錄
- 過濾 `.pt` 文件

---

## 待實現功能

### 1. Checkpoint 載入集成
**當前狀態**: 對話框可以選擇 checkpoint，但尚未實現載入邏輯

**需要做的**:
1. 在 `PPOTrainer` 中添加 `load_checkpoint()` 方法
2. 在 `_start_training_with_config()` 中調用載入
3. 驗證載入後訓練繼續正常

**實現優先級**: 高

---

### 2. GPU 加速驗證
**需要驗證**:
1. PyTorch 是否正確使用 CUDA
2. Batch size 256 是否會 OOM
3. 8 個並行環境的性能提升

**測試計劃**:
```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"當前設備: {torch.cuda.get_device_name(0)}")
```

**實現優先級**: 高

---

### 3. 並行環境數據驗證
**問題**: 用戶反映 "檢查一下平行環境運算出的數據是否都有連動到主環境"

**驗證方法**:
1. 在 `SubprocVecEnv` 中添加日誌
2. 檢查每個 worker 的 reward 和 done 狀態
3. 驗證 trajectory 收集的數據完整性

**驗證代碼**:
```python
# 在 agents/pytorch_trainer.py 的 _collect_trajectory_vectorized() 中
print(f"環境 {i}: reward={reward}, done={done}")
```

**實現優先級**: 中

---

### 4. 觀察速度對訓練影響分析
**問題**: "觀察是否有因為觀察速度導致訓練上的不理想"

**分析**:
- `ai_speed_multiplier` 只影響 UI 更新速度
- 不應該影響訓練數據收集
- 但需要確認 `steps_this_frame` 的計算邏輯

**驗證方法**:
1. 在不同 speed 下訓練
2. 比較 reward 曲線和收斂速度
3. 檢查是否有幀跳過導致的數據遺漏

**實現優先級**: 中

---

### 5. Loss 函數改進
**目標**: "讓 loss 每次能有持續往正面的修正"

**當前 PPO Loss**:
```python
policy_loss = -advantages * ratio
clipped_loss = -advantages * torch.clamp(ratio, 1-eps, 1+eps)
loss = torch.max(policy_loss, clipped_loss).mean()
```

**改進方向**:
1. **添加 Value Loss 監控**:
   ```python
   value_loss = F.mse_loss(values, returns)
   total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
   ```

2. **Advantage 標準化**:
   ```python
   advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
   ```

3. **Early Stopping**:
   ```python
   if kl_divergence > target_kl:
       break  # 停止 PPO 更新
   ```

4. **Learning Rate Schedule**:
   ```python
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
   ```

**實現優先級**: 高

---

### 6. 模型集成 (Ensemble)
**問題**: "增加模型是否可以協助整個演算法推進"

**分析**:
- 單一智能體環境中，ensemble 的效益有限
- 更適合用於：
  1. 多樣性探索（不同超參數的模型）
  2. 穩定性（投票機制）
  3. 知識蒸餾（教師-學生模型）

**可能的實現方式**:

1. **投票集成**:
   ```python
   actions = [agent.act(state) for agent in agents]
   final_action = max(set(actions), key=actions.count)
   ```

2. **加權平均**:
   ```python
   q_values = [agent.get_q_values(state) for agent in agents]
   avg_q = sum(q_values) / len(q_values)
   action = torch.argmax(avg_q)
   ```

3. **知識蒸餾**:
   ```python
   teacher_output = teacher.act(state)
   student_loss = KL_div(student_output, teacher_output)
   ```

**建議**:
- 先專注於單一模型訓練
- 達到穩定性能後再考慮 ensemble
- 更重要的是超參數調優和獎勵塑造

**實現優先級**: 低

---

## 使用指南

### 啟動訓練（新流程）
1. 運行 `python run_game.py`
2. 點擊 "AI 訓練" 按鈕
3. 在對話框中選擇：
   - 訓練算法（預設 PPO）
   - 是否從 checkpoint 繼續
   - 如果是，瀏覽並選擇 checkpoint 文件
4. 點擊 "開始訓練"

### GPU 訓練配置
```python
from utils.training_config import TrainingConfig, print_training_summary

config = TrainingConfig(use_gpu=True)
print_training_summary(config)

# 在訓練器中使用
trainer = PPOTrainer(env, agent, **config.get_ppo_kwargs())
```

### Checkpoint 管理
```python
from utils.training_config import list_available_checkpoints, get_latest_checkpoint

# 列出所有 PPO checkpoints
checkpoints = list_available_checkpoints("ppo")
for cp in checkpoints[:5]:  # 顯示最新 5 個
    print(f"{cp['file']}: iteration {cp['iteration']}, {cp['size_mb']:.2f} MB")

# 自動載入最新的
latest_path = get_latest_checkpoint("ppo")
```

---

## 測試結果

### UI 測試
```
✅ UI 初始化成功
視窗大小: 1440 x 840
遊戲區域: <rect(460, 0, 620, 840)>
狀態面板: <rect(1080, 0, 360, 840)>
✅ UI 測試完成 - 沒有錯誤！
```

### 待測試項目
- [ ] GPU 訓練性能
- [ ] Checkpoint 載入和恢復
- [ ] 並行環境數據同步
- [ ] 不同 speed 下的訓練效果
- [ ] 新獎勵塑造的效果

---

## 文件結構

```
traingame/
├── game/
│   ├── ui.py                    # 主 UI（已更新：對話框集成）
│   └── training_dialog.py       # 新：訓練配置對話框
├── utils/
│   ├── training_config.py       # 新：GPU 優化配置和 checkpoint 管理
│   └── training_logger.py       # 結構化訓練日誌
├── agents/
│   ├── ppo_agent.py
│   ├── pytorch_trainer.py       # PPO 訓練器
│   └── ...
├── test_ui_quick.py             # 新：快速 UI 測試
└── README.md
```

---

## 下一步行動

### 優先級 1（立即執行）
1. ✅ 完成 UI 重疊修復
2. ✅ 實現訓練對話框
3. ✅ 創建 GPU 優化配置
4. ⏳ 實現 checkpoint 載入功能
5. ⏳ 驗證 GPU 訓練

### 優先級 2（本週內）
1. ⏳ 驗證並行環境數據同步
2. ⏳ 分析觀察速度影響
3. ⏳ 改進 Loss 函數
4. ⏳ 添加訓練監控儀表板

### 優先級 3（未來考慮）
1. ⏳ 模型集成實驗
2. ⏳ 自動超參數搜索
3. ⏳ 分佈式訓練支持

---

## 性能預期

### GPU vs CPU（RTX 3060 Ti）
- **訓練速度**: 預期提升 3-5x
- **Batch size**: CPU 64 → GPU 256
- **並行環境**: CPU 4 → GPU 8
- **總體吞吐量**: 預期提升 10-15x

### 改進的獎勵塑造
- **收斂速度**: 預期提升 2-3x
- **穩定性**: 減少 50% 的抖動
- **最終性能**: 預期提升 20-30%

---

## 問題排查

### 如果 GPU 不可用
```python
import torch
print(torch.cuda.is_available())  # 應該是 True
print(torch.version.cuda)          # 檢查 CUDA 版本
```

### 如果訓練不穩定
1. 降低學習率：`lr: 2.5e-4 → 1e-4`
2. 增加 clip 範圍：`clip_eps: 0.2 → 0.3`
3. 降低 entropy 係數：`ent_coef: 0.01 → 0.001`

### 如果 OOM（記憶體不足）
1. 降低 batch size：`256 → 128`
2. 降低 horizon：`4096 → 2048`
3. 減少並行環境：`8 → 4`

---

## 總結

本次更新主要解決了：
1. ✅ UI 重疊問題（座標系統修正）
2. ✅ 訓練配置對話框（用戶友好）
3. ✅ GPU 優化配置（RTX 3060 Ti）
4. ✅ Checkpoint 管理工具
5. ✅ 改進的獎勵塑造

下一階段重點：
- 實現 checkpoint 載入
- 驗證 GPU 訓練效果
- 改進 Loss 函數持續優化
- 分析並行環境數據同步

預期效果：
- 訓練速度提升 10-15x
- 收斂速度提升 2-3x
- 最終性能提升 20-30%
