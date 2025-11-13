# 專案功能完整度 Review

**日期**: 2025-11-13  
**狀態**: 發現 AI 遊玩模式無回應問題

---

## 🎯 當前問題診斷

### 問題描述
用戶點擊「AI 遊玩」按鈕後，視窗**無回應/當機**。

### 根本原因分析

經過代碼檢查，發現以下關鍵問題：

#### 1. **AI Agent 未經訓練** ⚠️
```python
# game/ui.py line 360
if self.agent is None:
    try:
        self.agent = PPOAgent()  # ← 創建的是「未訓練」的新 agent
    except Exception:
        self.agent = None
```

**問題**：
- `PPOAgent()` 創建的網絡權重是**隨機初始化**的
- 沒有載入任何訓練好的模型
- 這個 agent 根本不會玩遊戲，只會做隨機決策

#### 2. **訓練器與 Agent 沒有連接** 🔴
```python
# game/ui.py line 371-383
# 啟動背景訓練
trainer = PPOTrainer()
training_env = GameEnv()
self.start_trainer(trainer, env=training_env, ...)  # ← 在背景訓練

# BUT...
# game/ui.py line 626-631 (主循環)
if self.mode == "AI":
    if self.agent is not None:
        a, _, _ = self.agent.act(s)  # ← 使用的是「未訓練」的 self.agent
        s, r, done, _ = self.env.step(a)
```

**問題**：
- 背景訓練器在 `training_env` 上訓練 `trainer.net`
- UI 主循環使用的是 `self.agent`（完全不同的網絡實例）
- **兩者完全獨立，訓練的結果不會影響遊玩的 agent**

#### 3. **無回應的真正原因** 💀

最可能的原因是：
1. **未訓練的 agent 表現極差**：隨機權重導致 agent 立刻死亡
2. **快速重複 reset**：每次死亡後立即 reset，進入無限循環
3. **UI 主線程阻塞**：在 `self.env.step()` 和處理死亡的循環中卡死
4. **pygame 事件處理不及時**：主循環太忙，無法處理用戶輸入

---

## ✅ 已完成的功能

### 1. **人類遊玩模式** ✅
- [x] 完整的遊戲邏輯
- [x] 空白鍵跳躍
- [x] 碰撞檢測
- [x] 分數系統
- [x] Game Over 對話框
- [x] 暫停功能（ESC）
- [x] 排行榜記錄

### 2. **遊戲環境** ✅
- [x] `GameEnv` 實現完整
- [x] 狀態觀察（5 維）
- [x] 動作空間（0=不動, 1=跳躍）
- [x] 獎勵函數
- [x] 障礙物生成
- [x] 物理模擬

### 3. **神經網絡架構** ✅
- [x] `ActorCritic` MLP [5, 64, 64, 2]
- [x] Policy head（logits）
- [x] Value head

### 4. **PPO 訓練器** ✅
- [x] `PPOTrainer` 實現完整
- [x] GAE 優勢估計
- [x] PPO clip 更新
- [x] TensorBoard 記錄
- [x] 模型 checkpoint 保存
- [x] Graceful shutdown（stop_event）

### 5. **UI 系統** ✅
- [x] 主選單
- [x] 遊玩區域渲染
- [x] 右側資訊面板
- [x] 訓練視窗（獨立線程）
- [x] Loss 曲線繪製
- [x] 排行榜顯示

---

## ❌ 缺失/問題的功能

### 🔴 高優先（阻塞使用）

#### 1. **AI 遊玩模式無法正常運作**
**狀態**: 會導致無回應/當機

**缺失內容**：
- [ ] 沒有載入訓練好的模型
- [ ] 訓練器訓練的網絡 ≠ 遊玩用的 agent
- [ ] 未訓練的 agent 表現極差導致快速死亡循環

**需要實作**：
```python
# 選項 A：使用訓練好的模型
class GameUI:
    def __init__(self):
        self.agent = None
        self.trained_model_path = "checkpoints/best_model.pth"
    
    def handle_click(self, pos):
        if self.btn_ai.collidepoint(pos):
            # 載入訓練好的模型
            if os.path.exists(self.trained_model_path):
                self.agent = PPOAgent()
                self.agent.net.load_state_dict(
                    torch.load(self.trained_model_path)
                )
                print(f"載入模型: {self.trained_model_path}")
            else:
                print("警告：沒有找到訓練好的模型，將使用隨機 agent")
                self.agent = PPOAgent()

# 選項 B：讓訓練器更新 UI 的 agent
class GameUI:
    def start_trainer(self, trainer, ...):
        # 讓 UI 的 agent 共享訓練器的網絡
        self.agent = PPOAgent()
        self.agent.net = trainer.net  # 共享權重
```

#### 2. **模型載入/保存 UI**
**狀態**: 完全缺失

**需要實作**：
- [ ] 選單按鈕：「載入模型」
- [ ] 檔案選擇對話框
- [ ] 模型路徑配置
- [ ] 載入成功/失敗提示

#### 3. **訓練控制**
**狀態**: 部分實作，但不完整

**現有問題**：
- [x] 可以啟動訓練器
- [x] 可以停止訓練器
- [ ] 無法暫停/恢復訓練
- [ ] 無法調整訓練參數（timesteps, lr, etc.）
- [ ] 訓練完成後沒有通知

**需要實作**：
- [ ] 訓練控制面板
- [ ] 暫停/恢復按鈕
- [ ] 參數配置 UI
- [ ] 訓練完成通知

---

### 🟡 中優先（影響體驗）

#### 4. **AI 模式的遊玩體驗**
**問題**：
- [ ] AI 死亡後自動重啟，用戶無法看清楚發生了什麼
- [ ] 沒有「慢動作」或「單步執行」模式
- [ ] 無法查看 AI 的決策過程（action probabilities）

**建議實作**：
```python
# 在 draw_panel 中顯示 AI 決策資訊
if self.mode == "AI" and self.agent:
    # 顯示動作機率
    prob = self.agent.last_action_prob  # 需要在 agent.act() 中記錄
    self.draw_text(f"跳躍機率: {prob:.2%}", ...)
    
    # 顯示價值估計
    value = self.agent.last_value
    self.draw_text(f"狀態價值: {value:.2f}", ...)
```

#### 5. **訓練進度監控**
**現有**：
- [x] TrainingWindow 顯示 loss 曲線
- [x] TensorBoard 記錄

**缺失**：
- [ ] 訓練進度百分比
- [ ] 預計剩餘時間
- [ ] 平均獎勵趨勢
- [ ] 最佳分數記錄

#### 6. **模型管理**
**完全缺失**：
- [ ] 列出所有已保存的模型
- [ ] 比較不同模型的性能
- [ ] 刪除舊模型
- [ ] 重命名模型

---

### 🟢 低優先（優化改進）

#### 7. **配置系統**
- [ ] 遊戲參數配置檔（JSON/YAML）
- [ ] 訓練超參數配置
- [ ] UI 設定（顏色、字體大小等）

#### 8. **錯誤處理**
- [ ] 更完善的異常捕獲
- [ ] 用戶友好的錯誤訊息
- [ ] 錯誤日誌記錄

#### 9. **性能優化**
- [ ] 向量化環境（多進程訓練）
- [ ] GPU 加速檢測和提示
- [ ] 訓練批次大小自適應

#### 10. **測試覆蓋**
- [x] 基本煙霧測試
- [ ] 訓練流程測試
- [ ] UI 互動測試
- [ ] 模型載入/保存測試

---

## 🎯 修復計劃

### 階段 1：緊急修復（立即執行）

**目標**：讓 AI 遊玩模式能夠正常運作

1. **修復訓練器與 Agent 的連接** [15 分鐘]
   ```python
   # 選項 1：共享網絡（實時看到訓練效果）
   def handle_click(self, pos):
       if self.btn_ai.collidepoint(pos):
           # 創建 agent，稍後與訓練器共享網絡
           if self.agent is None:
               self.agent = PPOAgent()
           
           # 啟動訓練器
           if self.trainer_thread is None:
               trainer = PPOTrainer()
               # 讓 agent 使用訓練器的網絡
               self.agent.net = trainer.net
               self.agent.opt = trainer.opt
               self.start_trainer(trainer, ...)
   ```

2. **添加訓練前的預訓練** [10 分鐘]
   ```python
   # 在啟動 AI 模式前，先訓練一小段時間
   print("正在預訓練 agent（500 步）...")
   trainer.train(total_timesteps=500, env=training_env, ...)
   print("預訓練完成，開始遊玩")
   ```

3. **添加 AI 快速重啟的延遲** [5 分鐘]
   ```python
   # 在 run() 中，AI 模式死亡後添加延遲
   if done and self.mode == "AI":
       pygame.time.wait(500)  # 延遲 500ms
       s = self.env.reset()
   ```

### 階段 2：基礎功能（1-2 小時）

4. **實作模型載入** [30 分鐘]
5. **添加訓練控制 UI** [45 分鐘]
6. **顯示 AI 決策資訊** [15 分鐘]

### 階段 3：優化體驗（可選）

7. 模型管理系統
8. 配置檔支援
9. 性能優化

---

## 📊 功能完整度統計

```
總功能數：20
已完成：5 (25%)
部分完成：3 (15%)
缺失/問題：12 (60%)

其中：
🔴 高優先：3 項（阻塞使用）
🟡 中優先：3 項（影響體驗）
🟢 低優先：6 項（優化改進）
```

---

## 🔍 問題根源總結

**AI 遊玩模式無回應的原因**：

1. **架構性問題**：
   - 訓練器訓練的網絡 ≠ UI 使用的 agent
   - 兩者完全獨立，沒有共享權重

2. **實現性問題**：
   - Agent 未訓練（隨機權重）
   - 表現極差導致快速死亡
   - 快速重啟進入無限循環
   - 主線程被阻塞

3. **設計性問題**：
   - 缺少模型載入機制
   - 缺少訓練/遊玩模式分離
   - 缺少錯誤處理和提示

**建議解決方案**：
- **短期**：讓 agent 共享訓練器的網絡 + 添加預訓練 + 延遲重啟
- **長期**：實作完整的模型管理系統 + 訓練/遊玩模式分離

---

## ✅ 下一步行動

1. ⏳ **立即執行**：修復訓練器與 Agent 連接
2. ⏳ **立即執行**：添加預訓練機制
3. ⏳ **立即執行**：添加 AI 重啟延遲
4. 📋 測試 AI 遊玩模式
5. 📋 根據測試結果調整策略
