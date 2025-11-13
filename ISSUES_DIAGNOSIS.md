# 問題診斷報告

**日期**: 2025-11-13  
**問題**: 1) AI 遊玩無回應  2) 碰撞檢測太敏感

---

## 🔍 診斷結果

### 問題 1：AI 遊玩無回應 ❌

**診斷發現**：
- ✅ Agent 可以正常創建和運作
- ✅ agent.act() 每步耗時 < 1ms（非常快）
- ✅ 主循環不會因運算阻塞

**真正原因**：**PyGame 事件處理不及時** 🔴

經過代碼審查發現：
```python
# game/ui.py line 658
if self.mode == "AI":
    if self.agent is not None:
        a, _, _ = self.agent.act(s)
        s, r, done, _ = self.env.step(a)
```

**問題**：未訓練的 agent 會**連續跳躍**（action=1），導致：
1. 球快速上升，撞到天花板
2. 立刻死亡（done=True）
3. 快速 reset，進入下一回合
4. 整個循環太快（< 1 秒），pygame 事件處理來不及
5. 用戶點擊、按鍵等事件累積在隊列中
6. **視窗看起來「無回應」**

**證據**：
- 診斷腳本顯示未訓練 agent 連續 5 次都選擇 action=1（跳躍）
- 這會導致球快速飛出螢幕上方

### 問題 2：碰撞檢測太敏感 ✅

**診斷結果**：
```
UI 渲染的球半徑: 12 像素
環境碰撞判定的 ball_margin: 15 像素
差異: 3 像素
```

**問題**：
- 視覺上球半徑是 12 像素
- 但碰撞判定使用 15 像素
- 結果：**球看起來還沒碰到障礙物，但已經判定碰撞**

**測試結果**：
```
gap_size: 90.89 像素（缺口實際大小）
判定範圍: 60.89 像素（扣除 2 × 15 = 30 像素）
減少了 33% 的可通過空間！
```

---

## ✅ 修復方案

### 修復 1：AI 無回應 - 強制處理 PyGame 事件

**方案**：在 AI 模式的每個循環中，強制處理 pygame 事件

```python
if self.mode == "AI":
    # 強制處理事件，避免視窗無回應
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # ESC 返回選單
                self.running = False
                self.mode = None
    
    if self.agent is not None:
        a, _, _ = self.agent.act(s)
        s, r, done, _ = self.env.step(a)
```

**優勢**：
- ✅ 即使 agent 快速死亡，視窗仍保持響應
- ✅ 可以按 ESC 退出 AI 模式
- ✅ 可以關閉視窗

### 修復 2：碰撞檢測敏感度 - 統一球半徑

**方案 A**（推薦）：將 ball_margin 改為 12，與視覺一致

```python
# game/environment.py line 138
ball_margin = 12.0  # 改為 12（與 UI 渲染的球半徑一致）
```

**方案 B**：將 UI 球半徑改為 15，與判定一致

```python
# game/ui.py line 149
pygame.draw.circle(self.screen, (255, 200, 50), (ball_x, ball_y), 15)
```

**推薦方案 A**，因為：
- 球半徑 12 看起來更合適
- 改動最小（只改一個數字）
- 不影響視覺效果

**效果對比**：
```
修復前：
  gap_size: 90 px → 判定範圍: 60 px（減少 33%）
  
修復後：
  gap_size: 90 px → 判定範圍: 66 px（減少 27%）
  增加 10% 的通過空間！
```

### 額外修復：添加 AI 模式的最小延遲

即使事件處理正常，AI 快速死亡循環仍然不好。添加最小渲染間隔：

```python
if done and self.mode == "AI":
    # ... 現有的死亡處理 ...
    pygame.time.wait(300)  # 已經有了，保持
    
    # 但還要確保渲染
    self.screen.fill(self.BG_COLOR)
    self.draw_playfield(s)
    self.draw_panel()
    pygame.display.flip()
```

---

## 🎯 修復優先級

1. **高優先**：修復碰撞敏感度（改 ball_margin = 12.0）[1 分鐘]
2. **高優先**：AI 模式處理事件（避免無回應）[5 分鐘]
3. **中優先**：改善 AI 死亡時的視覺反饋 [已完成]

---

## 📊 預期效果

### 修復後的 AI 模式：
- ✅ 視窗保持響應
- ✅ 可以按 ESC 返回選單
- ✅ 可以隨時關閉視窗
- ✅ 即使 agent 表現差，體驗也不會糟糕

### 修復後的碰撞檢測：
- ✅ 視覺與判定一致
- ✅ 不會有「明明沒碰到卻死了」的感覺
- ✅ 增加 10% 的通過空間
- ✅ 遊戲難度更合理

---

## 🧪 測試計劃

### 測試 1：碰撞檢測
1. 運行遊戲，選擇「人類遊玩」
2. 故意讓球接近障礙物邊緣
3. **驗證**：只有在視覺上明顯碰撞時才 Game Over

### 測試 2：AI 模式響應
1. 運行遊戲，選擇「AI 遊玩」
2. 等待 2-3 秒
3. **驗證**：
   - 視窗標題欄顯示「回應中」（不是「無回應」）
   - 可以移動視窗
   - 按 ESC 能返回選單

### 測試 3：AI 訓練效果
1. 繼續觀察 AI 遊玩 1-2 分鐘
2. **驗證**：
   - Agent 表現逐漸改善
   - 不會一直跳躍
   - 存活時間變長

---

## 📝 根本原因總結

### AI 無回應的真相：
不是代碼阻塞，而是：
1. 未訓練 agent 行為極端（連續跳躍）
2. 快速死亡導致快速循環
3. **PyGame 事件處理在主循環末尾，但 AI 模式中途就 continue**
4. 事件累積在隊列中未處理
5. 視窗看起來「凍結」

### 碰撞太敏感的真相：
- 視覺與判定不一致
- ball_margin (15) > 球半徑 (12)
- 玩家看到的 ≠ 實際判定的

---

## ✅ 立即執行

現在執行修復...
