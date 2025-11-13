```markdown
# train_game

簡易 Flappy-like 遊戲 + RL (PPO) 實驗專案骨架。此 README 專為 Windows (PowerShell) 使用者整理，包含環境建立、測試、以及如何啟動 UI/訓練器的步驟。

快速開始（Windows / PowerShell）

1) 建立並啟用虛擬環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安裝依賴（本專案使用已釘選的版本，請參照 `requirements.txt`）

```powershell
pip install -r requirements.txt
```

注意：如果要使用 PyTorch CUDA 版本，請依照你的 CUDA 版本到 https://pytorch.org/ 取得正確安裝指令；若無 GPU，使用 CPU 版：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3) 執行測試（單元/整合/頭less UI smoke）

```powershell
python -m pytest -q
```

4) 以 smoke 模式啟動 trainer（範例會印出幾個 episode 的結果）

```powershell
python -m agents.trainer
```

5) 啟動 Pygame UI（若在有顯示器的環境）

```powershell
python run_game.py
```

如果你在無頭 (headless) 環境，UI 可能無法啟動，請於本機桌面環境執行。

Git / 提交範例（中文提交訊息）：

```powershell
git add .
git commit -m "新增：PPO agent 與 UI 骨架與測試"
git push
```

啟用 pre-commit hooks（建議在本地環境執行一次以完成 hooks 安裝）：

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

CI badge（若已啟用 GitHub Actions）：

![CI](https://github.com/SmailDot/train_game/actions/workflows/ci.yml/badge.svg)

進階：以 PyTorch 訓練器做短跑驗證（CPU 範例）

```powershell
# 建議先安裝 requirements.txt 內的套件
python -c "from agents.pytorch_trainer import PPOTrainer; t=PPOTrainer(); t.train(total_timesteps=2000)"
```

疑難排解小提示
- 若出現 `pygame` 無法啟動，請確認正在使用有 GUI 的環境並且 `SDL_VIDEODRIVER` 未被設定為 headless。
- pre-commit 若在安裝 hooks 時發生 rev/checkout 問題，嘗試在能連上外部 Git 的網路環境重新執行 `pre-commit install`。

歡迎在本地執行上述命令，有任何錯誤訊息把輸出貼給我，我可以幫你一步步排查。

---
小節：本專案包含 `game/` (環境與 UI)、`agents/` (網路、PPO trainer)、`tests/` (pytest)、以及 CI + pre-commit configs。
```
 
## 變數、狀態與獎勵（Definitions）

- 狀態 (State) S：5 維向量 $[y, v_y, x_{obs}, y_{gap\_top}, y_{gap\_bottom}]$，其中 $y$、$v_y$、$y_{gap\_top}$、$y_{gap\_bottom}$ 已被歸一化至畫面高度，$x_{obs}$ 為障礙物到玩家的相對距離並以最大距離做正規化。
- 動作 (Action) A：二元，0 = 不跳、1 = 跳（一次性向上 impulse）。
- 獎勵 (Reward) R（已實作）：
	- 每步時間懲罰：$r_{step} = -0.1$
	- 通過障礙：$r_{pass} = +5.0$（當障礙橫過玩家位置且球處於 gap 範圍內）
	- 碰撞障礙或超出上下界（掉落或撞上天花板）：$r_{collision} = -5.0$

## PPO 與 GAE 相關公式（摘要）

以下列出訓練中使用的主要公式，方便把 trainer 的行為與 UI 上顯示的數值對應起來。

- 折扣回報（Return）：
$$G_t = \sum_{k=0}^{\infty} \gamma^{k} r_{t+k}$$

- 時間差分與廣義優勢估計（GAE）：
\begin{aligned}
\delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t) \\
\hat{A}_t &= \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
\end{aligned}

- PPO 剪裁目標（Clipped surrogate objective）:
\begin{aligned}
L^{CLIP}(\theta) = - \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
\end{aligned}
其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。

- 值函數損失（Value loss）：
$$L^{VF} = \mathbb{E}_t\left[ (V_\theta(s_t) - R_t)^2 \right]$$

- 熵項（鼓勵探索）：
$$S[\pi_\theta](s_t) = - \sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$$

- 總損失（訓練器內使用，權重可調）：
$$L = L^{CLIP} + c_{vf} L^{VF} - c_{ent} S[\pi]\; .$$

## 在 UI 上顯示的項目

- Model weights heatmap：顯示 actor 層（或提供的權重矩陣）縮放後的 2D 圖示。
- Loss 圖：多維度 (policy / value / entropy / total) 時序曲線（在右側 Panel 的 Loss 區塊）。訓練器若提供歷史值，UI 會即時繪製；否則顯示 placeholder。
- episode 分數：當回合結束時，當前回合分數會加入排行榜（Leaderboard）並在面板中顯示。

若要把其他公式或變數加入 README（例如更細的超參數表格或輸出正規化細節），我可以把它補進來。

## 從 Trainer 到 UI 的即時串接（範例）

UI 已支援由訓練器在背景執行時直接推送訓練指標到畫面：

- UI 方法：`GameUI.update_losses(metrics: dict)` -- 訓練器會以字典形式回報 metrics，常見 keys 包含：
	- `it` (iteration)、`loss` (total loss)、`policy_loss`、`value_loss`、`entropy`、`timesteps`、`mean_reward`、`episode_count`。
- UI 方法：`GameUI.start_trainer(trainer, **train_kwargs)` -- 在背景 thread 執行 `trainer.train(...)`，並自動把 `metrics_callback` 綁到 `ui.update_losses`。

範例：在你的啟動腳本或 `run_game.py` 中可以這樣呼叫：

```python
from agents.pytorch_trainer import PPOTrainer
from game.ui import GameUI

ui = GameUI()
trainer = PPOTrainer()

# 啟動 trainer，在背景執行；UI 會即時顯示 loss/entropy/value/total 與最新數值
ui.start_trainer(trainer, total_timesteps=10000, env=ui.env, log_interval=1)

# 然後啟動 UI loop（主 thread）
ui.run()
```

註：`start_trainer` 會在背景執行 `trainer.train(metrics_callback=ui.update_losses, **train_kwargs)`。
因此也可以直接把 `PPOTrainer.train` 中的 `metrics_callback` 參數指向你自己的回呼函式來做整合（例如紀錄到外部監控系統）。

