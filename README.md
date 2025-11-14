```markdown
# Train Game

基於 Pygame 的 Flappy-like 遊戲環境，整合 **PPO / SAC / DQN / Double DQN / TD3** 等強化學習演算法，可在單一 UI 中切換訓練，也能透過 `run_multi_train.py` 同步觀看多個演算法的訓練進度。
```

## 專案說明

- `game/`：遊戲環境、Pygame UI、訓練視窗 (TrainingWindow)、AlgorithmManager。
- `agents/`：神經網路、PPO Trainer、SAC、DQN/Double-DQN、TD3、共用 Replay Buffer。
- `checkpoints/`：訓練紀錄、排行榜、模型快照。
- `run_game.py`：單視窗 GameUI 入口；`run_multi_train.py`：一次啟動多視窗訓練。
- `tests/`：pytest 套件，包含環境/碰撞/簡易 UI smoke 測試。

## 核心功能

1. **多演算法管理**：右側演算法面板可單獨啟動/停止 PPO、SAC、DQN、Double DQN、TD3，並即時切換觀察對應的代理人。
2. **多視窗訓練**：`run_multi_train.py` 會為 `ALGORITHMS_TO_RUN` 內的每個 key 開啟專屬 UI，方便比較學習策略。
3. **訓練視覺化**：每個演算法皆可連動獨立 TrainingWindow，顯示神經網路拓撲與 policy/value/entropy loss。
4. **排行榜/分數追蹤**：AI 破紀錄時會以 `AI-演算法名稱` 格式寫入 `checkpoints/scores.json`，玩家/AI 成績可在 UI 右側查看 Top 5。
5. **背景訓練 + UI 串流**：AlgorithmManager 以背景執行緒訓練指定演算法，並將 metrics 傳回 UI，減少手動整合負擔。

## 環境設定（Windows / PowerShell）

1. 建立並啟用虛擬環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 安裝依賴（依 `requirements.txt`）

```powershell
pip install -r requirements.txt
```

若需 CUDA 版本 PyTorch，請到 <https://pytorch.org/> 取得對應指令；純 CPU 可直接：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3. 執行測試（單元 + 整合 + UI smoke）

```powershell
python -m pytest -q
```

4. 選擇性：先跑一個短版 PPO 訓練確認依賴

```powershell
python -c "from agents.pytorch_trainer import PPOTrainer; t=PPOTrainer(); t.train(total_timesteps=2000)"
```

5. 啟動 pre-commit hooks（確保格式一致）

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 訓練變數與獎勵

- **狀態** $s_t = [y, v_y, x_{obs}, y_{gap\_top}, y_{gap\_bottom}]$（均已正規化）。
- **動作** $a_t \in \{0,1\}$：0 = 不跳、1 = 跳。
- **獎勵**：
   - 通過障礙：$r_{pass} = +5$
   - 碰撞或飛出上下界：$r_{collision} = -5$
   - 其他時間步：0（已移除時間懲罰）。

## 訓練公式補充

### PPO（含 GAE）

折扣回報：
$$G_t = \sum_{k=0}^{\infty} \gamma^{k} r_{t+k}$$

優勢估計（GAE）：
\begin{aligned}
\delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t) \\
\hat{A}_t &= \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}
\end{aligned}

剪裁目標：
\begin{aligned}
L^{\text{CLIP}}(\theta) = - \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
\end{aligned}
其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。

值函數與熵項：
\begin{aligned}
L^{VF} &= \mathbb{E}_t[(V_\theta(s_t)-G_t)^2] \\
S[\pi_\theta] &= - \sum_a \pi_\theta(a|s_t)\log\pi_\theta(a|s_t)
\end{aligned}

總損失：
$$L = L^{\text{CLIP}} + c_{vf}L^{VF} - c_{ent} S[\pi_\theta]$$

### DQN / Double DQN（QLearningTrainer）

經驗回放樣本的目標值：
\begin{aligned}
y_t^{\text{DQN}} &= r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a') \\
y_t^{\text{DDQN}} &= r_t + \gamma Q_{\theta^-}\!\left(s_{t+1}, \arg\max_{a'} Q_{\theta}(s_{t+1}, a')\right)
\end{aligned}

平方損失：
$$L(\theta) = \mathbb{E}[(y_t - Q_{\theta}(s_t, a_t))^2]$$

### SAC（離散版）

Critic 目標：
\begin{aligned}
J_Q &= \mathbb{E}\left[(Q_{\phi}(s_t,a_t) - y_t)^2\right] \\
y_t &= r_t + \gamma \mathbb{E}_{a_{t+1}\sim\pi}\left[\min(Q_{\bar{\phi}}(s_{t+1},a_{t+1})) - \alpha\log\pi(a_{t+1}|s_{t+1})\right]
\end{aligned}

Actor 目標：
$$J_\pi = \mathbb{E}_{s_t\sim D}\left[\mathbb{E}_{a_t\sim\pi}\left[\alpha \log \pi(a_t|s_t) - Q_{\phi}(s_t,a_t)\right]\right]$$

雙網路軟更新：
$$\bar{\phi} \leftarrow \tau \phi + (1-\tau) \bar{\phi}$$

### TD3（連續版，供比較）

- 兩個 Critic 取最小值防止 Q-value 高估。
- 延遲 Actor 更新與 target policy smoothing：
\begin{aligned}
	ilde{a} &= \operatorname{clip}(\pi_{\theta^-}(s_{t+1}) + \epsilon, a_{low}, a_{high}) \\
y_t &= r_t + \gamma \min_i Q_{\phi_i^-}(s_{t+1}, \tilde{a})
\end{aligned}

## 功能說明

- **演算法控制面板**：顯示狀態（初始化/訓練中/儲存中）、迭代次數、啟停按鈕、快捷鍵 (1~4)。
- **Loss 視覺化**：側邊欄內嵌小型多序列圖，同步 TrainingWindow 資料。
- **排行榜**：AI 成績以 `AI-演算法` 命名，並記錄訓練 iteration。
- **向量化環境**：PPO 可設定並行 env 數（UI 內「並行環境」按鈕輪替）。
- **速度控制**：觀戰速度可於 UI 快速切換 x1/x2/x4/x8。

## 如何使用

### 1. 執行單視窗 UI

```powershell
python run_game.py
```

- 選擇「人類遊玩」以鍵盤體驗；
- 選擇「AI 訓練」啟動背景訓練，並使用右側面板切換演算法；
- 「排行榜」可查看歷史 Top 分數，再次點擊返回主選單。

### 2. 同步觀察多個演算法

```powershell
python run_multi_train.py
```

- 修改 `run_multi_train.py` 中的 `ALGORITHMS_TO_RUN` 以調整要開啟的演算法；
- 每個演算法會在獨立行程中開啟 GameUI，採 `spawn` 啟動方式避免互相干擾。

### 3. 直接調用 Trainer（無 UI）

```powershell
python -c "from agents.sac_trainer import SACTrainer; SACTrainer().train(total_timesteps=10000)"
```

或在互動式 notebook / script 中：

```python
from agents.q_learning_trainer import QLearningTrainer

trainer = QLearningTrainer(mode="double_dqn")
trainer.train(total_timesteps=5000)
```

### 4. Git 流程示例

```powershell
git add .
git commit -m "新增多演算法訓練器和使用者介面更新"
git push
```

## 常見問題

- **Pygame 無法啟動**：請在有 GUI 的環境執行，或確認 `SDL_VIDEODRIVER` 未被設為 `dummy`。
- **多演算法同時啟動卡住**：建議以 `run_multi_train.py` 分多個行程啟動，或在 UI 內一次只啟動一個演算法。
- **pre-commit 未通過**：重新執行 `pre-commit run --all-files`，依提示修復 `ruff`/`black`/`isort` 問題。

---

有任何新的需求（例如增加演算法、擴充視覺化、或接入雲端訓練）都可以開 Issue 或直接留言，持續迭代這個教學/展示專案。說明文件若需其他語言版本也歡迎提出！

````

