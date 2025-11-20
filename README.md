# 🎮 Train Game - 深度強化學習訓練平台

<div align="center">

**基於 Pygame + PyTorch 的 Flappy-like 遊戲環境，採用 PPO (Proximal Policy Optimization) 演算法**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 目錄

- [功能說明](#-功能說明)
- [系統結構圖](#-系統結構圖)
- [專案特徵設計](#-專案特徵設計)
- [訓練公式詳解](#-訓練公式詳解)
- [PPO 介紹](#-ppo-介紹)
- [損失函數參數意義](#-損失函數參數意義)

---

## 🚀 功能說明

本專案是一個完整的深度強化學習 (Deep Reinforcement Learning) 實驗平台，專為訓練 AI 玩類 Flappy Bird 遊戲而設計。

### 核心功能
1.  **高效並行訓練**：
    - 使用 **32 個並行環境 (Parallel Environments)** 進行訓練，極大化經驗收集速度。
    - 額外配置 **4 個獨立評估環境**，用於定期測試模型真實性能。
2.  **強大的 PPO 演算法**：
    - 採用 **Stable-Baselines3** 的 PPO 實作。
    - 客製化神經網路架構 (MLP 512x512)，專為高精度控制優化。
3.  **智能訓練機制**：
    - **Auto-Resume (自動恢復)**：訓練中斷後，可自動讀取最新的 Checkpoint 繼續訓練，無需重頭開始。
    - **動態難度調整**：隨著 AI 變強，遊戲速度與障礙物難度會自動提升。
4.  **完整視覺化與監控**：
    - 支援 **TensorBoard** 即時監控 Loss、Reward、Entropy 等指標。
    - 提供 **Replay (重播)** 功能，可視覺化觀看 AI 的實際操作表現。

---

## 📊 系統結構圖

### 1. AOV 結構圖 (System Architecture)

這展示了數據在系統中的流動與處理順序：

```mermaid
graph TD
    A[開始訓練 (Start)] --> B{檢查 Checkpoint?};
    B -- 是 --> C[載入舊模型 & 統計數據];
    B -- 否 --> D[初始化新模型];
    C --> E[並行環境 (32 Envs)];
    D --> E;
    
    subgraph Training_Loop [訓練迴圈]
        E -->|收集軌跡 (Rollout)| F[經驗緩衝區 (RolloutBuffer)];
        F -->|計算優勢 (GAE)| G[PPO 演算法核心];
        G -->|計算 Loss| H[策略更新 (Policy Update)];
        H -->|更新權重| I[神經網路 (Actor-Critic)];
    end
    
    I -->|定期評估| J[評估環境 (4 Envs)];
    J -->|保存最佳模型| K[Best Model Checkpoint];
    I -->|更新策略| E;
```

### 2. Breakdown 結構圖 (File Structure)

專案檔案模組化設計如下：

```text
Train_Game/
├── game/                   # 🎮 遊戲核心模組
│   ├── environment.py      # 物理引擎、獎勵函數、狀態定義
│   ├── ui.py               # 畫面渲染、使用者介面
│   └── vec_env.py          # 向量化環境包裝器
├── rl/                     # 🤖 強化學習模組
│   ├── train_sb3.py        # PPO 訓練主程式 (Entry Point)
│   └── game2048_env.py     # Gymnasium 介面適配器
├── agents/                 # 🧠 代理人模組
│   ├── ppo_agent.py        # PPO 代理人類別
│   └── networks.py         # 神經網路定義
├── models/                 # 💾 模型存檔
│   ├── best_model.zip      # 歷史最佳模型
│   └── vec_normalize.pkl   # 狀態標準化統計數據
└── logs/                   # 📈 訓練日誌 (TensorBoard)
```

---

## 🎨 專案特徵設計

### 1. 觀察空間 (Observation Space)
AI 並非直接看像素 (Pixels)，而是接收一個 **7 維的正規化向量**，這讓訓練更高效：

| 參數 | 說明 | 範圍 (正規化後) |
| :--- | :--- | :--- |
| `y` | 玩家垂直位置 | [0, 1] |
| `vy` | 玩家垂直速度 | [-1, 1] |
| `x_obs` | 距離下一個障礙物的水平距離 | [0, 1] |
| `gap_top` | 縫隙上緣位置 | [0, 1] |
| `gap_bottom` | 縫隙下緣位置 | [0, 1] |
| `rel_top` | 玩家距離縫隙上緣的相對距離 | [-1, 1] |
| `rel_bottom` | 玩家距離縫隙下緣的相對距離 | [-1, 1] |

### 2. 動作空間 (Action Space)
- **類型**：離散 (Discrete)
- **動作**：
    - `0`：**不跳** (受重力下墜)
    - `1`：**跳躍** (獲得向上速度)

### 3. 獎勵函數設計 (Reward Shaping)
這是引導 AI 行為的關鍵：

| 行為 | 獎勵值 | 設計目的 |
| :--- | :--- | :--- |
| **通關 (Win)** | **+1000.0** | 鼓勵 AI 追求最終勝利 (6666分)，而不僅僅是生存。 |
| **通過障礙** | **+5.0** | 給予階段性成就感，引導 AI 穿越縫隙。 |
| **位置對齊** | +0.08 (max) | 引導 AI 盡量保持在縫隙中央，減少碰撞風險。 |
| **時間懲罰** | **-0.01 / step** | 迫使 AI 不要猶豫，雖然卷軸是固定的，但這能減少無效操作。 |
| **碰撞/出界** | **-5.0** | 強烈懲罰死亡，讓 AI 學會避開危險。 |

### 4. 神經網路架構
- **架構**：MLP (多層感知機)
- **規模**：`[512, 512]` (兩層隱藏層，每層 512 個神經元)
- **激活函數**：ReLU

---

## 🧾 訓練公式詳解

在 PPO 訓練過程中，公式的計算是有**先後順序**的。以下依序說明：

### 第一步：計算折扣回報 (Discounted Return)
首先，我們要算出「這一場遊戲到底拿了多少分」。

$$
G_t = \sum_{k=0}^{\infty} \gamma^{k} r_{t+k}
$$

*   **含義**：從時間點 $t$ 開始，把未來所有的獎勵 $r$ 加總起來。
*   **$\gamma$ (Gamma)**：折扣因子。讓未來的獎勵「打折」，越遠的獎勵價值越低。

### 第二步：計算優勢 (Advantage Estimation - GAE)
接著，我們要評估「這一步走得比預期好多少？」。

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

*   **含義**：這是 PPO 判斷動作好壞的核心。
*   **$\delta_t$ (TD Error)**：實際發生的事 - 預期會發生的事。
*   **$\lambda$ (Lambda)**：平滑因子，用來平衡偏差與變異數。

### 第三步：計算概率比率 (Probability Ratio)
比較「新策略」和「舊策略」對同一個動作的看法。

$$
r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

*   **含義**：如果 $r_t > 1$，代表新策略比舊策略更喜歡這個動作。

### 第四步：計算裁剪目標 (Clipped Objective)
這是 PPO 防止「學壞」的關鍵步驟。

$$
L^{CLIP}(\theta) = -\mathbb{E}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]
$$

*   **含義**：如果新策略改變太大（超過 $\epsilon$ 範圍），就強制截斷，不讓它更新那麼多。這保證了訓練的穩定性。

### 第五步：計算總損失 (Total Loss)
最後，將所有目標結合，變成神經網路要優化的最終數字。

$$
L = L^{CLIP} + c_{vf} L^{VF} - c_{ent} S
$$

---

## 🧠 PPO 介紹

**Proximal Policy Optimization (近端策略優化)**

PPO 是目前最流行的深度強化學習演算法之一，由 OpenAI 提出。

### 核心概念
1.  **On-Policy (在線策略)**：AI 一邊玩，一邊學習。它只能使用「自己當前策略」產生的經驗來學習，不能使用別人的或過去的經驗。
2.  **Trust Region (信任區域)**：PPO 的核心思想是「不要改太多」。它限制了每次策略更新的幅度，確保新的策略不會偏離舊策略太遠。
3.  **Clipping (裁剪)**：這是 PPO 實現信任區域的方法。它直接把概率比率 $r_t(\theta)$ 限制在 $[1-\epsilon, 1+\epsilon]$ 之間，簡單粗暴但非常有效。

### 為什麼選擇 PPO？
- **穩定性高**：不容易因為參數設錯而導致訓練崩潰。
- **調參簡單**：相比於 DQN 或 DDPG，PPO 對超參數不那麼敏感。
- **適用性廣**：既能玩離散動作遊戲 (如 Super Mario)，也能玩連續動作遊戲 (如 機器人控制)。

---

## 📉 損失函數參數意義

在總損失函數 $L = L^{CLIP} + c_{vf} L^{VF} - c_{ent} S$ 中，各參數代表：

| 參數符號 | 英文名稱 | 中文含義 | 典型值 | 作用詳解 |
| :--- | :--- | :--- | :--- | :--- |
| **$\gamma$** | Gamma | 折扣因子 | 0.99 | 決定 AI 有多「遠見」。接近 1 代表 AI 很在意未來的死活；接近 0 代表 AI 只想現在立刻拿分。 |
| **$\lambda$** | Lambda | GAE 因子 | 0.95 | 平衡「估計值」與「實際值」的權重。用來計算優勢函數 $\hat{A}_t$。 |
| **$\epsilon$** | Epsilon | 裁剪範圍 | 0.1 ~ 0.2 | **安全閥**。限制策略更新的幅度，防止 AI 因為一次運氣好就過度自信。 |
| **$c_{vf}$** | VF Coef | 價值係數 | 0.5 ~ 1.0 | 決定我們要花多少力氣去訓練 Critic (評論家)。如果太低，AI 會搞不清楚狀況；太高則會忽略策略本身的優化。 |
| **$c_{ent}$** | Ent Coef | 熵係數 | 0.01 | **好奇心**。這個值越高，AI 越喜歡嘗試隨機動作（探索）；這個值越低，AI 越傾向於固守已知的打法（利用）。 |

---
