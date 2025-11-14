# 訓練輸出目錄說明

這個目錄包含所有演算法訓練過程的詳細數據，專門設計給 AI Coding 工具分析和優化使用。

## 目錄結構

```
outputs/
├── PPO/
│   ├── config/
│   │   └── hyperparams_YYYYMMDD_HHMMSS.json  # 超參數配置
│   ├── metrics/
│   │   ├── session_YYYYMMDD_HHMMSS.jsonl      # 訓練指標 (每次迭代)
│   │   └── episodes_YYYYMMDD_HHMMSS.jsonl     # 回合數據 (每次遊戲)
│   ├── analysis/
│   │   └── analysis_YYYYMMDD_HHMMSS.md        # 分析報告
│   └── training_summary.json                   # 訓練總結
├── SAC/
│   └── ... (同上結構)
├── DQN/
│   └── ...
├── Double DQN/
│   └── ...
└── TD3/
    └── ...
```

## 檔案格式說明

### 1. 超參數配置 (`hyperparams_*.json`)

記錄訓練時使用的所有超參數：

```json
{
  "algorithm": "PPO",
  "session_id": "20251114_133000",
  "timestamp": "2025-11-14T13:30:00",
  "hyperparameters": {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "batch_size": 64,
    "n_envs": 4
  }
}
```

### 2. 訓練指標 (`session_*.jsonl`)

JSONL 格式（每行一個 JSON 物件），記錄每次訓練迭代的指標：

```jsonl
{"iteration": 100, "timestamp": 120.5, "datetime": "2025-11-14T13:32:00", "policy_loss": 0.123, "value_loss": 0.456, "entropy": 0.789, "mean_reward": 5.2}
{"iteration": 110, "timestamp": 132.1, "datetime": "2025-11-14T13:32:12", "policy_loss": 0.118, "value_loss": 0.442, "entropy": 0.765, "mean_reward": 5.8}
```

**字段說明**：
- `iteration`: 訓練迭代次數
- `timestamp`: 相對於會話開始的秒數
- `datetime`: ISO 8601 格式的時間戳記
- `policy_loss`: Policy 網路損失
- `value_loss`: Value 網路損失
- `entropy`: 熵值（探索程度指標）
- `mean_reward`: 平均獎勵

### 3. 回合數據 (`episodes_*.jsonl`)

JSONL 格式，記錄每次遊戲回合的結果：

```jsonl
{"episode": 1, "score": 10, "steps": 42, "iteration": 100, "timestamp": 121.0, "datetime": "2025-11-14T13:32:01"}
{"episode": 2, "score": 15, "steps": 58, "iteration": 102, "timestamp": 125.2, "datetime": "2025-11-14T13:32:05"}
```

**字段說明**：
- `episode`: 回合編號
- `score`: 遊戲得分
- `steps`: 回合步數
- `iteration`: 對應的訓練迭代次數

### 4. 訓練總結 (`training_summary.json`)

會話結束時生成的統計摘要：

```json
{
  "algorithm": "PPO",
  "session_id": "20251114_133000",
  "final_iteration": 5000,
  "total_episodes": 250,
  "duration_seconds": 3600.5,
  "duration_formatted": "1.00h",
  "average_score": 12.5,
  "max_score": 45,
  "end_time": "2025-11-14T14:30:00",
  "files": {
    "metrics": "outputs/PPO/metrics/session_20251114_133000.jsonl",
    "episodes": "outputs/PPO/metrics/episodes_20251114_133000.jsonl",
    "hyperparameters": "outputs/PPO/config/hyperparams_20251114_133000.json"
  }
}
```

### 5. 分析報告 (`analysis_*.md`)

Markdown 格式的可讀分析報告，包含：
- 訓練統計摘要
- 性能趨勢分析
- 改進建議
- 數據檔案位置

## 如何使用

### 在訓練器中啟用日誌記錄

```python
from utils.training_logger import create_logger

# 創建日誌記錄器
logger = create_logger("PPO")

# 記錄超參數
logger.log_hyperparameters({
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "epsilon": 0.2,
    # ...更多參數
})

# 在訓練循環中記錄指標
for iteration in range(max_iterations):
    # ... 訓練程式碼 ...
    
    logger.log_metrics(iteration, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "mean_reward": mean_reward
    })

# 記錄遊戲回合
logger.log_episode(
    episode=episode_num,
    score=final_score,
    steps=total_steps,
    iteration=current_iteration
)

# 訓練結束時生成總結
logger.finalize(final_iteration=iteration, total_episodes=episode_count)
logger.generate_analysis_report()
```

### 讀取和分析數據

```python
import json

# 讀取指標數據
with open("outputs/PPO/metrics/session_20251114_133000.jsonl", "r") as f:
    metrics = [json.loads(line) for line in f]

# 分析 loss 趨勢
policy_losses = [m["policy_loss"] for m in metrics]
print(f"平均 Policy Loss: {sum(policy_losses) / len(policy_losses):.4f}")

# 讀取回合數據
with open("outputs/PPO/metrics/episodes_20251114_133000.jsonl", "r") as f:
    episodes = [json.loads(line) for line in f]

scores = [ep["score"] for ep in episodes]
print(f"平均分數: {sum(scores) / len(scores):.2f}")
print(f"最高分數: {max(scores)}")
```

## AI Coding 工具建議

當你需要優化演算法時，可以：

1. **檢查 `training_summary.json`** 快速了解整體表現
2. **分析 `session_*.jsonl`** 觀察 loss 和 reward 的變化趨勢
3. **對比 `hyperparams_*.json`** 找出表現最好的參數配置
4. **閱讀 `analysis_*.md`** 獲取自動生成的改進建議

## 清理舊數據

定期清理過時的訓練記錄以節省空間：

```bash
# 刪除 7 天前的會話數據
find outputs/ -name "session_*.jsonl" -mtime +7 -delete
find outputs/ -name "episodes_*.jsonl" -mtime +7 -delete
find outputs/ -name "analysis_*.md" -mtime +7 -delete
```

---

*此檔案用於幫助 AI Coding 工具理解訓練數據結構*
