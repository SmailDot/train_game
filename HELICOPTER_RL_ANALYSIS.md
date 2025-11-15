# 🚁 穿越障礙 PPO 訓練分析

本文件完整記錄我們在「穿越障礙」遊戲上使用 Stable-Baselines3 PPO 的設計、訓練流程、最佳化後的參數與驗證證據。所有內容皆取自本倉庫（`rl/`, `game/`, `logs/`, `outputs/`），不再引用其他專案。

---

## 1. 系統概覽

| 面向 | 說明 |
| --- | --- |
| 遊戲 | `game/environment.py` 的穿越障礙玩法：飛行器可做「跳 / 不跳」兩種動作以穿越動態間隙。 |
| Gym 環境 | `rl/game2048_env.py`（沿用舊檔名，但已綁定穿越障礙遊戲），觀察空間為 5 維連續向量 `(y, vy, x_obs, gap_top, gap_bottom)`，動作空間 `Discrete(2)`。 |
| 演算法 | `stable_baselines3.PPO` (`MlpPolicy`)。 |
| 並行訓練 | `make_vec_env` 建立 32 個 VecEnv + `VecMonitor`，並使用 `VecNormalize` 做觀察/獎勵標準化。 |
| 主要指令 | `python rl/train_sb3.py --target 6666 --n-envs 32 --total-timesteps 5000000 --seed 42`。 |
| 監控 & 證據 | TensorBoard run `logs/tensorboard/PPO_15`、最佳模型 `best_model/best_model.zip`、錄影 `outputs/videos/ppo_clear_seed150/`。 |

---

## 2. PPO 管線模組

| 模組/檔案 | 角色 |
| --- | --- |
| `rl/train_sb3.py:create_envs` | 建立 VecEnv、初始化/載入 VecNormalize 統計、支援 `render_mode`。 |
| `WinCallback` | 監控 `info['win']`，記錄破關次數與最高分。 |
| `EpisodeStatsCallback` | 將 `passed_count`、`scroll_speed`、`alignment_score`、`win` 等指標寫入 TensorBoard (`env/*`)。 |
| `AdaptiveEntropyCallback` | 以 4,096 步移動視窗觀察 win rate，自動於 0.004–0.012 間調整 `ent_coef`。 |
| `CheckpointCallback` + `EvalCallback` | 定期保存 `checkpoints/ppo_game2048_*` 並以 deterministic reward 選出 `best_model/best_model.zip`。 |
| `rl/eval_sb3.py` | 讀取 checkpoint、載入 VecNormalize、可錄影與輸出統計 JSON。 |

---

## 3. 神經網路節點與連線

![Connected PPO network](outputs/plots/PPO_policy_architecture.png)

- **輸入層**：5 個節點，分別對應穿越障礙的高度、速度與前方障礙幾何資訊。
- **共用隱藏層**：3 層、每層 256 ReLU；圖中節點彼此全連接，代表 Dense 結構。
- **Actor Head**：2 個 logits → softmax，輸出「跳 / 不跳」機率。
- **Critic Head**：1 個 state-value 節點，提供 GAE 與 value loss。

該拓撲由 `create_model()` 的 `policy_kwargs['net_arch']` 定義，確保 actor/critic 共享 backbone 後各自輸出。

---

## 4. Loss 折線與意義

![Loss curves](outputs/plots/PPO_15_losses.png)

| 線條 | 用途 | 訓練觀察 |
| --- | --- | --- |
| `train/loss` | PPO surrogate 目標（含 policy、value、entropy） | 由 0.059 下降至 ~0.053，顯示策略逐步收斂。 |
| `train/value_loss` | Critic MSE | 維持 0.035–0.04，價值函數穩定。 |
| `train/policy_gradient_loss` | Actor 梯度項 | 最終 -4.8e-4，代表裁剪約束讓梯度趨近 0。 |
| `train/entropy_loss` | 負熵（-ent_coef * entropy） | 在 -0.108 ~ -0.09，AdaptiveEntropy 保留適度探索。 |

效能折線 (`outputs/plots/PPO_15_performance.png`) 顯示 `rollout/ep_rew_mean` 從 16 提升到 787；`env/win_rate` 在訓練 rollouts 中為 0，但在獨立評估（§6）可達 52.5% win rate。

---

## 5. 最佳超參數（`get_training_config('6666')`）

| 類別 | 參數 | 值 | 說明 |
| --- | --- | --- | --- |
| 設備 | `device` | `cuda` (fallback `cpu`) | 自動偵測硬體。 |
| 網路 | `hidden_dim` | 256 | 三層共享 ReLU。 |
| 學習率 | `learning_rate` | 線性 1e-4 → 5e-5 | `make_linear_schedule`。 |
| 折扣/GAE | `gamma` / `gae_lambda` | 0.99 / 0.95 | 平衡長期獎勵。 |
| PPO | `clip_range` | 0.1 | 減少更新震盪。 |
|  | `ent_coef` | 0.01（callback 可調） | 控制探索度。 |
|  | `vf_coef` | 1.5 | 增強 value loss 權重。 |
| 效率 | `n_steps` / `batch_size` | 1,536 / 2,048 | 大批次訓練。 |
|  | `n_epochs` | 12 | 每批多次更新。 |
| 其他 | `max_grad_norm` | 0.3 | 限制梯度爆炸。 |
| 日誌 | `tensorboard_log` | `./logs/tensorboard/` | 方便監控。 |

---

## 6. 訓練→評估流程

1. `python rl/train_sb3.py --target 6666 --n-envs 32 --total-timesteps 5000000 --seed 42`
2. TensorBoard 監控：`tensorboard --logdir logs/tensorboard/PPO_15`
3. Scalar 匯出：`python tools/export_scalars.py --run PPO_15`
4. 圖表：`python tools/plot_scalars.py`、`python tools/plot_network.py`
5. 最佳模型：`best_model/best_model.zip`
6. 評估錄影：
   ```powershell
   python rl/eval_sb3.py \
     --model best_model/best_model.zip \
     --norm-path models/vec_normalize_6666.pkl \
     --episodes 40 \
     --deterministic \
     --seed 150 \
     --video-dir outputs/videos/ppo_clear_seed150 \
     --video-length 90000 \
     --report logs/eval/summary_latest.json
   ffmpeg -y -i outputs/videos/ppo_clear_seed150/eval_seed150_det.mp4 -filter:v "setpts=0.25*PTS" -an outputs/videos/ppo_clear_seed150/eval_seed150_det_4x.mp4
   ```

---

## 7. 評估結果（僅穿越障礙）

- `logs/eval/summary_latest.json`：40 回合、平均 reward 5,272.52、win 21 次（52.5%）。
- `outputs/videos/ppo_clear_seed150/eval_seed150_det.mp4`：48m19s 破關影片；4× 版 12m05s。
- 主要檔案 SHA256：
  - TBoard 事件 `logs/tensorboard/PPO_15/events...` → `9FBE5A6B...99C2B`
  - 最佳模型 → `B40E7EDF...274A3`
  - 影片 → `B0E7F464...08D4C`
  - 4× 影片 → `EF536B91...88FF8`

---

## 8. 後續建議

1. **加強 win rate 觀測**：除了 `env/win_rate`，可在 callback 另寫 `rollout/win_rate` 方便對照。
2. **自動課程**：依 `passed_count` 動態調整 `scroll_speed`，讓穿越障礙更具挑戰性。
3. **報告自動化**：以 `outputs/PPO_training_process.md` 為模板，在每次訓練結束自動生成紀錄。

> 以上內容均針對「穿越障礙」專案。若需驗證，請直接比對 `rl/train_sb3.py`、`rl/eval_sb3.py` 與 `logs/`、`outputs/` 目錄中的資料。
