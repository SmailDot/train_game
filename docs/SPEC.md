# 規格文件（State / Action / Reward）

此文件記錄遊戲與 RL 代理的正式介面與參數（中文）。

## State (S)
- S1：球的 Y 座標（像素），範圍 [0, ScreenHeight]，送入網路前正規化為 y/ScreenHeight。
- S2：球的垂直速度 Vy，範圍 [-MaxAbsVel, MaxAbsVel]，正規化為 Vy/MaxAbsVel。
- S3：最近障礙物水平距離 X_obs，範圍 [0, MaxDist]，正規化為 X_obs/MaxDist。
- S4：最近障礙物入口上沿 Y_gap_top，範圍 [0, ScreenHeight]，正規化為 Y_gap_top/ScreenHeight。
- S5：最近障礙物入口下沿 Y_gap_bottom，範圍 [0, ScreenHeight]，正規化為 Y_gap_bottom/ScreenHeight。

輸入向量為 5 維浮點數：S = [S1_norm, S2_norm, S3_norm, S4_norm, S5_norm]

## Action (A)
- A1：跳躍（離散二元）：0 = 不跳，1 = 跳。

訓練時從 policy 中採樣（stochastic）；部署或測試時可採 argmax（確定性）。

## Reward (R)
- 每一步時間懲罰：-0.1
- 通過障礙物獎勵：+10（球的中心成功通過障礙物入口時給予一次）
- 碰撞懲罰：-100（碰撞發生，episode 終止）

註：可選擇對 reward 做 scaling 或 running normalization 以穩定 PPO。

## Episode 與 n 值
- 每個 episode 從球 spawn 開始，直到碰撞或達到 step 上限（例如 1000）為止。
- n 從 1 開始，每次 episode 結束（失敗或到達上限） n += 1。
