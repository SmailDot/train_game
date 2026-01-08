#!/usr/bin/env python3
"""
創建 Loss Function 的 3D 可視化
展示 Value Loss, Policy Loss, 和訓練步數之間的關係
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 讀取數據
csv_path = Path("outputs/metrics/loss_convergence_15M.csv")
if not csv_path.exists():
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

print(f"Loading data from {csv_path}...")
df_raw = pd.read_csv(csv_path)

# Pivot table
df = df_raw.pivot(index="step", columns="tag", values="value")
df = df.reset_index()

# 簡化列名
column_mapping = {
    "step": "Step",
    "train/loss(總損失)": "Total Loss",
    "train/value_loss(價值損失)": "Value Loss",
    "train/entropy_loss(熵損失)": "Entropy Loss",
    "train/policy_gradient_loss(策略梯度損失)": "Policy Loss",
    "rollout/ep_rew_mean(平均回合獎勵)": "Reward",
}
df = df.rename(columns=column_mapping)
df = df.ffill().bfill()

# 為了讓3D圖更清晰，採樣數據點
sample_rate = 5  # 每5個點取一個
df_sampled = df.iloc[::sample_rate].copy()

print(f"Sampled {len(df_sampled)} points for 3D visualization")

# 創建輸出目錄
output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# ===== 3D 圖表 1: Value Loss vs Policy Loss vs Steps =====
print("\n創建 3D Loss 曲面...")

fig = go.Figure()

# 添加 3D 線條
fig.add_trace(
    go.Scatter3d(
        x=df_sampled["Step"],
        y=df_sampled["Value Loss"],
        z=df_sampled["Policy Loss"],
        mode="lines+markers",
        marker=dict(
            size=3,
            color=df_sampled["Step"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Training Steps"),
        ),
        line=dict(color=df_sampled["Step"], colorscale="Viridis", width=2),
        text=[
            f"Step: {s:,.0f}<br>Value Loss: {v:.4f}<br>Policy Loss: {p:.6f}"
            for s, v, p in zip(
                df_sampled["Step"], df_sampled["Value Loss"], df_sampled["Policy Loss"]
            )
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Training Trajectory",
    )
)

# 標記起點和終點
fig.add_trace(
    go.Scatter3d(
        x=[df_sampled["Step"].iloc[0]],
        y=[df_sampled["Value Loss"].iloc[0]],
        z=[df_sampled["Policy Loss"].iloc[0]],
        mode="markers",
        marker=dict(size=10, color="green", symbol="circle"),
        name="Start",
        text=f"Start: Step {df_sampled['Step'].iloc[0]:,.0f}",
        hovertemplate="%{text}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter3d(
        x=[df_sampled["Step"].iloc[-1]],
        y=[df_sampled["Value Loss"].iloc[-1]],
        z=[df_sampled["Policy Loss"].iloc[-1]],
        mode="markers",
        marker=dict(size=10, color="red", symbol="diamond"),
        name="End",
        text=f"End: Step {df_sampled['Step'].iloc[-1]:,.0f}",
        hovertemplate="%{text}<extra></extra>",
    )
)

fig.update_layout(
    title="3D Loss Trajectory: Training Convergence Path",
    scene=dict(
        xaxis_title="Training Steps",
        yaxis_title="Value Loss",
        zaxis_title="Policy Loss",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    ),
    width=1200,
    height=800,
    showlegend=True,
)

output_file = output_dir / "loss_3d_trajectory_15M.html"
fig.write_html(str(output_file))
print(f"✅ 已保存: {output_file}")

# ===== 3D 圖表 2: Total Loss 隨時間的變化（帶顏色編碼的性能）=====
print("\n創建 3D Loss-Reward 關係圖...")

fig2 = go.Figure()

# 使用 Reward 作為顏色
fig2.add_trace(
    go.Scatter3d(
        x=df_sampled["Step"],
        y=df_sampled["Total Loss"],
        z=df_sampled["Reward"],
        mode="lines+markers",
        marker=dict(
            size=4,
            color=df_sampled["Reward"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Reward"),
            cmin=df_sampled["Reward"].min(),
            cmax=df_sampled["Reward"].max(),
        ),
        line=dict(color=df_sampled["Reward"], colorscale="RdYlGn", width=3),
        text=[
            f"Step: {s:,.0f}<br>Total Loss: {l:.4f}<br>Reward: {r:.2f}"
            for s, l, r in zip(
                df_sampled["Step"], df_sampled["Total Loss"], df_sampled["Reward"]
            )
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Training Progress",
    )
)

# 標記起點和終點
fig2.add_trace(
    go.Scatter3d(
        x=[df_sampled["Step"].iloc[0]],
        y=[df_sampled["Total Loss"].iloc[0]],
        z=[df_sampled["Reward"].iloc[0]],
        mode="markers",
        marker=dict(size=12, color="blue", symbol="circle"),
        name="Start",
        showlegend=True,
    )
)

fig2.add_trace(
    go.Scatter3d(
        x=[df_sampled["Step"].iloc[-1]],
        y=[df_sampled["Total Loss"].iloc[-1]],
        z=[df_sampled["Reward"].iloc[-1]],
        mode="markers",
        marker=dict(size=12, color="gold", symbol="diamond"),
        name="End",
        showlegend=True,
    )
)

fig2.update_layout(
    title="3D View: Total Loss vs Reward Over Training",
    scene=dict(
        xaxis_title="Training Steps",
        yaxis_title="Total Loss",
        zaxis_title="Average Reward",
        camera=dict(eye=dict(x=1.3, y=-1.3, z=1.0)),
    ),
    width=1200,
    height=800,
    showlegend=True,
)

output_file2 = output_dir / "loss_reward_3d_15M.html"
fig2.write_html(str(output_file2))
print(f"✅ 已保存: {output_file2}")

# ===== 3D 圖表 3: 多維損失空間 =====
print("\n創建多維損失空間圖...")

fig3 = go.Figure()

# 使用 Total Loss 作為顏色
fig3.add_trace(
    go.Scatter3d(
        x=df_sampled["Value Loss"],
        y=df_sampled["Policy Loss"],
        z=df_sampled["Entropy Loss"],
        mode="markers",
        marker=dict(
            size=5,
            color=df_sampled["Total Loss"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Total Loss"),
        ),
        text=[
            f"Step: {s:,.0f}<br>Total Loss: {tl:.4f}<br>Value Loss: {vl:.4f}<br>Policy Loss: {pl:.6f}<br>Entropy Loss: {el:.4f}"
            for s, tl, vl, pl, el in zip(
                df_sampled["Step"],
                df_sampled["Total Loss"],
                df_sampled["Value Loss"],
                df_sampled["Policy Loss"],
                df_sampled["Entropy Loss"],
            )
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Loss States",
    )
)

fig3.update_layout(
    title="3D Loss Space: Value, Policy, and Entropy Components",
    scene=dict(
        xaxis_title="Value Loss",
        yaxis_title="Policy Loss",
        zaxis_title="Entropy Loss",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    ),
    width=1200,
    height=800,
)

output_file3 = output_dir / "loss_space_3d_15M.html"
fig3.write_html(str(output_file3))
print(f"✅ 已保存: {output_file3}")

print("\n" + "=" * 60)
print("✅ 所有 3D 可視化圖表已生成！")
print("=" * 60)
print(f"📁 保存位置: {output_dir}")
print("\n生成的文件:")
print("  • loss_3d_trajectory_15M.html - 訓練軌跡 3D 視圖")
print("  • loss_reward_3d_15M.html - Loss-Reward 關係 3D 視圖")
print("  • loss_space_3d_15M.html - 多維損失空間 3D 視圖")
print("\n💡 提示: 在瀏覽器中打開 HTML 文件可進行交互式 3D 探索")
print("=" * 60)
