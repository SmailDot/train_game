#!/usr/bin/env python3
"""
可視化 15M 步訓練的 Loss Function 收斂過程
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 使用支援中文的字體
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# 讀取數據
csv_path = Path("outputs/metrics/loss_convergence_15M.csv")
if not csv_path.exists():
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

print(f"Loading data from {csv_path}...")
df_raw = pd.read_csv(csv_path)

# Pivot table to get tags as columns
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
    "env/win_rate(通關率)": "Win Rate",
    "train/approx_kl(近似KL散度)": "KL Divergence",
    "train/explained_variance(價值預測準確度)": "Explained Variance",
    "train/learning_rate(學習率)": "Learning Rate",
}
df = df.rename(columns=column_mapping)

# Forward fill and backward fill to handle missing values
df = df.ffill().bfill()

print(f"Loaded {len(df)} data points")
print(f"Step range: {df['Step'].min()} - {df['Step'].max()}")
print(f"Available columns: {df.columns.tolist()}")

# 創建輸出目錄
output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# ===== 圖表 1: 總損失收斂圖 =====
print("\n繪製總損失收斂圖...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["Step"], df["Total Loss"], linewidth=1.5, color="#e74c3c", alpha=0.8)
ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
ax.set_ylabel("Total Loss", fontsize=12, fontweight="bold")
ax.set_title("Total Loss Convergence (15M Steps)", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(df["Step"].min(), df["Step"].max())

# 添加統計信息
final_loss = df["Total Loss"].iloc[-1]
min_loss = df["Total Loss"].min()
mean_loss = df["Total Loss"].mean()
textstr = f"Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f}\nMean Loss: {mean_loss:.4f}"
ax.text(
    0.02,
    0.98,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig(output_dir / "loss_total_convergence_15M.png", dpi=300, bbox_inches="tight")
print(f"✅ 已保存: {output_dir / 'loss_total_convergence_15M.png'}")
plt.close()

# ===== 圖表 2: 三種損失分量對比 =====
print("\n繪製損失分量對比圖...")
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(
    df["Step"],
    df["Value Loss"],
    linewidth=1.5,
    label="Value Loss",
    color="#3498db",
    alpha=0.8,
)
ax.plot(
    df["Step"],
    df["Policy Loss"],
    linewidth=1.5,
    label="Policy Loss",
    color="#2ecc71",
    alpha=0.8,
)
ax.plot(
    df["Step"],
    df["Entropy Loss"],
    linewidth=1.5,
    label="Entropy Loss",
    color="#f39c12",
    alpha=0.8,
)

ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
ax.set_ylabel("Loss Value", fontsize=12, fontweight="bold")
ax.set_title("Loss Components Convergence (15M Steps)", fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(df["Step"].min(), df["Step"].max())

plt.tight_layout()
plt.savefig(
    output_dir / "loss_components_convergence_15M.png", dpi=300, bbox_inches="tight"
)
print(f"✅ 已保存: {output_dir / 'loss_components_convergence_15M.png'}")
plt.close()

# ===== 圖表 3: 4子圖綜合視圖 =====
print("\n繪製綜合視圖...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 子圖1: Total Loss
axes[0, 0].plot(df["Step"], df["Total Loss"], linewidth=1.5, color="#e74c3c")
axes[0, 0].set_xlabel("Training Steps", fontsize=10)
axes[0, 0].set_ylabel("Total Loss", fontsize=10)
axes[0, 0].set_title("Total Loss", fontsize=12, fontweight="bold")
axes[0, 0].grid(True, alpha=0.3, linestyle="--")

# 子圖2: Value Loss vs Policy Loss
axes[0, 1].plot(
    df["Step"], df["Value Loss"], linewidth=1.5, label="Value Loss", color="#3498db"
)
axes[0, 1].plot(
    df["Step"], df["Policy Loss"], linewidth=1.5, label="Policy Loss", color="#2ecc71"
)
axes[0, 1].set_xlabel("Training Steps", fontsize=10)
axes[0, 1].set_ylabel("Loss Value", fontsize=10)
axes[0, 1].set_title("Value & Policy Loss", fontsize=12, fontweight="bold")
axes[0, 1].legend(loc="best", fontsize=9)
axes[0, 1].grid(True, alpha=0.3, linestyle="--")

# 子圖3: Reward & Win Rate
ax3 = axes[1, 0]
ax3_twin = ax3.twinx()
line1 = ax3.plot(
    df["Step"], df["Reward"], linewidth=1.5, label="Reward", color="#9b59b6"
)
line2 = ax3_twin.plot(
    df["Step"], df["Win Rate"], linewidth=1.5, label="Win Rate", color="#1abc9c"
)
ax3.set_xlabel("Training Steps", fontsize=10)
ax3.set_ylabel("Reward", fontsize=10, color="#9b59b6")
ax3_twin.set_ylabel("Win Rate", fontsize=10, color="#1abc9c")
ax3.set_title("Performance Metrics", fontsize=12, fontweight="bold")
ax3.tick_params(axis="y", labelcolor="#9b59b6")
ax3_twin.tick_params(axis="y", labelcolor="#1abc9c")
ax3.grid(True, alpha=0.3, linestyle="--")
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="best", fontsize=9)

# 子圖4: Explained Variance & KL Divergence
ax4 = axes[1, 1]
ax4_twin = ax4.twinx()
line3 = ax4.plot(
    df["Step"],
    df["Explained Variance"],
    linewidth=1.5,
    label="Explained Variance",
    color="#e67e22",
)
line4 = ax4_twin.plot(
    df["Step"],
    df["KL Divergence"],
    linewidth=1.5,
    label="KL Divergence",
    color="#34495e",
)
ax4.set_xlabel("Training Steps", fontsize=10)
ax4.set_ylabel("Explained Variance", fontsize=10, color="#e67e22")
ax4_twin.set_ylabel("KL Divergence", fontsize=10, color="#34495e")
ax4.set_title("Training Quality Metrics", fontsize=12, fontweight="bold")
ax4.tick_params(axis="y", labelcolor="#e67e22")
ax4_twin.tick_params(axis="y", labelcolor="#34495e")
ax4.grid(True, alpha=0.3, linestyle="--")
lines = line3 + line4
labels = [l.get_label() for l in lines]
ax4.legend(lines, labels, loc="best", fontsize=9)

plt.suptitle("15M Steps Training Overview", fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout()
plt.savefig(output_dir / "loss_overview_15M.png", dpi=300, bbox_inches="tight")
print(f"✅ 已保存: {output_dir / 'loss_overview_15M.png'}")
plt.close()

# ===== 圖表 4: 學習曲線（前期 vs 後期對比）=====
print("\n繪製學習曲線對比...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 前期（前20%）
early_cutoff = int(len(df) * 0.2)
df_early = df.iloc[:early_cutoff]
ax1.plot(
    df_early["Step"],
    df_early["Total Loss"],
    linewidth=1.5,
    color="#e74c3c",
    label="Total Loss",
)
ax1.plot(
    df_early["Step"],
    df_early["Value Loss"],
    linewidth=1.5,
    color="#3498db",
    alpha=0.7,
    label="Value Loss",
)
ax1.plot(
    df_early["Step"],
    df_early["Policy Loss"],
    linewidth=1.5,
    color="#2ecc71",
    alpha=0.7,
    label="Policy Loss",
)
ax1.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
ax1.set_ylabel("Loss Value", fontsize=12, fontweight="bold")
ax1.set_title("Early Training Phase (First 20%)", fontsize=13, fontweight="bold")
ax1.legend(loc="best", fontsize=10)
ax1.grid(True, alpha=0.3, linestyle="--")

# 後期（後20%）
late_cutoff = int(len(df) * 0.8)
df_late = df.iloc[late_cutoff:]
ax2.plot(
    df_late["Step"],
    df_late["Total Loss"],
    linewidth=1.5,
    color="#e74c3c",
    label="Total Loss",
)
ax2.plot(
    df_late["Step"],
    df_late["Value Loss"],
    linewidth=1.5,
    color="#3498db",
    alpha=0.7,
    label="Value Loss",
)
ax2.plot(
    df_late["Step"],
    df_late["Policy Loss"],
    linewidth=1.5,
    color="#2ecc71",
    alpha=0.7,
    label="Policy Loss",
)
ax2.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
ax2.set_ylabel("Loss Value", fontsize=12, fontweight="bold")
ax2.set_title("Late Training Phase (Last 20%)", fontsize=13, fontweight="bold")
ax2.legend(loc="best", fontsize=10)
ax2.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(output_dir / "loss_phases_comparison_15M.png", dpi=300, bbox_inches="tight")
print(f"✅ 已保存: {output_dir / 'loss_phases_comparison_15M.png'}")
plt.close()

# ===== 輸出統計摘要 =====
print("\n" + "=" * 60)
print("📊 訓練統計摘要 (15M Steps)")
print("=" * 60)
print(f"\n🎯 最終狀態:")
print(f"  • Total Loss: {df['Total Loss'].iloc[-1]:.4f}")
print(f"  • Value Loss: {df['Value Loss'].iloc[-1]:.4f}")
print(f"  • Policy Loss: {df['Policy Loss'].iloc[-1]:.6f}")
print(f"  • Entropy Loss: {df['Entropy Loss'].iloc[-1]:.4f}")
print(f"  • Reward: {df['Reward'].iloc[-1]:.2f}")
print(f"  • Win Rate: {df['Win Rate'].iloc[-1]:.2%}")
print(f"  • Explained Variance: {df['Explained Variance'].iloc[-1]:.4f}")

print(f"\n📈 Loss 統計:")
print(
    f"  • Total Loss - Min: {df['Total Loss'].min():.4f}, Mean: {df['Total Loss'].mean():.4f}"
)
print(
    f"  • Value Loss - Min: {df['Value Loss'].min():.4f}, Mean: {df['Value Loss'].mean():.4f}"
)
print(
    f"  • Policy Loss - Min: {df['Policy Loss'].min():.6f}, Mean: {df['Policy Loss'].mean():.6f}"
)

print(f"\n🏆 性能峰值:")
print(f"  • Max Reward: {df['Reward'].max():.2f}")
print(f"  • Max Win Rate: {df['Win Rate'].max():.2%}")
print(f"  • Best Explained Variance: {df['Explained Variance'].max():.4f}")

# 計算收斂程度（最後10%的標準差）
late_10pct = int(len(df) * 0.9)
df_late_stable = df.iloc[late_10pct:]
print(f"\n🎲 收斂穩定性 (最後10%):")
print(f"  • Total Loss 標準差: {df_late_stable['Total Loss'].std():.6f}")
print(f"  • Value Loss 標準差: {df_late_stable['Value Loss'].std():.6f}")
print(f"  • Policy Loss 標準差: {df_late_stable['Policy Loss'].std():.8f}")

print("\n" + "=" * 60)
print(f"✅ 所有圖表已保存到: {output_dir}")
print("=" * 60)
