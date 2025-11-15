"""
更深入分析：繪製所有參數的變化曲線
找出崩潰前後的關鍵變化點
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

print("=" * 80)
print("📊 繪製參數變化趨勢圖（#5940 → #14460）")
print("=" * 80)

# === 配置 ===
START_ITER = 5940
END_ITER = 14460
SAMPLE_INTERVAL = 50  # 更密集採樣
CHECKPOINT_DIR = "checkpoints"
CRASH_ITER = 7436  # 已知的崩潰點

# === 收集檢查點 ===
print("\n收集檢查點...")
checkpoints_to_analyze = []
for iter_num in range(START_ITER, END_ITER + 1, SAMPLE_INTERVAL):
    checkpoint_iter = (iter_num // 10) * 10
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_iter}.pt")
    if os.path.exists(checkpoint_file):
        checkpoints_to_analyze.append((checkpoint_iter, checkpoint_file))

print(f"✅ 找到 {len(checkpoints_to_analyze)} 個檢查點")

# === 收集參數統計 ===
print("\n分析參數...")
param_history = defaultdict(
    lambda: {
        "iters": [],
        "mean": [],
        "std": [],
        "norm": [],
        "abs_mean": [],
        "min": [],
        "max": [],
    }
)

for iter_num, checkpoint_file in checkpoints_to_analyze:
    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        model_state = checkpoint.get(
            "model_state", checkpoint.get("model_state_dict", {})
        )

        for param_name, param_tensor in model_state.items():
            if param_tensor.dtype in [torch.float32, torch.float16]:
                param_np = param_tensor.cpu().numpy().flatten()

                param_history[param_name]["iters"].append(iter_num)
                param_history[param_name]["mean"].append(float(np.mean(param_np)))
                param_history[param_name]["std"].append(float(np.std(param_np)))
                param_history[param_name]["norm"].append(
                    float(np.linalg.norm(param_np))
                )
                param_history[param_name]["abs_mean"].append(
                    float(np.mean(np.abs(param_np)))
                )
                param_history[param_name]["min"].append(float(np.min(param_np)))
                param_history[param_name]["max"].append(float(np.max(param_np)))

        print(f"   ✓ #{iter_num}", end="\r")
    except Exception as e:
        print(f"   ✗ #{iter_num}: {e}")

print(f"\n✅ 完成，共 {len(param_history)} 個參數")

# === 繪製關鍵參數的趨勢圖 ===
print("\n生成趨勢圖...")

# 創建圖表
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
fig.suptitle("模型參數變化趨勢 (#5940 → #14460)", fontsize=16)

plot_idx = 0
param_names = list(param_history.keys())

for param_name in param_names[:8]:  # 繪製前 8 個參數
    row = plot_idx // 2
    col = plot_idx % 2
    ax = axes[row, col]

    data = param_history[param_name]
    iters = data["iters"]

    # 繪製 norm 和 abs_mean
    ax.plot(iters, data["norm"], "b-", label="L2 Norm", linewidth=1)
    ax2 = ax.twinx()
    ax2.plot(iters, data["abs_mean"], "r-", label="Abs Mean", linewidth=1, alpha=0.7)

    # 標記崩潰點
    ax.axvline(x=CRASH_ITER, color="red", linestyle="--", linewidth=2, label="崩潰點")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 Norm", color="b")
    ax2.set_ylabel("Abs Mean", color="r")
    ax.set_title(param_name, fontsize=10)
    ax.tick_params(axis="y", labelcolor="b")
    ax2.tick_params(axis="y", labelcolor="r")
    ax.grid(True, alpha=0.3)

    plot_idx += 1

plt.tight_layout()
plt.savefig("checkpoints/param_trends.png", dpi=150)
print("✅ 圖表已保存到: checkpoints/param_trends.png")

# === 計算崩潰前後的參數變化 ===
print("\n分析崩潰前後的參數變化...")

crash_analysis = []

for param_name, data in param_history.items():
    iters = data["iters"]
    norms = data["norm"]

    # 找崩潰前後的最近檢查點
    before_idx = None
    after_idx = None

    for i, iter_num in enumerate(iters):
        if iter_num <= CRASH_ITER:
            before_idx = i
        if iter_num > CRASH_ITER and after_idx is None:
            after_idx = i
            break

    if before_idx is not None and after_idx is not None:
        norm_before = norms[before_idx]
        norm_after = norms[after_idx]

        if norm_before > 0:
            change_pct = (norm_after - norm_before) / norm_before * 100

            crash_analysis.append(
                {
                    "param": param_name,
                    "iter_before": iters[before_idx],
                    "iter_after": iters[after_idx],
                    "norm_before": norm_before,
                    "norm_after": norm_after,
                    "change_pct": change_pct,
                    "abs_change": abs(change_pct),
                }
            )

# 按變化幅度排序
crash_analysis.sort(key=lambda x: x["abs_change"], reverse=True)

print(f"\n🔍 崩潰前後參數變化 (Top 10):")
print(f"   {'參數':<20} {'崩潰前':>12} {'崩潰後':>12} {'變化%':>10}")
print("-" * 60)
for i, item in enumerate(crash_analysis[:10]):
    print(
        f"{i+1:2d}. {item['param']:<20} {item['norm_before']:>12.6f} {item['norm_after']:>12.6f} {item['change_pct']:>+9.1f}%"
    )

# === 分析整體趨勢 ===
print("\n分析整體參數趨勢...")

overall_trends = {}

for param_name, data in param_history.items():
    norms = data["norm"]

    if len(norms) > 5:
        # 計算線性趨勢（斜率）
        x = np.arange(len(norms))
        slope = np.polyfit(x, norms, 1)[0]

        # 計算變異係數（CV）
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        cv = (std_norm / mean_norm * 100) if mean_norm > 0 else 0

        # 計算總變化
        total_change_pct = (
            ((norms[-1] - norms[0]) / norms[0] * 100) if norms[0] > 0 else 0
        )

        overall_trends[param_name] = {
            "slope": slope,
            "cv": cv,
            "total_change_pct": total_change_pct,
            "mean_norm": mean_norm,
        }

# 找出趨勢最不穩定的參數
unstable_params = sorted(overall_trends.items(), key=lambda x: x[1]["cv"], reverse=True)

print(f"\n📉 最不穩定的參數 (變異係數最高):")
print(f"   {'參數':<20} {'CV%':>10} {'總變化%':>12} {'平均 Norm':>12}")
print("-" * 60)
for i, (param_name, trends) in enumerate(unstable_params[:10]):
    print(
        f"{i+1:2d}. {param_name:<20} {trends['cv']:>9.1f}% {trends['total_change_pct']:>+11.1f}% {trends['mean_norm']:>12.6f}"
    )

# === 保存詳細分析 ===
print("\n保存詳細分析...")

detailed_report = {
    "analysis_time": str(np.datetime64("now")),
    "iteration_range": [START_ITER, END_ITER],
    "crash_iteration": CRASH_ITER,
    "checkpoints_analyzed": len(checkpoints_to_analyze),
    "parameters_tracked": len(param_history),
    "crash_impact": crash_analysis[:20],
    "unstable_parameters": [
        {"param": name, **trends} for name, trends in unstable_params[:20]
    ],
    "all_parameter_stats": {
        name: {
            "final_norm": data["norm"][-1] if data["norm"] else 0,
            "mean_norm": np.mean(data["norm"]) if data["norm"] else 0,
            "std_norm": np.std(data["norm"]) if data["norm"] else 0,
        }
        for name, data in param_history.items()
    },
}

with open("checkpoints/detailed_parameter_analysis.json", "w", encoding="utf-8") as f:
    json.dump(detailed_report, f, ensure_ascii=False, indent=2)

print("✅ 詳細報告已保存到: checkpoints/detailed_parameter_analysis.json")

# === 總結建議 ===
print("\n" + "=" * 80)
print("📋 分析結論")
print("=" * 80)

print(f"\n統計:")
print(f"   檢查點數量: {len(checkpoints_to_analyze)}")
print(f"   追蹤的參數: {len(param_history)}")
print(f"   崩潰迭代: #{CRASH_ITER}")

if crash_analysis:
    max_change = crash_analysis[0]
    print(f"\n崩潰前後最大變化:")
    print(f"   參數: {max_change['param']}")
    print(f"   變化幅度: {max_change['change_pct']:+.1f}%")

if unstable_params:
    most_unstable = unstable_params[0]
    print(f"\n最不穩定的參數:")
    print(f"   參數: {most_unstable[0]}")
    print(f"   變異係數: {most_unstable[1]['cv']:.1f}%")

print(f"\n🎯 訓練改進建議:")

# 根據分析結果給出建議
high_cv_count = sum(1 for _, trends in unstable_params if trends["cv"] > 50)
if high_cv_count > 0:
    print(f"   ⚠️ {high_cv_count} 個參數的變異係數 > 50%")
    print(f"      → 建議: 降低學習率 (目前 0.00025 → 建議 0.0001)")
    print(f"      → 建議: 增加梯度裁剪強度")

big_changes = [item for item in crash_analysis if abs(item["change_pct"]) > 20]
if big_changes:
    print(f"   ⚠️ {len(big_changes)} 個參數在崩潰前後變化 > 20%")
    print(f"      → 建議: 增加訓練穩定性 (使用 batch normalization)")
    print(f"      → 建議: 降低 PPO clip range")

print(f"\n📈 趨勢圖:")
print(f"   請查看: checkpoints/param_trends.png")

print("\n" + "=" * 80)
