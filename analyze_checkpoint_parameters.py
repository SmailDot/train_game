"""
分析睡覺期間（#5940 到 #14460）的檢查點參數變化
找出導致崩潰的參數調整模式
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

print("=" * 80)
print("🔬 深度分析檢查點參數變化（#5940 → #14460）")
print("=" * 80)

# === 配置 ===
START_ITER = 5940  # 睡覺後開始
END_ITER = 14460  # 醒來時
SAMPLE_INTERVAL = 100  # 每 100 次迭代採樣一次
CHECKPOINT_DIR = "checkpoints"

# === 收集要分析的檢查點 ===
print("\n第一步：收集檢查點檔案")
print("-" * 80)

checkpoints_to_analyze = []
for iter_num in range(START_ITER, END_ITER + 1, SAMPLE_INTERVAL):
    # 找最接近的檢查點（每 10 次保存一次）
    checkpoint_iter = (iter_num // 10) * 10
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_iter}.pt")
    if os.path.exists(checkpoint_file):
        checkpoints_to_analyze.append((checkpoint_iter, checkpoint_file))

print(f"✅ 找到 {len(checkpoints_to_analyze)} 個檢查點用於分析")
print(f"   範圍: #{checkpoints_to_analyze[0][0]} → #{checkpoints_to_analyze[-1][0]}")

# === 分析參數變化 ===
print("\n第二步：分析模型參數統計")
print("-" * 80)

param_stats = defaultdict(
    list
)  # {param_name: [(iter, mean, std, min, max, norm), ...]}
optimizer_stats = defaultdict(list)  # {key: [(iter, value), ...]}

for iter_num, checkpoint_file in checkpoints_to_analyze:
    try:
        # 載入檢查點（僅 CPU，不需 GPU）
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

        # 分析模型參數 (使用 model_state 而非 model_state_dict)
        if "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = None

        if model_state is not None:

            for param_name, param_tensor in model_state.items():
                if param_tensor.dtype in [torch.float32, torch.float16]:
                    param_np = param_tensor.cpu().numpy().flatten()

                    stats = {
                        "iter": iter_num,
                        "mean": float(np.mean(param_np)),
                        "std": float(np.std(param_np)),
                        "min": float(np.min(param_np)),
                        "max": float(np.max(param_np)),
                        "norm": float(np.linalg.norm(param_np)),
                        "abs_mean": float(np.mean(np.abs(param_np))),
                        "zeros_pct": float(np.sum(param_np == 0) / len(param_np) * 100),
                    }
                    param_stats[param_name].append(stats)

        # 分析優化器狀態 (使用 optimizer_state 而非 optimizer_state_dict)
        if "optimizer_state" in checkpoint:
            opt_state = checkpoint["optimizer_state"]
        elif "optimizer_state_dict" in checkpoint:
            opt_state = checkpoint["optimizer_state_dict"]
        else:
            opt_state = None

        if opt_state is not None:
            if "param_groups" in opt_state and len(opt_state["param_groups"]) > 0:
                pg = opt_state["param_groups"][0]
                optimizer_stats["learning_rate"].append((iter_num, pg.get("lr", 0)))
                optimizer_stats["eps"].append((iter_num, pg.get("eps", 0)))
                optimizer_stats["weight_decay"].append(
                    (iter_num, pg.get("weight_decay", 0))
                )

        # 分析其他元數據
        if "iteration" in checkpoint:
            optimizer_stats["checkpoint_iteration"].append(
                (iter_num, checkpoint["iteration"])
            )

        print(f"   ✓ 分析 #{iter_num}", end="\r")

    except Exception as e:
        print(f"   ✗ 無法載入 #{iter_num}: {e}")

print(f"\n✅ 完成 {len(checkpoints_to_analyze)} 個檢查點分析")

# === 檢測異常變化 ===
print("\n第三步：檢測異常參數變化")
print("-" * 80)

anomalies = []

for param_name, stats_list in param_stats.items():
    if len(stats_list) < 5:
        continue

    # 提取時間序列
    iters = [s["iter"] for s in stats_list]
    norms = [s["norm"] for s in stats_list]
    means = [s["mean"] for s in stats_list]
    stds = [s["std"] for s in stats_list]
    zeros_pcts = [s["zeros_pct"] for s in stats_list]

    # 檢測 1: 參數範數爆炸（norm 突然增加 > 50%）
    for i in range(1, len(norms)):
        if norms[i - 1] > 0:
            change_pct = (norms[i] - norms[i - 1]) / norms[i - 1] * 100
            if abs(change_pct) > 50:
                anomalies.append(
                    {
                        "type": "norm_explosion" if change_pct > 0 else "norm_collapse",
                        "param": param_name,
                        "iter": iters[i],
                        "change_pct": change_pct,
                        "norm_before": norms[i - 1],
                        "norm_after": norms[i],
                    }
                )

    # 檢測 2: 參數變成全零（dead neurons）
    for i in range(len(zeros_pcts)):
        if zeros_pcts[i] > 90:  # 超過 90% 是零
            anomalies.append(
                {
                    "type": "dead_parameters",
                    "param": param_name,
                    "iter": iters[i],
                    "zeros_pct": zeros_pcts[i],
                }
            )

    # 檢測 3: 標準差崩潰（參數不再更新）
    for i in range(1, len(stds)):
        if stds[i] < stds[0] * 0.1:  # 標準差降到初始的 10% 以下
            anomalies.append(
                {
                    "type": "std_collapse",
                    "param": param_name,
                    "iter": iters[i],
                    "std_before": stds[0],
                    "std_after": stds[i],
                }
            )

# 按迭代次數排序
anomalies.sort(key=lambda x: x["iter"])

# 輸出異常
print(f"\n🚨 檢測到 {len(anomalies)} 個異常")

if anomalies:
    print("\n前 20 個異常:")
    for i, anomaly in enumerate(anomalies[:20]):
        print(f"\n[{i+1}] 類型: {anomaly['type']}")
        print(f"    參數: {anomaly['param']}")
        print(f"    迭代: #{anomaly['iter']}")
        for key, value in anomaly.items():
            if key not in ["type", "param", "iter"]:
                if isinstance(value, float):
                    print(f"    {key}: {value:.6f}")
                else:
                    print(f"    {key}: {value}")

# === 分析學習率變化 ===
print("\n第四步：分析優化器參數變化")
print("-" * 80)

if "learning_rate" in optimizer_stats:
    lr_data = optimizer_stats["learning_rate"]
    print(f"\n📊 學習率變化:")
    print(f"   初始 LR (#{lr_data[0][0]}): {lr_data[0][1]:.8f}")
    print(f"   最終 LR (#{lr_data[-1][0]}): {lr_data[-1][1]:.8f}")

    # 檢查學習率是否有異常跳變
    for i in range(1, len(lr_data)):
        if lr_data[i][1] != lr_data[i - 1][1]:
            change_pct = (lr_data[i][1] - lr_data[i - 1][1]) / lr_data[i - 1][1] * 100
            if abs(change_pct) > 10:
                print(
                    f"   ⚠️ 學習率變化 #{lr_data[i][0]}: {lr_data[i-1][1]:.8f} → {lr_data[i][1]:.8f} ({change_pct:+.1f}%)"
                )

# === 找出最可疑的層 ===
print("\n第五步：識別最可疑的層")
print("-" * 80)

# 統計每個參數的異常次數
param_anomaly_count = defaultdict(int)
for anomaly in anomalies:
    param_anomaly_count[anomaly["param"]] += 1

# 排序
sorted_params = sorted(param_anomaly_count.items(), key=lambda x: x[1], reverse=True)

print(f"\n🎯 異常次數最多的參數 (Top 10):")
for i, (param_name, count) in enumerate(sorted_params[:10]):
    print(f"   {i+1}. {param_name}: {count} 次異常")

# === 生成時間線分析 ===
print("\n第六步：生成參數變化時間線")
print("-" * 80)

# 按迭代分組異常
iter_anomaly_count = defaultdict(int)
for anomaly in anomalies:
    # 以 100 為單位分組
    iter_group = (anomaly["iter"] // 100) * 100
    iter_anomaly_count[iter_group] += 1

print(f"\n📈 異常密度分布:")
for iter_group in sorted(iter_anomaly_count.keys())[:20]:
    count = iter_anomaly_count[iter_group]
    bar = "█" * min(count, 50)
    print(f"   #{iter_group:5d}-{iter_group+99:5d}: {bar} ({count})")

# === 保存詳細報告 ===
print("\n第七步：保存分析報告")
print("-" * 80)

report = {
    "analysis_time": datetime.now().isoformat(),
    "iteration_range": [START_ITER, END_ITER],
    "checkpoints_analyzed": len(checkpoints_to_analyze),
    "total_anomalies": len(anomalies),
    "anomaly_breakdown": {
        "norm_explosion": len([a for a in anomalies if a["type"] == "norm_explosion"]),
        "norm_collapse": len([a for a in anomalies if a["type"] == "norm_collapse"]),
        "dead_parameters": len(
            [a for a in anomalies if a["type"] == "dead_parameters"]
        ),
        "std_collapse": len([a for a in anomalies if a["type"] == "std_collapse"]),
    },
    "top_problematic_params": [
        {"param": param, "anomaly_count": count} for param, count in sorted_params[:20]
    ],
    "anomaly_timeline": [
        {"iter_range": f"{iter_group}-{iter_group+99}", "count": count}
        for iter_group, count in sorted(iter_anomaly_count.items())
    ],
    "detailed_anomalies": anomalies[:100],  # 保存前 100 個
}

with open("checkpoints/parameter_analysis_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("✅ 報告已保存到: checkpoints/parameter_analysis_report.json")

# === 總結和建議 ===
print("\n" + "=" * 80)
print("📋 分析總結")
print("=" * 80)

print(f"\n統計:")
print(f"   分析的檢查點: {len(checkpoints_to_analyze)} 個")
print(f"   分析的參數: {len(param_stats)} 個")
print(f"   檢測到的異常: {len(anomalies)} 個")

print(f"\n異常類型分布:")
for anomaly_type, count in report["anomaly_breakdown"].items():
    print(f"   {anomaly_type}: {count} 次")

if sorted_params:
    print(f"\n最可疑的層:")
    for i, (param_name, count) in enumerate(sorted_params[:3]):
        print(f"   {i+1}. {param_name} ({count} 次異常)")

print(f"\n🎯 建議:")
if report["anomaly_breakdown"]["norm_explosion"] > 10:
    print(f"   ⚠️ 檢測到 {report['anomaly_breakdown']['norm_explosion']} 次參數範數爆炸")
    print(f"      建議: 降低學習率或增加梯度裁剪")

if report["anomaly_breakdown"]["dead_parameters"] > 10:
    print(f"   ⚠️ 檢測到 {report['anomaly_breakdown']['dead_parameters']} 次死亡參數")
    print(f"      建議: 調整初始化方法或使用 Leaky ReLU")

if report["anomaly_breakdown"]["std_collapse"] > 10:
    print(f"   ⚠️ 檢測到 {report['anomaly_breakdown']['std_collapse']} 次標準差崩潰")
    print(f"      建議: 增加探索噪音或調整學習率衰減")

print(f"\n" + "=" * 80)
