"""
檢查檢查點檔案的實際結構
"""

import os

import torch

checkpoint_file = "checkpoints/checkpoint_5940.pt"

print("=" * 80)
print(f"🔍 檢查檢查點結構: {checkpoint_file}")
print("=" * 80)

try:
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    print(f"\n檢查點的 keys:")
    for key in checkpoint.keys():
        print(f"   - {key}: {type(checkpoint[key])}")

    # 如果有 model_state_dict，顯示參數名稱
    if "model_state_dict" in checkpoint:
        print(f"\nmodel_state_dict 的參數:")
        for i, (param_name, param_tensor) in enumerate(
            checkpoint["model_state_dict"].items()
        ):
            if i < 20:  # 只顯示前 20 個
                print(f"   {param_name}: {param_tensor.shape} ({param_tensor.dtype})")
        print(f"   ... 共 {len(checkpoint['model_state_dict'])} 個參數")

    # 如果有 optimizer_state_dict，顯示結構
    if "optimizer_state_dict" in checkpoint:
        print(f"\noptimizer_state_dict 的結構:")
        opt_state = checkpoint["optimizer_state_dict"]
        for key in opt_state.keys():
            if key != "state":
                print(f"   {key}: {opt_state[key]}")

    # 如果有 iteration
    if "iteration" in checkpoint:
        print(f"\niteration: {checkpoint['iteration']}")

    # 如果有其他元數據
    for key in ["episode", "best_score", "avg_score"]:
        if key in checkpoint:
            print(f"{key}: {checkpoint[key]}")

except Exception as e:
    print(f"❌ 無法載入: {e}")
    import traceback

    traceback.print_exc()
