# GPU 加速設置指南

## 📋 系統配置

- **GPU**: NVIDIA GeForce RTX 3060 Ti (8GB VRAM)
- **驅動版本**: 572.83
- **CUDA 版本**: 12.8
- **Python 版本**: 3.12.2

## 🔧 安裝步驟

### 1. 卸載 CPU 版本的 PyTorch

```bash
pip uninstall torch torchvision torchaudio -y
```

### 2. 安裝 CUDA 版本的 PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**預計下載大小**: ~2.9 GB  
**預計時間**: 3-5 分鐘（取決於網速）

### 3. 驗證安裝

```bash
python test_gpu.py
```

應該看到：
```
✅ CUDA 可用: True
🔢 CUDA 版本: 12.8
🖥️  GPU 0: NVIDIA GeForce RTX 3060 Ti
   總記憶體: 8.00 GB
```

## 🚀 使用 GPU 訓練

### 自動配置（推薦）

遊戲 UI 會自動檢測 GPU 並使用優化的配置：

```python
# game/ui.py 中的 _register_algorithms() 會自動檢測
use_cuda = torch.cuda.is_available()
if use_cuda:
    print(f"✅ 檢測到 GPU: {torch.cuda.get_device_name(0)}")
```

### GPU 優化配置

在 `utils/training_config.py` 中：

```python
RTX_3060TI_CONFIG = {
    "device": "cuda",              # 使用 GPU
    "batch_size": 256,             # 增大 batch size
    "ppo_epochs": 10,              # 增加 PPO 更新次數
    "lr": 2.5e-4,                  # 學習率
    "horizon": 4096,               # 增加 rollout 長度
}
```

### 手動指定設備

```python
from utils.training_config import TrainingConfig

# GPU 訓練
config = TrainingConfig(use_gpu=True)
trainer = PPOTrainer(**config.get_ppo_kwargs())

# CPU 訓練（備用）
config = TrainingConfig(use_gpu=False)
```

## 📊 性能預期

### RTX 3060 Ti 性能指標

| 項目 | CPU | GPU | 提升 |
|------|-----|-----|------|
| Batch Size | 64 | 256 | 4x |
| PPO Epochs | 4 | 10 | 2.5x |
| Parallel Envs | 4 | 8 | 2x |
| 矩陣運算 | 基準 | ~15x | 15x |
| **總體訓練速度** | 基準 | **~10-15x** | **10-15x** |

### 記憶體使用

- **模型**: ~50 MB
- **Batch (256)**: ~200 MB
- **梯度 + 優化器**: ~100 MB
- **總計**: ~350 MB / 8192 MB (4% 使用率)

**結論**: RTX 3060 Ti 8GB 記憶體綽綽有餘！

## ⚙️ 訓練配置建議

### PPO 訓練（GPU 優化）

```python
{
    "device": "cuda",
    "batch_size": 256,      # 充分利用 GPU
    "ppo_epochs": 10,       # 更多更新次數
    "lr": 2.5e-4,           # 穩定學習率
    "gamma": 0.99,
    "lam": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,       # 降低 entropy
    "horizon": 4096,        # 大 rollout
}
```

### 並行環境配置

```python
# GPU 模式：8 個並行環境
n_envs = 8

# 每個環境收集 512 步
# 總計：8 * 512 = 4096 步/batch
```

## 🔍 故障排除

### 問題 1: CUDA 不可用

**症狀**: `torch.cuda.is_available()` 返回 `False`

**解決方案**:
1. 確認安裝的是 CUDA 版本: `pip show torch | findstr cu`
2. 檢查 NVIDIA 驅動: `nvidia-smi`
3. 重新安裝: 
   ```bash
   pip uninstall torch -y
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

### 問題 2: 記憶體不足 (OOM)

**症狀**: `RuntimeError: CUDA out of memory`

**解決方案**:
```python
# 降低 batch_size
"batch_size": 128  # 從 256 降到 128

# 或降低 horizon
"horizon": 2048  # 從 4096 降到 2048

# 或減少並行環境
n_envs = 4  # 從 8 降到 4
```

### 問題 3: GPU 利用率低

**症狀**: `nvidia-smi` 顯示 GPU 使用率 < 30%

**可能原因**:
1. Batch size 太小
2. 數據傳輸瓶頸
3. CPU 預處理慢

**解決方案**:
```python
# 增大 batch size
"batch_size": 512  # 如果記憶體允許

# 使用 pin_memory 加速數據傳輸
# （在 DataLoader 中設置）
```

## 📈 監控 GPU 使用

### 實時監控

```bash
# 每秒更新一次
nvidia-smi -l 1

# 或使用 watch (如果有安裝)
watch -n 1 nvidia-smi
```

### Python 代碼監控

```python
import torch

# 當前記憶體使用
allocated = torch.cuda.memory_allocated() / 1024**2  # MB
reserved = torch.cuda.memory_reserved() / 1024**2    # MB

print(f"已分配: {allocated:.1f} MB")
print(f"已保留: {reserved:.1f} MB")

# GPU 利用率（需要額外套件）
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU 利用率: {util.gpu}%")
```

## 🎯 最佳實踐

### 1. 預熱 GPU

```python
# 第一次運行時預熱
if torch.cuda.is_available():
    dummy = torch.randn(1, 1).cuda()
    _ = dummy + dummy
    torch.cuda.synchronize()
```

### 2. 混合精度訓練（進階）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 清理未使用的記憶體

```python
# 訓練後清理
torch.cuda.empty_cache()
```

### 4. 設置隨機種子

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## 📚 參考資源

- [PyTorch CUDA 官方文檔](https://pytorch.org/docs/stable/cuda.html)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [RTX 3060 Ti 規格](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/)

## ✅ 檢查清單

安裝完成後，確認以下項目：

- [ ] `torch.cuda.is_available()` 返回 `True`
- [ ] `nvidia-smi` 顯示 GPU 信息
- [ ] `test_gpu.py` 所有測試通過
- [ ] GPU 訓練速度提升 10x 以上
- [ ] 訓練視窗正確顯示 "PPO 訓練視窗 (CUDA)"

## 🎉 開始訓練

```bash
python run_game.py
```

點擊 "AI 訓練" → 選擇 PPO → 開始訓練

享受 GPU 加速的訓練速度！🚀
