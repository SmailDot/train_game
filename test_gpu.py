"""
GPU 和 CUDA 配置測試腳本
"""

import sys

import torch


def test_cuda_setup():
    """測試 CUDA 設置"""
    print("=" * 60)
    print("🔍 PyTorch 和 CUDA 配置檢查")
    print("=" * 60)

    # PyTorch 版本
    print(f"\n📦 PyTorch 版本: {torch.__version__}")

    # CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA 可用: {cuda_available}")

    if not cuda_available:
        print("\n❌ CUDA 不可用！")
        print("可能原因:")
        print("1. 安裝了 CPU 版本的 PyTorch")
        print("2. NVIDIA 驅動未正確安裝")
        print("3. CUDA toolkit 版本不匹配")
        print("\n請執行以下命令重新安裝 CUDA 版本:")
        print("pip uninstall torch torchvision torchaudio -y")
        print(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        )
        return False

    # CUDA 詳細信息
    print(f"🔢 CUDA 版本: {torch.version.cuda}")
    print(f"🎮 cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"📊 可用 GPU 數量: {torch.cuda.device_count()}")

    # GPU 詳細信息
    for i in range(torch.cuda.device_count()):
        print(f"\n🖥️  GPU {i}: {torch.cuda.get_device_name(i)}")
        total_mem = torch.cuda.get_device_properties(i).total_memory
        print(f"   總記憶體: {total_mem / 1024**3:.2f} GB")
        props = torch.cuda.get_device_properties(i)
        print(f"   計算能力: {props.major}.{props.minor}")

    # 測試張量運算
    print("\n" + "=" * 60)
    print("🧪 GPU 運算測試")
    print("=" * 60)

    try:
        # 創建張量
        x = torch.randn(1000, 1000)
        print(f"✓ CPU 張量創建: {x.shape}")

        # 移動到 GPU
        x_gpu = x.cuda()
        print(f"✓ GPU 張量創建: {x_gpu.shape}, device: {x_gpu.device}")

        # GPU 矩陣乘法
        result = torch.mm(x_gpu, x_gpu)
        print(f"✓ GPU 矩陣乘法: {result.shape}")

        # 檢查記憶體使用
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        print("\n💾 GPU 記憶體使用:")
        print(f"   已分配: {memory_allocated:.2f} MB")
        print(f"   已保留: {memory_reserved:.2f} MB")

        print("\n✅ GPU 運算測試通過！")
        return True

    except Exception as e:
        print(f"\n❌ GPU 運算測試失敗: {e}")
        return False


def test_training_speed():
    """比較 CPU vs GPU 訓練速度"""
    if not torch.cuda.is_available():
        print("\n⚠️  GPU 不可用，跳過速度測試")
        return

    print("\n" + "=" * 60)
    print("⚡ CPU vs GPU 速度比較")
    print("=" * 60)

    import time

    # 測試數據
    size = 5000
    iterations = 10

    # CPU 測試
    print(f"\n🖥️  CPU 測試 ({iterations} 次 {size}x{size} 矩陣乘法)...")
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)

    start = time.time()
    for _ in range(iterations):
        _ = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"   耗時: {cpu_time:.3f} 秒")

    # GPU 測試
    print(f"\n🎮 GPU 測試 ({iterations} 次 {size}x{size} 矩陣乘法)...")
    x_gpu = x_cpu.cuda()
    y_gpu = y_cpu.cuda()

    # Warm up
    _ = torch.mm(x_gpu, y_gpu)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iterations):
        _ = torch.mm(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"   耗時: {gpu_time:.3f} 秒")

    # 速度提升
    speedup = cpu_time / gpu_time
    print(f"\n🚀 GPU 加速比: {speedup:.1f}x")

    if speedup > 10:
        print("✅ GPU 性能優秀！")
    elif speedup > 5:
        print("✅ GPU 性能良好")
    elif speedup > 2:
        print("⚠️  GPU 性能一般")
    else:
        print("❌ GPU 性能不佳，可能存在配置問題")


def test_neural_network():
    """測試神經網路在 GPU 上運行"""
    if not torch.cuda.is_available():
        print("\n⚠️  GPU 不可用，跳過神經網路測試")
        return

    print("\n" + "=" * 60)
    print("🧠 神經網路 GPU 測試")
    print("=" * 60)

    import torch.nn as nn

    # 創建簡單網路
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(5, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    # 移動到 GPU
    net = SimpleNet().cuda()
    print("✓ 網路已創建並移動到 GPU")

    # 測試前向傳播
    x = torch.randn(32, 5).cuda()
    output = net(x)
    print(f"✓ 前向傳播: input {x.shape} -> output {output.shape}")

    # 測試反向傳播
    loss = output.sum()
    loss.backward()
    print("✓ 反向傳播完成")

    print("\n✅ 神經網路 GPU 測試通過！")


if __name__ == "__main__":
    print("\n" + "🎯" * 30)
    print("CUDA PyTorch 配置檢查工具")
    print("🎯" * 30 + "\n")

    # 測試 CUDA 設置
    cuda_ok = test_cuda_setup()

    if cuda_ok:
        # 速度比較
        test_training_speed()

        # 神經網路測試
        test_neural_network()

        print("\n" + "=" * 60)
        print("🎉 所有測試通過！您的 GPU 已正確配置")
        print("=" * 60)
        print("\n💡 建議:")
        print("1. 在訓練時設置 device='cuda' 來使用 GPU")
        print("2. 使用較大的 batch_size (如 256) 以充分利用 GPU")
        print("3. 增加並行環境數量 (如 8) 提升訓練效率")
        print("\n📚 查看 utils/training_config.py 了解 GPU 優化配置")
    else:
        print("\n" + "=" * 60)
        print("❌ CUDA 配置有問題，請檢查安裝")
        print("=" * 60)
        sys.exit(1)
