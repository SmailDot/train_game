# Fix Report

## 1. GPU Detection Issue
- **Problem**: The application reports "⚠️ 未檢測到 GPU，使用 CPU 訓練".
- **Cause**: The installed PyTorch version is `2.9.1+cpu`, which does not support CUDA.
- **Solution**: You need to reinstall PyTorch with CUDA support.
  1. Uninstall the current version: `pip uninstall torch torchvision torchaudio -y`
  2. Install the CUDA version (refer to `GPU_SETUP.md` for details):
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
     *(Note: Use `cu118` or `cu121` depending on your driver compatibility. `GPU_SETUP.md` suggests `cu128` which is for very new drivers)*.

## 2. SB3 Replay Crash
- **Problem**: `AssertionError: Expecting a torch Tensor, but got <class 'numpy.ndarray'>` when running SB3 Replay.
- **Cause**: The `predict_values` method in Stable-Baselines3 expects a PyTorch Tensor, but a NumPy array was being passed.
- **Fix**: Modified `agents/sb3_replay_agent.py` to convert the observation to a Tensor using `self.model.policy.obs_to_tensor(obs)` before calling `predict_values`.
- **Status**: Fixed. You should now be able to use the SB3 Replay feature.
