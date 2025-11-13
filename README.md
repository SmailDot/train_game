# train_game

簡易 Flappy-like 遊戲 + RL (PPO) 實驗專案骨架。

快速開始（Windows / PowerShell）

1. 建立虛擬環境並啟用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 安裝需求（預設為最小測試套件）：

```powershell
pip install -r requirements.txt
```

3. 以 smoke 測試啟動 trainer：

```powershell
python -m agents.trainer
```

4. 執行單元測試：

```powershell
python -m pytest -q
```

GitHub 工作流程（建議）

```
git commit -m "新增：PPO agent skeleton 與測試（中文提交）"
```

若你要將本地專案上傳到 `https://github.com/SmailDot/train_game.git`：

```powershell
git init
git remote add origin https://github.com/SmailDot/train_game.git
git add .
git commit -m "初始化專案：新增程式骨架與文件（中文）"
git branch -M main
git push -u origin main
```

（確保你已經在 GitHub 建立 repository 並擁有 push 權限）
CI badge (示範)：

![CI](https://github.com/SmailDot/train_game/actions/workflows/ci.yml/badge.svg)

啟用 pre-commit hooks：

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

若想啟動 PyTorch 版訓練器：

```powershell
# 建議安裝 CPU 版 torch 或依照你的 CUDA 版本安裝
pip install -r requirements.txt
python -c "from agents.pytorch_trainer import PPOTrainer; t=PPOTrainer(); t.train(total_timesteps=2000)"
```
