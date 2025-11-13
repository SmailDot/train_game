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
- 請在每次提交時使用中文提交訊息，例如：

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
