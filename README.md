```markdown
# train_game

簡易 Flappy-like 遊戲 + RL (PPO) 實驗專案骨架。此 README 專為 Windows (PowerShell) 使用者整理，包含環境建立、測試、以及如何啟動 UI/訓練器的步驟。

快速開始（Windows / PowerShell）

1) 建立並啟用虛擬環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安裝依賴（本專案使用已釘選的版本，請參照 `requirements.txt`）

```powershell
pip install -r requirements.txt
```

注意：如果要使用 PyTorch CUDA 版本，請依照你的 CUDA 版本到 https://pytorch.org/ 取得正確安裝指令；若無 GPU，使用 CPU 版：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3) 執行測試（單元/整合/頭less UI smoke）

```powershell
python -m pytest -q
```

4) 以 smoke 模式啟動 trainer（範例會印出幾個 episode 的結果）

```powershell
python -m agents.trainer
```

5) 啟動 Pygame UI（若在有顯示器的環境）

```powershell
python run_game.py
```

如果你在無頭 (headless) 環境，UI 可能無法啟動，請於本機桌面環境執行。

Git / 提交範例（中文提交訊息）：

```powershell
git add .
git commit -m "新增：PPO agent 與 UI 骨架與測試"
git push
```

啟用 pre-commit hooks（建議在本地環境執行一次以完成 hooks 安裝）：

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

CI badge（若已啟用 GitHub Actions）：

![CI](https://github.com/SmailDot/train_game/actions/workflows/ci.yml/badge.svg)

進階：以 PyTorch 訓練器做短跑驗證（CPU 範例）

```powershell
# 建議先安裝 requirements.txt 內的套件
python -c "from agents.pytorch_trainer import PPOTrainer; t=PPOTrainer(); t.train(total_timesteps=2000)"
```

疑難排解小提示
- 若出現 `pygame` 無法啟動，請確認正在使用有 GUI 的環境並且 `SDL_VIDEODRIVER` 未被設定為 headless。
- pre-commit 若在安裝 hooks 時發生 rev/checkout 問題，嘗試在能連上外部 Git 的網路環境重新執行 `pre-commit install`。

歡迎在本地執行上述命令，有任何錯誤訊息把輸出貼給我，我可以幫你一步步排查。

---
小節：本專案包含 `game/` (環境與 UI)、`agents/` (網路、PPO trainer)、`tests/` (pytest)、以及 CI + pre-commit configs。
```
