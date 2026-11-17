# AGENTS.md

給 AI 協作工具（Claude Code、Codex 等）閱讀的專案規則。開始修改前請先讀完本文件。

## 1. 專案目的

本專案是「Flappy-like 遊戲 + PPO 強化學習」的訓練平台：用 Pygame 做遊戲環境、Stable-Baselines3 (PyTorch) 訓練 AI 過關。
同時它也是課程第九週作業的**作品展示網站**來源：`docs/` 目錄會以 GitHub Pages 發佈成靜態頁面。

目前的里程碑（以 README 與 CHANGELOG 為準）：v5 Final，100 次確定性評估通關 99 場，平均獎勵 7647。

## 2. 專案結構

| 路徑 | 用途 | 修改規則 |
| :--- | :--- | :--- |
| `run_game.py` | 啟動遊戲入口 | 可改，但保持可直接執行 |
| `game/` | 遊戲本體（環境 `environment.py`、UI、向量化環境 `vec_env.py`） | 改動需同步檢查 README 的觀察空間與獎勵表 |
| `rl/train_sb3.py` | SB3 PPO 訓練主程式（命令列參數用 `--help` 查） | 新參數要加 `help=` 說明 |
| `rl/eval_sb3.py` | 評估已訓練模型 | — |
| `rl/game2048_env.py` | SB3 環境檔（檔名沿用舊專案名） | 不要重新命名，避免破壞既有的載入流程 |
| `agents/`、`utils/`、`tools/` | agent 實作、訓練設定與日誌、繪圖工具 | 非必要不動 |
| `models/`、`best_model/`、`checkpoints/`、`logs/` | 訓練產物 | **禁止手動刪除或覆寫**；不要把新的大型權重加進 git |
| `outputs/` | 圖表、指標 CSV、評估影片 | 只新增、不覆寫既有檔案 |
| `docs/` | 作品展示網頁（GitHub Pages 來源）與文件 | 網頁規則見第 4 節 |
| `docs/SPEC.md` | 舊版規格（5 維狀態、舊獎勵） | **已過時**，以 README 與程式碼為準，修改時請更新或註明 |

## 3. 關鍵規格（以程式碼與 README 為準）

- **觀察空間**：7 維正規化向量 `[y, vy, x_obs, gap_top, gap_bottom, rel_top, rel_bottom]`，每一維都在固定範圍內。
- **動作空間**：`Discrete(2)`，`0` 不跳、`1` 跳。
- **獎勵**：通關 +1000、通過障礙 +5、存活每步 +0.01、中心對齊最多 +0.05、死亡 -5。
- **網路**：MLP，`[256, 256]`，ReLU。
- **訓練**：32 個並行環境、FrameSkip = 4、VecNormalize、PPO clip / value / entropy 三項損失。

若要改動上述任何一項，必須同時更新 README 的對應段落與 CHANGELOG（新增一段日期標題）。

## 4. 作品展示網頁規則（`docs/`）

- 首頁是 `docs/index.html`，**單一檔案**，所有 CSS 寫在 `<style>` 內，不引入框架或 CDN，方便離線檢視。
- 語言：繁體中文（台灣用語），標題用 `<h2>` 起跳，不要用多個 `<h1>`。
- 色彩沿用首頁既有的 CSS 變數（`--bg`、`--card`、`--accent`），不要硬寫新色碼散落各處。
- 圖片放在 `docs/assets/`，使用相對路徑（例如 `assets/loss_overview_15M.png`）。**不要**用 `../outputs/...`，GitHub Pages 只會發佈 `docs/` 內的檔案。
- 大型影片（`outputs/videos/` 底下的 mp4，最大 38 MB）**不要**複製進 `docs/`，改用 GitHub 連結。
- 新增頁面的流程：
  1. 在 `docs/pages/<slug>.html` 建立新檔，`<head>` 與導覽列直接複製 `docs/index.html` 的結構。
  2. 在 `docs/index.html` 的導覽列加上連結。
  3. 內容數字必須來自 `outputs/metrics/` 或 README/CHANGELOG，不可憑空寫入。

## 5. 數據與事實規則

- 任何成果數字（通關率、獎勵、步數）都要能在 repo 內找到出處；找不到就寫「未知」或不寫。
- 不要編造模型表現、影片內容或論文引用。
- 修改 `models/` 內的檔案前，先確認使用者要求；預設不動。

## 6. 程式碼規則

- Python 3.8+ 語法相容（README 宣告 3.8+；CI 使用 3.11）。
- 提交前執行 `ruff check .`（CI 會跑，失敗就不能合併）。
- 測試用 `python -m pytest -q`；目前 repo 沒有固定測試檔，新增功能時請同時補上對應測試。
- 不要提交 `__pycache__/`、`*.pyc`、`checkpoints/*.zip`（`.gitignore` 已排除）。

## 7. 提交規則

- 不直接在 `main` 上工作，使用分支（例如 `week9-agents-md`）。
- Commit 訊息用祈使句，一行摘要，必要時附說明。
- 不要 force push `main`。

## 8. 新增內容時 AI 應該怎麼做

以「新增一個實驗結果頁」為例：

1. 讀本文件與 `README.md`，確認數字來源（`outputs/metrics/*.csv` 或 CHANGELOG）。
2. 依第 4 節建立 `docs/pages/<slug>.html`，並更新 `docs/index.html` 的導覽。
3. 若有新圖，放進 `docs/assets/`，不要改動 `outputs/` 內的原檔。
4. 完成後說明改了哪些檔案、數字來自哪裡，並列出尚未驗證的部分。

以「把新專案加到展示網站」為例：在 `docs/index.html` 的專案區塊新增一張卡片（標題、一句話簡介、連結），不要改動既有卡片。
