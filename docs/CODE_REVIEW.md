# Code Review 檢查清單

此檔為 Code Review Sub-Agent 使用的核對表，非強制性修正。請以建議方式提交給 Game Design Agent。

1. 可讀性與風格
- 檔案是否有清楚的 docstring / module-level 說明？
- 函式與變數命名是否具語意性（避免 v1, v2 等不明命名）？
- 是否遵守 PEP8（行寬、縮排）？

2. 型別與靜態檢查
- 是否使用 type hints（重要 public API）？
- mypy / ruff 的基本錯誤是否為零？

3. 測試覆蓋
- 是否有對應的 unit tests（physics、collision、obstacle）？
- 是否包含整合測試或 smoke test？

4. 效能與資源
- 遊戲主迴圈是否避免在每幀做大量 I/O 或重算（如每幀重新建立大型物件）？
- collision 檢查是否使用簡單 AABB 或其他輕量演算法？
- 訓練過程是否以 batch/向量化收集經驗以提高效能？

5. RL 特定檢查
- PPO 參數是否有合理預設（clip, gamma, lambda, entropy_coef）？
- 是否有 checkpoint 機制與恢復流程？
- 是否對 reward 做 normalization 或 scaling？註明原因。

6. 安全性與穩定性
- 是否有 try/except 包覆 I/O、模型儲存等可能失敗的程式段？
- long-running process（training）是否有 graceful shutdown 與 periodic checkpoint？

回饋格式建議：
- issue 或 PR comment 使用短句 + 建議修正，必要時附上 code snippet。
