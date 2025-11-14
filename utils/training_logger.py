"""
訓練日誌記錄器
為每個演算法創建結構化的輸出檔案，方便 AI Coding 工具分析和優化
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingLogger:
    """記錄訓練過程的詳細數據"""

    def __init__(self, algorithm_name: str, output_dir: str = "outputs"):
        self.algorithm_name = algorithm_name
        self.output_dir = Path(output_dir) / algorithm_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 創建子目錄
        self.metrics_dir = self.output_dir / "metrics"
        self.config_dir = self.output_dir / "config"
        self.analysis_dir = self.output_dir / "analysis"

        for d in [self.metrics_dir, self.config_dir, self.analysis_dir]:
            d.mkdir(exist_ok=True)

        # 訓練會話 ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()

        # 數據緩存
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.episode_buffer: List[Dict[str, Any]] = []

        # 文件路徑
        self.metrics_file = self.metrics_dir / f"session_{self.session_id}.jsonl"
        self.episode_file = self.metrics_dir / f"episodes_{self.session_id}.jsonl"
        self.summary_file = self.output_dir / "training_summary.json"
        self.hyperparams_file = self.config_dir / f"hyperparams_{self.session_id}.json"

    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """記錄超參數配置"""
        config = {
            "algorithm": self.algorithm_name,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": hyperparams,
        }

        with open(self.hyperparams_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✅ 已記錄超參數: {self.hyperparams_file}")

    def log_metrics(self, iteration: int, metrics: Dict[str, float]) -> None:
        """記錄訓練指標"""
        entry = {
            "iteration": iteration,
            "timestamp": time.time() - self.session_start,
            "datetime": datetime.now().isoformat(),
            **metrics,
        }

        self.metrics_buffer.append(entry)

        # 每 10 條記錄寫入一次
        if len(self.metrics_buffer) >= 10:
            self._flush_metrics()

    def log_episode(
        self,
        episode: int,
        score: float,
        steps: int,
        iteration: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """記錄單次遊戲回合數據"""
        entry = {
            "episode": episode,
            "score": score,
            "steps": steps,
            "iteration": iteration,
            "timestamp": time.time() - self.session_start,
            "datetime": datetime.now().isoformat(),
        }

        if extra:
            entry.update(extra)

        self.episode_buffer.append(entry)

        # 每 5 條記錄寫入一次
        if len(self.episode_buffer) >= 5:
            self._flush_episodes()

    def _flush_metrics(self) -> None:
        """將指標寫入檔案"""
        if not self.metrics_buffer:
            return

        with open(self.metrics_file, "a", encoding="utf-8") as f:
            for entry in self.metrics_buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self.metrics_buffer.clear()

    def _flush_episodes(self) -> None:
        """將回合數據寫入檔案"""
        if not self.episode_buffer:
            return

        with open(self.episode_file, "a", encoding="utf-8") as f:
            for entry in self.episode_buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self.episode_buffer.clear()

    def finalize(self, final_iteration: int, total_episodes: int) -> None:
        """結束訓練會話，寫入總結"""
        # 刷新所有緩存
        self._flush_metrics()
        self._flush_episodes()

        # 計算訓練時長
        duration = time.time() - self.session_start

        # 讀取所有episode數據計算統計
        episodes = []
        if self.episode_file.exists():
            with open(self.episode_file, "r", encoding="utf-8") as f:
                for line in f:
                    episodes.append(json.loads(line))

        scores = [ep["score"] for ep in episodes] if episodes else [0]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0

        summary = {
            "algorithm": self.algorithm_name,
            "session_id": self.session_id,
            "final_iteration": final_iteration,
            "total_episodes": total_episodes,
            "duration_seconds": duration,
            "duration_formatted": (
                f"{duration/3600:.2f}h" if duration > 3600 else f"{duration/60:.2f}min"
            ),
            "average_score": avg_score,
            "max_score": max_score,
            "end_time": datetime.now().isoformat(),
            "files": {
                "metrics": str(self.metrics_file),
                "episodes": str(self.episode_file),
                "hyperparameters": str(self.hyperparams_file),
            },
        }

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✅ 訓練總結已儲存: {self.summary_file}")

    def generate_analysis_report(self) -> str:
        """生成分析報告供 AI 讀取"""
        report_file = self.analysis_dir / f"analysis_{self.session_id}.md"

        # 讀取數據
        metrics = []
        if self.metrics_file.exists():
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    metrics.append(json.loads(line))

        episodes = []
        if self.episode_file.exists():
            with open(self.episode_file, "r", encoding="utf-8") as f:
                for line in f:
                    episodes.append(json.loads(line))

        # 生成 Markdown 報告
        report = f"""# {self.algorithm_name} 訓練分析報告

## 會話資訊
- **Session ID**: {self.session_id}
- **開始時間**: {datetime.fromtimestamp(self.session_start).isoformat()}
- **總迭代次數**: {metrics[-1]['iteration'] if metrics else 0}
- **總回合數**: {len(episodes)}

## 訓練指標統計

### 分數表現
- 平均分數: {sum(ep['score'] for ep in episodes) / len(episodes) if episodes else 0:.2f}
- 最高分數: {max((ep['score'] for ep in episodes), default=0)}
- 最低分數: {min((ep['score'] for ep in episodes), default=0)}

### 訓練趨勢
"""

        if metrics:
            recent = metrics[-100:]  # 最近 100 條
            avg_loss = sum(m.get("policy_loss", 0) for m in recent) / len(recent)
            avg_reward = sum(m.get("mean_reward", 0) for m in recent) / len(recent)
            report += f"""
- 最近 100 次迭代平均 loss: {avg_loss:.4f}
- 最近 100 次迭代平均 reward: {avg_reward:.2f}
"""

        report += """
## 建議改進方向

1. **學習率調整**: 根據 loss 曲線判斷是否需要調整學習率
2. **探索策略**: 觀察 epsilon/entropy 值，決定是否需要調整探索參數
3. **網絡架構**: 檢查是否需要增加或減少隱藏層數量
4. **訓練穩定性**: 觀察 loss 波動，考慮調整 batch size 或 update frequency

## 數據檔案位置

- 指標數據: `{self.metrics_file}`
- 回合數據: `{self.episode_file}`
- 超參數: `{self.hyperparams_file}`
- 訓練總結: `{self.summary_file}`

---
*此報告由 TrainingLogger 自動生成*
"""

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        return str(report_file)


def create_logger(algorithm_name: str, output_dir: str = "outputs") -> TrainingLogger:
    """創建訓練日誌記錄器的便捷函數"""
    return TrainingLogger(algorithm_name, output_dir)
