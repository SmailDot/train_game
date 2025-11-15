#!/usr/bin/env python3
"""Simple evaluation utility for SB3 PPO checkpoints."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Ensure repository modules can be imported when script is run directly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))


def _import_training_utils():
    from rl.train_sb3 import create_envs as _create_envs

    return _create_envs


create_envs = _import_training_utils()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SB3 PPO model")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the .zip checkpoint"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes to run",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (greedy actions)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of vectorized environments to evaluate in parallel",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable VecNormalize (set to --no-normalize to disable)",
    )
    parser.add_argument(
        "--norm-path",
        type=str,
        help="Path to VecNormalize statistics (.pkl) to load before evaluation",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        help=(
            "Directory to store recorded evaluation videos "
            "(records the first episode)"
        ),
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=2_000,
        help="Maximum number of frames captured in the video",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Optional path to write aggregate metrics as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the evaluation environments",
    )
    return parser.parse_args()


class StreamingVideoRecorder:
    """Incrementally write frames to disk using MoviePy's FFmpeg writer."""

    def __init__(self, video_path: Path, fps: int = 30, max_frames: int = 0):
        self.video_path = video_path
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.max_frames = max(0, int(max_frames))
        self._writer: Optional[FFMPEG_VideoWriter] = None
        self._frames_written = 0

    def write(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        if self.max_frames and self._frames_written >= self.max_frames:
            return

        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = FFMPEG_VideoWriter(
                str(self.video_path), (width, height), self.fps
            )

        self._writer.write_frame(frame)
        self._frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def summarize_metrics(
    rewards: List[float], lengths: List[int], passes: List[int], wins: List[bool]
) -> Dict[str, float]:
    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def safe_std(values: List[float]) -> float:
        return float(np.std(values)) if values else 0.0

    win_total = int(np.sum(wins)) if wins else 0
    episodes = len(rewards)
    win_rate = win_total / episodes if episodes else 0.0

    return {
        "episodes": episodes,
        "reward_mean": safe_mean(rewards),
        "reward_std": safe_std(rewards),
        "length_mean": safe_mean(lengths),
        "length_std": safe_std(lengths),
        "passes_mean": safe_mean(passes),
        "passes_std": safe_std(passes),
        "wins_total": win_total,
        "win_rate": win_rate,
    }


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    if not args.normalize and args.norm_path:
        print(
            "⚠️ --norm-path specified but normalization disabled; ignoring stats file."
        )

    render_mode = "rgb_array" if args.video_dir else None
    env = create_envs(
        n_envs=args.n_envs,
        normalize=args.normalize,
        training=False,
        norm_path=args.norm_path,
        seed=args.seed,
        render_mode=render_mode,
    )

    if args.normalize and isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False

    video_recorder: Optional[StreamingVideoRecorder] = None
    if args.video_dir:
        if args.n_envs != 1:
            raise ValueError("Video capture requires --n-envs 1")
        video_dir = Path(args.video_dir)
        suffix = "det" if args.deterministic else "stoch"
        video_name = f"eval_seed{args.seed}_{suffix}.mp4"
        video_path = video_dir / video_name
        print(
            "🎥 Recording evaluation video to "
            f"{video_path} (max {args.video_length} frames)"
        )
        video_recorder = StreamingVideoRecorder(
            video_path, fps=30, max_frames=args.video_length
        )

    print(f"📁 Loading model from {args.model}")
    model = PPO.load(args.model, env=env, device="auto")

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    passed_counts: List[int] = []
    wins: List[bool] = []

    obs = env.reset()
    state = None

    def _capture_frame():
        if not video_recorder:
            return
        frame = env.render(mode="rgb_array")
        video_recorder.write(frame)

    _capture_frame()

    completed = 0
    try:
        while completed < args.episodes:
            action, state = model.predict(
                obs, state=state, deterministic=args.deterministic
            )
            obs, rewards, dones, infos = env.step(action)

            _capture_frame()

            for env_idx, done in enumerate(dones):
                if not done:
                    continue

                info = infos[env_idx] if isinstance(infos, list) else infos
                episode = info.get("episode", {})
                episode_rewards.append(float(episode.get("r", rewards[env_idx])))
                episode_lengths.append(int(episode.get("l", 0)))
                passed_counts.append(int(info.get("passed_count", 0)))
                wins.append(bool(info.get("win", False)))
                completed += 1

                if completed >= args.episodes:
                    break

            if np.all(dones):
                obs = env.reset()
                state = None
                _capture_frame()
    finally:
        if video_recorder:
            video_recorder.close()

    metrics = summarize_metrics(episode_rewards, episode_lengths, passed_counts, wins)

    print("\n📊 Evaluation summary")
    for key, value in metrics.items():
        print(
            f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}"
        )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"📝 Metrics written to {report_path}")

    env.close()


if __name__ == "__main__":
    main()
