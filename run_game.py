"""Launcher for the Pygame UI.

Run with:
    python run_game.py

This will open a 1280x720 window. If torch is installed the PPOAgent will be used
for AI mode; otherwise a simple fallback agent will be used.
"""

import argparse

from agents.ppo_agent import PPOAgent
from game.ui import GameUI


def parse_args():
    parser = argparse.ArgumentParser(description="Train Game UI")
    parser.add_argument(
        "--replay-model",
        type=str,
        default=None,
        help="Path to an SB3 PPO zip to use for Replay mode (default: auto-detect)",
    )
    parser.add_argument(
        "--auto-replay",
        action="store_true",
        help=(
            "Start the UI directly in SB3 Replay mode (requires --replay-model "
            "or auto-detected model)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        agent = PPOAgent()
    except Exception:
        agent = None

    ui = GameUI(agent=agent, replay_model_path=args.replay_model)

    if args.auto_replay:
        ui.enable_auto_replay()

    ui.run()


if __name__ == "__main__":
    main()
