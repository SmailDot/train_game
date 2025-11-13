"""Launcher for the Pygame UI.

Run with:
    python run_game.py

This will open a 1280x720 window. If torch is installed the PPOAgent will be used
for AI mode; otherwise a simple fallback agent will be used.
"""
from agents.ppo_agent import PPOAgent
from game.ui import GameUI


def main():
    try:
        agent = PPOAgent()
    except Exception:
        agent = None

    ui = GameUI(agent=agent)
    ui.run()


if __name__ == "__main__":
    main()
