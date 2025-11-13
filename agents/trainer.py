"""Trainer harness: runs episodes using GameEnv and an agent instance.
This is intentionally minimal so it can run as a smoke test without heavy deps.
"""

from agents.ppo_agent import PPOAgent
from game.environment import GameEnv


class Trainer:
    def __init__(self, episodes=5):
        self.env = GameEnv()
        self.agent = PPOAgent()
        self.episodes = episodes

    def run(self):
        results = []
        n = 1
        for ep in range(self.episodes):
            s = self.env.reset()
            total_r = 0.0
            done = False
            steps = 0
            while not done and steps < 200:
                a, logp, val = self.agent.act(s)
                s, r, done, _ = self.env.step(a)
                total_r += r
                steps += 1
            results.append((total_r, steps))
            print(f"Episode {n}: reward={total_r:.2f} steps={steps}")
            n += 1
        return results


if __name__ == "__main__":
    t = Trainer(episodes=3)
    t.run()
