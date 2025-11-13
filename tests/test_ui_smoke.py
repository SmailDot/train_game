from game.environment import GameEnv


def test_ui_smoke_steps():
    """Headless smoke test: step the env a few times to simulate UI logic.

    This ensures the core loop (step/reset/done handling) runs without errors.
    """
    env = GameEnv(seed=42)
    s = env.reset()
    for _ in range(20):
        s, r, done, _ = env.step(0)
        if done:
            s = env.reset()
    assert True
