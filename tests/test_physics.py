from game.environment import GameEnv


def test_gravity_integration():
    env = GameEnv(seed=0)
    env.reset()
    y0 = env.y
    vy0 = env.vy
    # step a few frames without jumping
    for _ in range(3):
        s, r, done, _ = env.step(0)
        if done:
            break
    assert env.y != y0 or env.vy != vy0
    # gravity should increase vy (positive) from initial 0 in this env
    assert env.vy > vy0
