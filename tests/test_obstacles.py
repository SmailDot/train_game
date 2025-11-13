from game.environment import GameEnv


def test_obstacle_respawn():
    env = GameEnv(seed=2)
    env.reset()
    # push obstacle past respawn threshold
    env.ob_x = -60.0
    prev_gap_top = env.gap_top
    s, r, done, _ = env.step(0)
    # after step, obstacle should respawn with ob_x reset to positive value
    assert env.ob_x > 0
    # gap should have changed or at least be within screen bounds
    assert 0 <= env.gap_top < env.ScreenHeight
