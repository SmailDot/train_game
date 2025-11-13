from game.environment import GameEnv


def test_collision_with_obstacle():
    env = GameEnv(seed=1)
    env.reset()
    # force obstacle to be near player and set ball outside gap
    env.ob_x = 5.0
    # put ball well above gap
    env.y = env.gap_top - 20.0
    s, r, done, _ = env.step(0)
    assert done is True
    # collision penalty changed to -5 (plus small time penalty)
    # we expect a moderately negative reward here
    assert r <= -5.0
