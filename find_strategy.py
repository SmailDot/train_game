"""找到能通過障礙物的策略"""

from game.environment import GameEnv

# 測試不同的跳躍策略
strategies = [
    ("不跳", lambda s: 0),
    ("一直跳", lambda s: 1),
    ("每2步跳1次", lambda s: s % 2 == 0),
    ("每3步跳1次", lambda s: s % 3 == 0),
    ("每4步跳1次", lambda s: s % 4 == 0),
    ("每5步跳1次", lambda s: s % 5 == 0),
    ("前2後2", lambda s: (s % 4) < 2),
]

for name, strategy in strategies:
    env = GameEnv(seed=42)
    state = env.reset()

    total_reward = 0
    for step in range(100):
        action = 1 if strategy(step) else 0
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(
                f"{name:15s}: 獎勵={total_reward:+3.0f}, 步數={step+1:2d}, 通過={env.passed_count}"
            )
            break
    else:
        print(
            f"{name:15s}: 獎勵={total_reward:+3.0f}, 步數=100+, 通過={env.passed_count}"
        )
