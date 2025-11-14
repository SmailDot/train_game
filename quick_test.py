"""快速測試環境"""

from game.environment import GameEnv

env = GameEnv(seed=42)
state = env.reset()

print(
    f"初始: y={env.y:.1f}, 障礙物 x={env.obstacles[0][0]:.1f}, 間隙=[{env.obstacles[0][1]:.1f}, {env.obstacles[0][2]:.1f}]"
)

total_reward = 0
for step in range(100):
    action = 1 if step % 4 < 2 else 0  # 交替跳躍
    state, reward, done, info = env.step(action)
    total_reward += reward

    if reward != 0:
        print(f"步驟 {step}: 獎勵={reward:.0f}, y={env.y:.1f}")

    if done:
        print(
            f"\n回合結束: 總獎勵={total_reward:.0f}, 步數={step+1}, 通過={env.passed_count}"
        )
        break
