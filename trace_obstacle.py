"""詳細追蹤第一個障礙物"""

from game.environment import GameEnv

env = GameEnv(seed=42)
state = env.reset()

print(f"初始狀態:")
print(f"  球: y={env.y:.1f}, 範圍=[{env.y-12:.1f}, {env.y+12:.1f}]")
print(
    f"  障礙物: x={env.obstacles[0][0]:.1f}, 間隙=[{env.obstacles[0][1]:.1f}, {env.obstacles[0][2]:.1f}]"
)
print(
    f"  球在間隙內: {env.obstacles[0][1] <= env.y-12 and env.y+12 <= env.obstacles[0][2]}"
)
print()

# 策略：每5步跳1次
for step in range(60):
    action = 1 if step % 5 == 0 else 0

    # 記錄障礙物位置
    obs_x_before = env.obstacles[0][0]

    state, reward, done, info = env.step(action)

    obs_x_after = env.obstacles[0][0]
    ball_top = env.y - 12
    ball_bottom = env.y + 12
    gap_top = env.obstacles[0][1]
    gap_bottom = env.obstacles[0][2]

    # 只顯示關鍵時刻
    if abs(obs_x_after) < 20 or reward != 0 or done or step % 10 == 0:
        in_gap = gap_top <= ball_top and ball_bottom <= gap_bottom
        print(
            f"步驟 {step:2d}: 障礙物 x={obs_x_after:6.1f}, 球 y={env.y:5.1f} [{ball_top:5.1f}, {ball_bottom:5.1f}], 在間隙內={in_gap}, 獎勵={reward:+2.0f}"
        )

    if done:
        print(f"\n結束原因: ", end="")
        if ball_top < 0:
            print(f"撞天花板 (球頂={ball_top:.1f})")
        elif ball_bottom > 600:
            print(f"撞地面 (球底={ball_bottom:.1f})")
        elif -12 < obs_x_after < 12:
            print(f"撞障礙物 (x={obs_x_after:.1f})")
        print(f"通過障礙物數: {env.passed_count}")
        break
