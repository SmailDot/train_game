"""詳細調試環境通過檢測邏輯"""

from game.environment import GameEnv

print("=" * 60)
print("詳細調試障礙物通過檢測")
print("=" * 60)

env = GameEnv(seed=42)
state = env.reset()

print(f"\n初始狀態:")
print(f"  球心 y: {env.y:.2f}")
print(f"  球速 vy: {env.vy:.2f}")
print(f"  屏幕高度: {env.ScreenHeight}")
print(f"  球半徑: 12")
print(f"  球頂部: {env.y - 12:.2f}")
print(f"  球底部: {env.y + 12:.2f}")

print(f"\n初始障礙物:")
for i, obs in enumerate(env.obstacles):
    print(
        f"  障礙物 {i+1}: x={obs[0]:.2f}, 間隙=[{obs[1]:.2f}, {obs[2]:.2f}], 高度={obs[2]-obs[1]:.2f}"
    )

# 簡單策略：每3步跳一次
print("\n開始模擬...")
for step in range(200):
    action = step % 3 == 0

    # 顯示當前球的狀態
    if step % 5 == 0 or done:
        ball_top = env.y - 12
        ball_bottom = env.y + 12
        print(
            f"步驟 {step}: y={env.y:.2f}, vy={env.vy:.2f}, 頂={ball_top:.2f}, 底={ball_bottom:.2f}, 動作={'跳' if action else '無'}"
        )

        # 檢查是否接近邊界
        if ball_top < 20:
            print(f"  ⚠️ 接近天花板！(距離 {ball_top:.2f})")
        if ball_bottom > 580:
            print(f"  ⚠️ 接近地面！(距離 {600-ball_bottom:.2f})")

    state, reward, done, info = env.step(action)

    # 檢查是否有障礙物即將通過
    for i, obs in enumerate(env.obstacles):
        ob_x = obs[0]
        if -20 < ob_x < 20:  # 障礙物在玩家附近
            gap_top, gap_bottom = obs[1], obs[2]
            ball_top = env.y - 12
            ball_bottom = env.y + 12

            # 碰撞檢測範圍
            collision_window = 12
            in_collision_zone = -collision_window < ob_x < collision_window
            # 通過檢測範圍
            in_pass_zone = ob_x <= 0

            print(f"\n步驟 {step}: 障礙物 #{i+1} 在玩家位置附近")
            print(f"  障礙物 x: {ob_x:.2f}")
            print(f"  球位置: y={env.y:.2f} (頂={ball_top:.2f}, 底={ball_bottom:.2f})")
            print(f"  間隙範圍: [{gap_top:.2f}, {gap_bottom:.2f}]")
            print(f"  在碰撞檢測區域: {in_collision_zone} (x in [-12, 12])")
            print(f"  在通過檢測區域: {in_pass_zone} (x <= 0)")
            print(
                f"  球是否在間隙內: {gap_top <= ball_top and ball_bottom <= gap_bottom}"
            )
            print(
                f"  球是否與障礙物碰撞: {ball_top < gap_top or ball_bottom > gap_bottom}"
            )
            print(f"  已標記為通過: {obs[3]}")

            if reward != 0:
                print(f"  ⭐ 獲得獎勵/懲罰: {reward}")

    if reward > 0:
        print(f"\n✅ 步驟 {step}: 成功通過障礙物！獎勵 = {reward}")
        print(f"   球位置: y={env.y:.2f}")
        print(f"   通過計數: {env.passed_count}")

    if done:
        print(f"\n❌ 步驟 {step}: 回合結束")
        print(f"   最終獎勵: {info['episode_score']}")
        print(f"   通過障礙物數: {env.passed_count}")
        break

print("\n" + "=" * 60)
print("測試完成")
print("=" * 60)
