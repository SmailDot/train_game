"""測試環境獎勵機制是否正常工作"""

from game.environment import GameEnv

# 測試 1: 隨機動作
print("=" * 60)
print("測試 1: 隨機動作測試")
print("=" * 60)

env = GameEnv(seed=42)
state = env.reset()

episodes_completed = 0
total_rewards = []
episode_reward = 0
episode_steps = 0

for step in range(1000):
    # 隨機選擇動作（50% 機率跳躍）
    action = step % 3 == 0  # 每3步跳一次

    state, reward, done, info = env.step(action)
    episode_reward += reward
    episode_steps += 1

    if reward > 0:
        print(f"✅ 步驟 {step}: 獲得獎勵 +{reward:.0f}! (通過障礙物)")
    elif reward < 0:
        print(f"❌ 步驟 {step}: 獲得懲罰 {reward:.0f} (碰撞)")

    if done:
        episodes_completed += 1
        total_rewards.append(episode_reward)
        print(f"\n📊 回合 {episodes_completed} 結束:")
        print(f"   總獎勵: {episode_reward:.0f}")
        print(f"   存活步數: {episode_steps}")
        print(f"   通過障礙物數: {env.passed_count}\n")

        episode_reward = 0
        episode_steps = 0
        state = env.reset()

print("\n" + "=" * 60)
print("測試結果統計")
print("=" * 60)
print(f"完成回合數: {episodes_completed}")
print(
    f"平均獎勵: {sum(total_rewards) / len(total_rewards) if total_rewards else 0:.2f}"
)
print(f"最高獎勵: {max(total_rewards) if total_rewards else 0:.0f}")
print(f"最低獎勵: {min(total_rewards) if total_rewards else 0:.0f}")

# 測試 2: 一直跳躍
print("\n" + "=" * 60)
print("測試 2: 一直跳躍策略")
print("=" * 60)

env = GameEnv(seed=42)
state = env.reset()
episode_reward = 0

for step in range(100):
    action = 1  # 一直跳
    state, reward, done, info = env.step(action)
    episode_reward += reward

    if done:
        print(f"回合結束: 獎勵={episode_reward:.0f}, 步數={step+1}")
        break

# 測試 3: 從不跳躍
print("\n" + "=" * 60)
print("測試 3: 從不跳躍策略")
print("=" * 60)

env = GameEnv(seed=42)
state = env.reset()
episode_reward = 0

for step in range(100):
    action = 0  # 不跳
    state, reward, done, info = env.step(action)
    episode_reward += reward

    if done:
        print(f"回合結束: 獎勵={episode_reward:.0f}, 步數={step+1}")
        break

print("\n✅ 環境測試完成！")
