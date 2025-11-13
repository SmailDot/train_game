"""診斷腳本：測試碰撞檢測和 AI 模式"""

import sys
import time
import traceback

sys.path.insert(0, ".")

from game.environment import GameEnv

try:
    from agents.ppo_agent import PPOAgent
except Exception as exc:  # pragma: no cover - debug helper only
    print("\n❌ 無法匯入 PPOAgent：", exc)
    traceback.print_exc()
    PPOAgent = None

print("=" * 60)
print("診斷 1：碰撞檢測敏感度測試")
print("=" * 60)

env = GameEnv(seed=42)
state = env.reset()

print("\n初始狀態：")
print(f"  球位置 y: {env.y:.2f}")
print(f"  球速度 vy: {env.vy:.2f}")
print(f"  ScreenHeight: {env.ScreenHeight}")

# 獲取第一個障礙物
obs = env.obstacles[0]
print("\n最近障礙物：")
print(f"  x: {obs[0]:.2f}")
print(f"  gap_top: {obs[1]:.2f}")
print(f"  gap_bottom: {obs[2]:.2f}")
print(f"  gap_size: {obs[2] - obs[1]:.2f}")

# 測試碰撞邏輯
ball_margin = 15.0  # 環境中使用的
print("\n碰撞檢測設定：")
print(f"  ball_margin: {ball_margin}")
print("  UI 球半徑: 12")
print(f"  環境中的判定範圍: gap_top + {ball_margin} 到 gap_bottom - {ball_margin}")
print(f"  實際判定範圍: {obs[1] + ball_margin:.2f} 到 {obs[2] - ball_margin:.2f}")
print(f"  判定範圍大小: {(obs[2] - ball_margin) - (obs[1] + ball_margin):.2f}")

# 模擬球接近障礙物
print("\n模擬碰撞檢測：")
test_y_positions = [
    obs[1] - 5,  # 在 gap 外（上方）
    obs[1] + ball_margin - 2,  # 接近邊界（上方）
    obs[1] + ball_margin + 2,  # 剛進入安全區
    (obs[1] + obs[2]) / 2,  # 正中央
    obs[2] - ball_margin - 2,  # 剛離開安全區
    obs[2] - ball_margin + 2,  # 接近邊界（下方）
    obs[2] + 5,  # 在 gap 外（下方）
]

for test_y in test_y_positions:
    in_safe_zone = obs[1] + ball_margin < test_y < obs[2] - ball_margin
    distance_to_top = test_y - (obs[1] + ball_margin)
    distance_to_bottom = (obs[2] - ball_margin) - test_y
    print(
        f"  y={test_y:6.2f}: {'✅ 安全' if in_safe_zone else '❌ 碰撞'} "
        f"(距上界:{distance_to_top:5.2f}, 距下界:{distance_to_bottom:5.2f})"
    )

print("\n" + "=" * 60)
print("診斷 2：球半徑與判定範圍的不匹配")
print("=" * 60)

print("\n問題分析：")
print("  1. UI 渲染的球半徑: 12 像素")
print(f"  2. 環境碰撞判定的 ball_margin: {ball_margin} 像素")
print(f"  3. 差異: {15 - 12} 像素")
print("\n這意味著：")
print("  - 視覺上球還沒碰到障礙物")
print("  - 但碰撞判定已經觸發")
print("  - 玩家感覺「太敏感」")

print("\n" + "=" * 60)
print("診斷 3：測試 AI Agent 是否能正常運作")
print("=" * 60)

try:
    if PPOAgent is None:
        raise RuntimeError("PPOAgent 尚未正確匯入")
    agent = PPOAgent()
    print("\n✅ 成功創建 PPOAgent")

    # 測試 act 方法
    state = env.reset()
    print("\n測試 agent.act():")
    for i in range(5):
        action, logp, value = agent.act(state)
        print(f"  步驟 {i+1}: action={action}, logp={logp:.4f}, value={value:.4f}")
        state, reward, done, info = env.step(action)
        if done:
            print(f"    → 遊戲結束，分數: {info.get('episode_score', 0):.2f}")
            break

    if not done:
        print("  → Agent 存活 5 步")

except Exception as e:  # pragma: no cover - debug helper only
    print("\n❌ 創建或使用 PPOAgent 失敗：", e)
    traceback.print_exc()

print("\n" + "=" * 60)
print("診斷 4：測試主循環是否會阻塞")
print("=" * 60)

print("\n模擬 AI 模式主循環（10 步）：")
env.reset()
agent = PPOAgent() if PPOAgent else None
if agent is None:
    sys.exit(1)

for step in range(10):
    start = time.time()

    state = env._get_state()
    action, _, _ = agent.act(state)
    state, reward, done, info = env.step(action)

    elapsed = (time.time() - start) * 1000
    print(
        f"  步驟 {step+1}: action={action}, reward={reward:5.2f}, "
        f"done={done}, 耗時={elapsed:.2f}ms"
    )

    if done:
        print("    → 遊戲結束，重置環境")
        env.reset()

print("\n結論：")
print("  如果每步耗時 < 10ms，主循環不會阻塞")
print("  如果每步耗時 > 100ms，可能導致 UI 無回應")

print("\n" + "=" * 60)
print("診斷完成")
print("=" * 60)
