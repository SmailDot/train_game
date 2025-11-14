"""快速測試：驗證遊戲不再在 164 分或 1000 步時強制結束"""

from game.environment import GameEnv


def test_no_step_limit():
    """測試遊戲沒有步數限制"""
    env = GameEnv()

    print(f"環境 max_steps 設定: {env.max_steps}")
    assert env.max_steps is None, "max_steps 應該是 None (無限制)"

    # 重置環境
    state = env.reset()

    # 模擬運行超過 1000 步
    steps = 0
    total_reward = 0
    done = False

    print("\n開始測試...")
    print("目標：運行超過 1000 步且不被步數限制強制結束")

    while not done and steps < 2000:
        # 始終跳躍（簡單策略）
        action = 1 if steps % 10 < 5 else 0
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        # 每 100 步打印一次
        if steps % 100 == 0:
            score = info.get("episode_score", total_reward)
            print(f"步數: {steps}, 分數: {score:.1f}, 完成: {done}")

        # 如果在 1000 步時結束，檢查原因
        if steps == 1000 and done:
            print(f"\n❌ 錯誤：在第 1000 步時遊戲結束！")
            print(f"   最終分數: {info.get('episode_score', total_reward):.1f}")
            print(f"   這表示 max_steps 限制仍然存在")
            assert False, "遊戲不應該在 1000 步時結束"

    if steps > 1000:
        print(f"\n✅ 成功：遊戲運行了 {steps} 步，沒有被步數限制強制結束")
        print(f"   最終分數: {info.get('episode_score', total_reward):.1f}")
        print(
            f"   遊戲結束原因: {'碰撞' if done and not info.get('win') else '勝利' if info.get('win') else '自然結束'}"
        )
    elif done:
        print(f"\n✅ 遊戲在 {steps} 步時自然結束（碰撞或勝利）")
        print(f"   最終分數: {info.get('episode_score', total_reward):.1f}")
        print(f"   是否勝利: {info.get('win', False)}")
    else:
        print(f"\n⚠️ 測試在 {steps} 步時停止（達到測試上限 2000）")


if __name__ == "__main__":
    test_no_step_limit()
    print("\n" + "=" * 60)
    print("測試完成！")
