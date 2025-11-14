"""測試訓練好的模型能否突破 164 分"""

import torch

from agents.networks import ActorCritic
from game.environment import GameEnv


def test_trained_model():
    """使用訓練好的模型測試"""

    # 檢查檢查點
    import os

    checkpoint_dir = "checkpoints"
    checkpoints = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]

    if not checkpoints:
        print("❌ 沒有找到訓練檢查點")
        return

    # 使用最新的檢查點
    latest = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest)

    print(f"載入檢查點: {checkpoint_path}")

    # 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 嘗試不同的鍵名
    if "model_state" in checkpoint:
        net.load_state_dict(checkpoint["model_state"])
    elif "model_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["model_state_dict"])
    elif "net" in checkpoint:
        net.load_state_dict(checkpoint["net"])
    else:
        # 直接載入（可能整個文件就是 state_dict）
        net.load_state_dict(checkpoint)

    net.eval()

    # 創建環境
    env = GameEnv()
    print(f"環境 max_steps: {env.max_steps}")

    # 運行 5 個回合
    print("\n" + "=" * 60)
    print("開始測試 5 個回合...")
    print("=" * 60)

    scores = []
    steps_list = []

    for episode in range(5):
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0

        while not done:
            # 使用模型選擇動作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                logits, _ = net(state_tensor)
                prob = torch.sigmoid(logits).item()
                action = 1 if prob > 0.5 else 0

            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

            # 防止無限循環
            if steps > 5000:
                print(f"  警告：第 {episode+1} 回合超過 5000 步，強制結束")
                break

        score = info.get("episode_score", episode_reward)
        win = info.get("win", False)
        scores.append(score)
        steps_list.append(steps)

        status = "🏆 勝利" if win else "💥 碰撞"
        print(f"回合 {episode+1}: {status} | 分數: {score:.0f} | 步數: {steps}")

    print("\n" + "=" * 60)
    print("測試結果統計:")
    print("=" * 60)
    print(f"平均分數: {sum(scores)/len(scores):.1f}")
    print(f"最高分數: {max(scores):.0f}")
    print(f"最低分數: {min(scores):.0f}")
    print(f"平均步數: {sum(steps_list)/len(steps_list):.0f}")

    # 檢查是否有分數卡在 164
    scores_at_164 = [s for s in scores if 163 <= s <= 165]
    if len(scores_at_164) >= 3:
        print(f"\n⚠️ 警告：有 {len(scores_at_164)} 個回合的分數在 164 左右")
        print("   這可能表示仍然存在限制")
    elif max(scores) > 200:
        print("\n✅ 成功：AI 能夠突破 164 分！")
    else:
        print("\n✅ 測試完成，沒有發現明顯的 164 分限制")


if __name__ == "__main__":
    test_trained_model()
