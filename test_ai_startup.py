"""測試 AI 啟動流程 - 找出阻塞點"""

import sys
import threading
import time
import traceback

sys.path.insert(0, ".")

print("=" * 60)
print("測試 AI 啟動流程")
print("=" * 60)

print("\n步驟 1: 導入模組...")
print("\n步驟 2: 創建 PPOAgent...")
try:
    from game.environment import GameEnv

    print("  ✅ GameEnv 導入成功")
except Exception as e:
    print(f"  ❌ GameEnv 導入失敗: {e}")
    sys.exit(1)

try:
    from agents.ppo_agent import PPOAgent

    print("  ✅ PPOAgent 導入成功")
except Exception as e:
    print(f"  ❌ PPOAgent 導入失敗: {e}")
    sys.exit(1)

try:
    from agents.pytorch_trainer import PPOTrainer

    print("  ✅ PPOTrainer 導入成功")
except Exception as e:
    print(f"  ❌ PPOTrainer 導入失敗: {e}")
    sys.exit(1)

start = time.time()
try:
    agent = PPOAgent()
    elapsed = (time.time() - start) * 1000
    print(f"  ✅ PPOAgent 創建成功 (耗時: {elapsed:.2f}ms)")
except Exception as e:
    print(f"  ❌ PPOAgent 創建失敗: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n步驟 3: 創建 PPOTrainer...")
start = time.time()
try:
    trainer = PPOTrainer()
    elapsed = (time.time() - start) * 1000
    print(f"  ✅ PPOTrainer 創建成功 (耗時: {elapsed:.2f}ms)")
except Exception as e:
    print(f"  ❌ PPOTrainer 創建失敗: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n步驟 4: 共享網絡...")
start = time.time()
try:
    agent.net = trainer.net
    agent.opt = trainer.opt
    elapsed = (time.time() - start) * 1000
    print(f"  ✅ 網絡共享成功 (耗時: {elapsed:.2f}ms)")
except Exception as e:
    print(f"  ❌ 網絡共享失敗: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n步驟 5: 創建訓練環境...")
start = time.time()
try:
    training_env = GameEnv()
    elapsed = (time.time() - start) * 1000
    print(f"  ✅ 訓練環境創建成功 (耗時: {elapsed:.2f}ms)")
except Exception as e:
    print(f"  ❌ 訓練環境創建失敗: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n步驟 6: 啟動訓練線程（模擬）...")
stop_event = threading.Event()


def _runner():
    try:
        print("    [訓練線程] 開始訓練...")
        # 模擬訓練啟動（只運行幾步）
        trainer.train(
            total_timesteps=100,  # 只訓練 100 步
            env=training_env,
            metrics_callback=lambda m: print(f"    [訓練線程] 步驟 {m.get('it', 0)}"),
            stop_event=stop_event,
            log_interval=50,
        )
        print("    [訓練線程] 訓練完成")
    except Exception as e:
        print(f"    [訓練線程] ❌ 訓練失敗: {e}")
        traceback.print_exc()


start = time.time()
t = threading.Thread(target=_runner, daemon=False)
print("  啟動線程...")
t.start()
elapsed = (time.time() - start) * 1000
print(f"  ✅ 線程啟動成功 (耗時: {elapsed:.2f}ms)")

print("\n步驟 7: 測試主線程是否被阻塞...")
print("  主線程等待 0.5 秒...")
time.sleep(0.5)
print("  ✅ 主線程沒有被阻塞")

print("\n步驟 8: 測試 agent 是否能正常使用...")
try:
    state = training_env.reset()
    action, logp, value = agent.act(state)
    print(f"  ✅ Agent 可以正常決策: action={action}")
except Exception as e:
    print(f"  ❌ Agent 決策失敗: {e}")
    traceback.print_exc()

print("\n步驟 9: 停止訓練線程...")
stop_event.set()
t.join(timeout=2.0)
if t.is_alive():
    print("  ⚠️  訓練線程未在 2 秒內停止")
else:
    print("  ✅ 訓練線程已停止")

print("\n" + "=" * 60)
print("診斷結果：")
print("=" * 60)

print(
    """
如果以上步驟都成功：
  → 說明 AI 啟動邏輯本身沒有問題
  → 問題可能在 UI 事件處理或 PyGame 初始化

如果某個步驟失敗或阻塞：
  → 該步驟就是導致無回應的原因
"""
)
