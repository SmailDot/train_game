"""
Run multiple training sessions in parallel, each in its own window.

This script uses multiprocessing to launch a separate GameUI instance for each
selected algorithm, with automatic window positioning.
"""

import multiprocessing
import os
import time

# Set start method to 'spawn' for compatibility with Pygame and macOS/Windows
# This must be done before any other multiprocessing or pygame imports
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    # Start method can only be set once
    pass

from game.ui import GameUI


def run_algorithm_in_window(algorithm_key: str, window_index: int, total_windows: int):
    """
    Initializes and runs a GameUI instance for a specific algorithm.

    Args:
        algorithm_key: The key of the algorithm to run (e.g., 'ppo', 'sac').
        window_index: Index of this window (0-based)
        total_windows: Total number of windows to arrange
    """
    print(f"Launching window for algorithm: {algorithm_key}...")

    # 設定視窗位置（自動排列）
    try:
        import ctypes

        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
    except Exception:
        # 如果無法獲取螢幕尺寸，使用預設值
        screen_width = 1920
        screen_height = 1080

    # 計算排列（2列 N行）
    cols = 2
    rows = (total_windows + cols - 1) // cols

    window_width = screen_width // cols
    window_height = screen_height // rows

    col = window_index % cols
    row = window_index // cols

    x = col * window_width
    y = row * window_height

    # 設定 SDL 視窗位置
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"

    try:
        ui = GameUI()
        ui.ai_manager.set_active(algorithm_key)

        # Programmatically start the AI training mode
        ui.selected_mode = "AI"
        ui.mode = "AI"
        ui.running = True
        ui.current_score = 0.0
        ui.game_over = False
        ui.paused = False
        ui.viewer_round = 0
        ui.last_ai_action = None
        ui.last_ai_action_prob = 0.0
        ui.last_ai_value = 0.0
        ui.agent_ready = False
        ui.ai_status = "initializing"

        # Start the training process in the background
        ui._start_algorithm_training(
            key=algorithm_key, force_reset=False, async_mode=True
        )

        # Run the UI loop
        ui.run()
    except Exception as e:
        print(f"Error in process for {algorithm_key}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # --- Configuration ---
    # Add the keys of the algorithms you want to run in parallel.
    # Available keys by default: "ppo", "sac", "td3", "dqn", "double_dqn"
    ALGORITHMS_TO_RUN = [
        "ppo",
        "sac",
        "dqn",
        "double_dqn",
        "td3",
    ]
    # ---------------------

    print(f"啟動 {len(ALGORITHMS_TO_RUN)} 個訓練視窗...")
    print("每個視窗將自動排列在螢幕上")
    print("按 Ctrl+C 或關閉所有視窗以停止\n")

    processes = []
    for idx, algo_key in enumerate(ALGORITHMS_TO_RUN):
        p = multiprocessing.Process(
            target=run_algorithm_in_window,
            args=(algo_key, idx, len(ALGORITHMS_TO_RUN)),
            daemon=False,
        )
        p.start()
        processes.append(p)
        # 稍微延遲啟動，避免資源衝突
        time.sleep(0.5)

    print(f"\n✅ 已啟動 {len(processes)} 個訓練進程")
    print("等待所有進程結束...\n")

    # 等待所有進程
    for p in processes:
        p.join()

    print("所有訓練視窗已關閉")
    # ---------------------

    print("Starting multi-algorithm training session...")
    print(f"Will launch windows for: {', '.join(ALGORITHMS_TO_RUN)}")

    processes = []
    for algo_key in ALGORITHMS_TO_RUN:
        process = multiprocessing.Process(
            target=run_algorithm_in_window, args=(algo_key,)
        )
        processes.append(process)
        process.start()
        # Stagger the launches slightly to avoid resource contention
        time.sleep(2)

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("All training windows have been closed.")
