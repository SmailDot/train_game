"""
Run multiple training sessions in parallel, each in its own window.

This script uses multiprocessing to launch a separate GameUI instance for each
selected algorithm.
"""

import multiprocessing
import time

# Set start method to 'spawn' for compatibility with Pygame and macOS/Windows
# This must be done before any other multiprocessing or pygame imports
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    # Start method can only be set once
    pass

from game.ui import GameUI


def run_algorithm_in_window(algorithm_key: str):
    """
    Initializes and runs a GameUI instance for a specific algorithm.

    Args:
        algorithm_key: The key of the algorithm to run (e.g., 'ppo', 'sac').
    """
    print(f"Launching window for algorithm: {algorithm_key}...")
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
    # Available keys by default: "ppo", "sac", "dqn", "double_dqn"
    ALGORITHMS_TO_RUN = [
        "ppo",
        "sac",
        "dqn",
        "double_dqn",
    ]
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
