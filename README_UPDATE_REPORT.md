# README Update Report

## Summary
The `README.md` file has been updated to remove references to irrelevant algorithms (SAC, TD3, DQN, etc.), ensuring the documentation aligns with the current PPO-only focus of the project.

## Changes
1. **Roadmap Section**: Verified that the "Add more RL algorithms (SAC, TD3, DDPG)" item is removed.
2. **Training Formulas**: Verified that formulas for DQN, SAC, and TD3 are removed.
3. **Code Cleanup**:
   - Removed `_launch_multi_window_view` from `game/ui.py` which referenced "PPO, SAC, DQN, Double DQN, TD3".
   - Removed the "Multi-Window View" button from the UI.

## Verification
- `grep` search for "SAC", "TD3", "DQN" in `README.md` returns no matches.
- `game/ui.py` no longer contains code or print statements referencing these algorithms.
- Unit tests passed successfully.
