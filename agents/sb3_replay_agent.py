"""Wrapper for replaying Stable-Baselines3 PPO models inside the GameUI.

The class keeps the public ``act`` interface expected by ``GameUI`` so the
SB3 policy can be swapped in without touching the rest of the loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from stable_baselines3 import PPO
except Exception as exc:  # pragma: no cover - optional dependency
    PPO = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class SB3ReplayAgent:
    """Lightweight adapter around an SB3 PPO policy."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        if PPO is None:  # pragma: no cover - handled at runtime
            raise RuntimeError(
                "stable-baselines3 is not installed. "
                "Please install it before using SB3 replay mode."
            ) from _IMPORT_ERROR

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"SB3 model not found: {path}")

        load_kwargs = {}
        if device:
            load_kwargs["device"] = device

        self.model = PPO.load(str(path), **load_kwargs)
        self.model_path = str(path)

    def act(self, state, explore: bool = False) -> Tuple[int, float, float]:
        obs = np.asarray(state, dtype=np.float32).reshape(1, -1)
        deterministic = not explore
        action, _ = self.model.predict(obs, deterministic=deterministic)
        value = 0.0
        if hasattr(self.model.policy, "predict_values"):
            value_arr = self.model.policy.predict_values(obs)
            try:
                value = float(value_arr.squeeze())
            except Exception:
                value = 0.0
        return int(action), 0.0, float(value)
