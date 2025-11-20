"""Wrapper for replaying Stable-Baselines3 PPO models inside the GameUI.

The class keeps the public ``act`` interface expected by ``GameUI`` so the
SB3 policy can be swapped in without touching the rest of the loop.
"""

from __future__ import annotations

import pickle
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

        # Try to load VecNormalize statistics
        # Common naming convention: model_name.zip -> vec_normalize_model_name.pkl
        # Or check for vec_normalize_6666.pkl if 6666 is in the name
        self.obs_rms = None
        self.epsilon = 1e-8
        self.clip_obs = 10.0

        # Attempt 1: Look for vec_normalize_{target}.pkl in the same dir
        # Assuming filename format ppo_game2048_{target}_final.zip
        filename = path.name
        parent = path.parent

        norm_path = None

        # Check for specific known patterns
        if "6666" in filename:
            norm_path = parent / "vec_normalize_6666.pkl"
        elif "test" in filename:
            norm_path = parent / "vec_normalize_test.pkl"
        elif "best_model" in filename:
            # If using best_model, try to find the main normalization file in models/
            # Since EvalCallback doesn't save pkl by default, we fallback to the final one
            # which is usually close enough.
            fallback_path = parent.parent / "models" / "vec_normalize_6666.pkl"
            if fallback_path.exists():
                norm_path = fallback_path

        # Check for generic pattern: replace .zip with .pkl and prepend vec_normalize_
        if not norm_path or not norm_path.exists():
            # Try direct replacement if user renamed it
            candidate = parent / filename.replace(".zip", ".pkl")
            if candidate.exists():
                norm_path = candidate

        if norm_path and norm_path.exists():
            try:
                with open(norm_path, "rb") as f:
                    vec_normalize = pickle.load(f)
                if hasattr(vec_normalize, "obs_rms"):
                    self.obs_rms = vec_normalize.obs_rms
                    self.epsilon = vec_normalize.epsilon
                    self.clip_obs = vec_normalize.clip_obs
                    print(f"✅ Loaded VecNormalize stats from {norm_path}")
            except Exception as e:
                print(f"⚠️ Failed to load VecNormalize stats: {e}")

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is not None:
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
        return obs

    def act(self, state, explore: bool = False) -> Tuple[int, float, float]:
        obs = np.asarray(state, dtype=np.float32).reshape(1, -1)

        # Apply normalization if available
        obs = self._normalize_obs(obs)

        deterministic = not explore
        action, _ = self.model.predict(obs, deterministic=deterministic)
        value = 0.0
        if hasattr(self.model.policy, "predict_values"):
            # Convert numpy obs to tensor for predict_values
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
            value_arr = self.model.policy.predict_values(obs_tensor)
            try:
                value = float(value_arr.item())
            except Exception:
                value = 0.0
        return int(action), 0.0, float(value)
