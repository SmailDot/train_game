from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional


class ReplayBuffer:
    """A minimal experience replay buffer that stores transition dictionaries.

    Each transition is stored as a mapping with the common keys:
        state, action, reward, next_state, done
    Additional keys (for example, continuous_action or log_prob) are preserved
    so algorithms such as TD3 can recover auxiliary data during updates.
    """

    def __init__(self, capacity: int = 100_000):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self._capacity = int(capacity)
        self._buffer: Deque[Dict] = deque(maxlen=self._capacity)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def push(self, transition: Dict) -> None:
        self._buffer.append(dict(transition))

    append = push  # backwards-compatible alias

    def extend(self, transitions: Iterable[Dict]) -> None:
        for transition in transitions:
            self.push(transition)

    def sample(self, batch_size: int) -> List[Dict]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._buffer):
            message = (
                f"Cannot sample {batch_size} transitions from buffer of size "
                f"{len(self._buffer)}"
            )
            raise ValueError(message)
        # Random sampling without replacement keeps the simple dependency footprint
        import random

        indices = random.sample(range(len(self._buffer)), batch_size)
        return [self._buffer[idx] for idx in indices]

    def peek(self, count: Optional[int] = None) -> List[Dict]:
        if count is None or count >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-int(count) :]
