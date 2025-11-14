from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from game.environment import GameEnv
from game.training_window import TrainingWindow


@dataclass
class AlgorithmDescriptor:
    key: str
    name: str
    trainer_factory: Callable[[], object]
    use_vector_envs: bool = True
    vector_envs: int = 4
    hotkey: Optional[int] = None
    action_label: str = ""
    color: Tuple[int, int, int] = (180, 180, 210)
    window_title: Optional[str] = None


@dataclass
class AlgorithmState:
    descriptor: AlgorithmDescriptor
    trainer: Optional[object] = None
    agent: Optional[object] = None
    training_window: Optional[TrainingWindow] = None
    trainer_thread: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None
    status: str = "idle"
    iterations: int = 0
    ai_round: int = 0
    n: int = 0
    viewer_round: int = 0
    latest_metrics: Dict[str, float] = field(default_factory=dict)
    loss_history: Dict[str, List[float]] = field(
        default_factory=lambda: {"policy": [], "value": [], "entropy": [], "total": []}
    )
    agent_ready: bool = False


class AlgorithmManager:
    def __init__(self) -> None:
        self._algorithms: Dict[str, AlgorithmState] = {}
        self._active_key: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def active_key(self) -> Optional[str]:
        return self._active_key

    def active_state(self) -> Optional[AlgorithmState]:
        if self._active_key is None:
            return None
        return self._algorithms.get(self._active_key)

    def register(self, descriptor: AlgorithmDescriptor) -> None:
        if descriptor.key in self._algorithms:
            raise ValueError(f"Algorithm {descriptor.key} already registered")
        self._algorithms[descriptor.key] = AlgorithmState(descriptor=descriptor)
        if self._active_key is None:
            self._active_key = descriptor.key

    def keys(self) -> List[str]:
        return list(self._algorithms.keys())

    def descriptor(self, key: str) -> AlgorithmDescriptor:
        return self._algorithms[key].descriptor

    def state(self, key: Optional[str] = None) -> AlgorithmState:
        if key is None:
            key = self._active_key
        if key is None or key not in self._algorithms:
            raise KeyError("Algorithm not registered")
        return self._algorithms[key]

    def set_active(self, key: str) -> None:
        if key not in self._algorithms:
            raise KeyError(f"Unknown algorithm {key}")
        self._active_key = key

    def stop(self, key: str, wait: bool = True) -> None:
        slot = self.state(key)
        if slot.stop_event is not None:
            slot.stop_event.set()
        if wait and slot.trainer_thread is not None:
            slot.trainer_thread.join(timeout=5.0)
        if slot.training_window is not None:
            slot.training_window.stop()
            slot.training_window = None
        slot.trainer_thread = None
        slot.stop_event = None
        slot.trainer = None
        slot.agent_ready = False
        slot.status = "idle"

    def stop_all(self) -> None:
        for key in list(self._algorithms.keys()):
            self.stop(key)

    def _extract_weights(
        self, agent
    ) -> Optional[object]:  # pragma: no cover - best effort
        candidates = (
            getattr(agent, "net", None),
            getattr(agent, "actor", None),
            getattr(agent, "q_network", None),
        )
        for module in candidates:
            if module is None:
                continue
            getter = getattr(module, "get_weight_matrix", None)
            if callable(getter):
                try:
                    return getter()
                except Exception:
                    continue
        return None

    def _ingest_metrics(self, key: str, metrics: Dict[str, float]) -> None:
        slot = self._algorithms[key]
        with self._lock:
            slot.latest_metrics = dict(metrics)
            slot.iterations = max(
                slot.iterations, int(metrics.get("it", slot.iterations))
            )
            slot.ai_round += int(metrics.get("episode_count") or 0)
            slot.n = slot.iterations
            for series, metric_key in (
                ("policy", "policy_loss"),
                ("value", "value_loss"),
                ("entropy", "entropy"),
                ("total", "loss"),
            ):
                value = metrics.get(metric_key)
                if value is None:
                    continue
                try:
                    slot.loss_history[series].append(float(value))
                    if len(slot.loss_history[series]) > 400:
                        slot.loss_history[series] = slot.loss_history[series][-400:]
                except Exception:
                    pass

    def start(
        self,
        key: str,
        env_factory: Callable[[], GameEnv],
        metrics_consumer: Optional[Callable[[str, Dict[str, float]], None]] = None,
        force_reset: bool = False,
        setup_callback: Optional[Callable[[AlgorithmState], None]] = None,
        vector_env_override: Optional[int] = None,
    ) -> AlgorithmState:
        slot = self.state(key)
        if slot.trainer_thread is not None and slot.trainer_thread.is_alive():
            return slot

        trainer = slot.descriptor.trainer_factory()
        build_agent = getattr(trainer, "build_agent", None)
        if build_agent is None:
            raise AttributeError(
                f"Trainer {trainer.__class__.__name__} must implement build_agent()"
            )
        agent = build_agent()
        slot.agent = agent
        slot.trainer = trainer
        slot.agent_ready = True
        slot.status = "training"
        title = slot.descriptor.window_title or f"{slot.descriptor.name} 訓練視窗"
        slot.training_window = slot.training_window or TrainingWindow(title=title)
        slot.training_window.start()
        stop_event = threading.Event()
        slot.stop_event = stop_event

        if setup_callback is not None:
            try:
                setup_callback(slot)
            except Exception:
                slot.status = "error"
                slot.stop_event = None
                if slot.training_window is not None:
                    try:
                        slot.training_window.stop()
                    except Exception:
                        pass
                    slot.training_window = None
                raise

        def _callback(metrics: Dict[str, float]):
            self._ingest_metrics(key, metrics)
            # 優先使用 metrics 中的權重（如果有的話）
            weights = metrics.get("weights")
            # 如果 metrics 沒有權重，則嘗試從 agent 提取
            if weights is None:
                weights = self._extract_weights(agent)
            if slot.training_window is not None:
                try:
                    slot.training_window.update_data(metrics, weights=weights)
                except Exception:
                    pass
            if metrics_consumer is not None:
                metrics_consumer(key, metrics)

        def _runner():
            try:
                kwargs = {
                    "metrics_callback": _callback,
                    "stop_event": stop_event,
                    "initial_iteration": 0 if force_reset else slot.iterations,
                }
                if slot.descriptor.use_vector_envs:
                    env_count = slot.descriptor.vector_envs
                    if vector_env_override is not None:
                        env_count = max(1, int(vector_env_override))
                    kwargs["envs"] = [env_factory() for _ in range(env_count)]
                else:
                    kwargs["env"] = env_factory()
                trainer.train(**kwargs)
            finally:
                slot.status = "idle"
                slot.trainer_thread = None
                slot.stop_event = None

        thread = threading.Thread(target=_runner, daemon=False, name=f"train-{key}")
        thread.start()
        slot.trainer_thread = thread
        return slot

    def latest_metrics(self, key: Optional[str] = None) -> Dict[str, float]:
        return dict(self.state(key).latest_metrics)

    def loss_history(self, key: Optional[str] = None) -> Dict[str, List[float]]:
        slot = self.state(key)
        return {name: list(values) for name, values in slot.loss_history.items()}

    def agents_snapshot(self) -> Dict[str, AlgorithmState]:
        return dict(self._algorithms)
