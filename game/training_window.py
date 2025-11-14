"""Utility for displaying a lightweight PPO training dashboard in a side window.

The dashboard runs in a dedicated process so that the primary Pygame window
remains responsive.  Metrics and optional network weight snapshots flow through
an inter-process queue.
"""

from __future__ import annotations

import os
import queue
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue
from typing import Dict, Iterable, List, Optional

import numpy as np

_MAX_HISTORY = 200


@dataclass
class _Payload:
    metrics: Dict
    weights: Optional[np.ndarray]


def _prepare_fonts(pygame):
    fonts = {}
    candidates = [
        "microsoftyahei",
        "microsoftyaheimicrosoftyaheiui",
        "microsoftyaheiui",
        "simhei",
        "arial",
    ]
    for name in candidates:
        try:
            fonts["body"] = pygame.font.SysFont(name, 20)
            fonts["large"] = pygame.font.SysFont(name, 28)
            fonts["title"] = pygame.font.SysFont(name, 32)
            test = fonts["body"].render("測試", True, (255, 255, 255))
            if test.get_width() > 0:
                break
        except Exception:
            continue
    if "body" not in fonts:
        fonts["body"] = pygame.font.Font(None, 20)
        fonts["large"] = pygame.font.Font(None, 28)
        fonts["title"] = pygame.font.Font(None, 32)
    return fonts


def _limit_history(histories: Dict[str, list], steps: Optional[list] = None) -> None:
    overflow = 0
    for series in histories.values():
        overflow = max(overflow, len(series) - _MAX_HISTORY)
    if steps is not None:
        overflow = max(overflow, len(steps) - _MAX_HISTORY)
    if overflow > 0:
        for series in histories.values():
            if overflow >= len(series):
                series.clear()
            else:
                del series[:overflow]
        if steps is not None:
            if overflow >= len(steps):
                steps.clear()
            else:
                del steps[:overflow]


def _update_histories(histories: Dict[str, list], metrics: Dict, steps: list) -> None:
    key_map = {
        "policy": "policy_loss",
        "value": "value_loss",
        "entropy": "entropy",
        "total": "loss",
    }
    appended = False
    for series_name, metric_key in key_map.items():
        value = metrics.get(metric_key)
        if value is None:
            continue
        try:
            histories.setdefault(series_name, []).append(float(value))
            appended = True
        except Exception:
            continue
    if appended:
        iteration = metrics.get("it")
        if iteration is None:
            try:
                iteration = steps[-1] + 1
            except Exception:
                iteration = len(histories.get("total", []))
        try:
            steps.append(int(iteration))
        except Exception:
            steps.append(steps[-1] + 1 if steps else 0)
    _limit_history(histories, steps)


def _draw_network(
    surface, fonts, iteration: int, weight_matrix: Optional[np.ndarray]
) -> int:
    import pygame

    width, height = surface.get_size()
    margin = 40

    title = fonts["title"].render(
        f"神經網路（第 {iteration} 次更新）", True, (100, 200, 255)
    )
    surface.blit(title, (margin, margin))

    net_top = margin + title.get_height() + 16
    max_net_height = max(220, height - net_top - 240)
    net_height = min(max_net_height, max(240, int(height * 0.4)))
    net_height = max(220, net_height)

    area_width = max(420, width - 2 * margin)
    layers = [5, 64, 64, 2]
    names = ["Input\n(State)", "Hidden 1", "Hidden 2", "Output\n(Action)"]
    spacing_x = area_width / max(1, len(layers) - 1)

    display_cap = max(6, int(net_height / 36))

    positions: list[list[tuple[int, int]]] = []
    truncated: List[bool] = []
    for index, count in enumerate(layers):
        layer_x = int(margin + index * spacing_x)
        visible = min(count, display_cap)
        gap = net_height / (visible + 1)
        locs = []
        for node_idx in range(visible):
            layer_y = int(net_top + (node_idx + 1) * gap)
            locs.append((layer_x, layer_y))
        flag = count > visible
        if flag:
            locs.append((layer_x, int(net_top + net_height - gap * 0.5)))
        positions.append(locs)
        truncated.append(flag)

    intensity: Optional[list[float]] = None
    if weight_matrix is not None:
        try:
            weights = np.asarray(weight_matrix, dtype=np.float32)
            if weights.ndim == 1:
                weights = weights[None, :]
            magnitudes = np.abs(weights).mean(axis=0)
            maximum = float(np.max(magnitudes)) if np.max(magnitudes) != 0 else 1.0
            intensity = (magnitudes / maximum).tolist()
        except Exception:
            intensity = None

    for layer_idx in range(len(positions) - 1):
        for start_idx, start_pos in enumerate(positions[layer_idx]):
            if truncated[layer_idx] and start_idx == len(positions[layer_idx]) - 1:
                continue
            for end_idx, end_pos in enumerate(positions[layer_idx + 1]):
                if (
                    truncated[layer_idx + 1]
                    and end_idx == len(positions[layer_idx + 1]) - 1
                ):
                    continue
                blend = layer_idx / (len(positions) - 1)
                color = (
                    int(100 + blend * 110),
                    int(160 - blend * 110),
                    int(255 - blend * 70),
                )
                pygame.draw.line(surface, color, start_pos, end_pos, 1)

    for layer_idx, locs in enumerate(positions):
        for node_idx, pos in enumerate(locs):
            is_placeholder = truncated[layer_idx] and node_idx == len(locs) - 1
            blend = layer_idx / (len(layers) - 1)
            base_color = [
                int(80 + blend * 120),
                int(120 + blend * 30),
                int(255 - blend * 50),
            ]
            if (
                layer_idx == 1
                and not is_placeholder
                and intensity is not None
                and node_idx < len(intensity)
            ):
                boost = intensity[node_idx]
                base_color[0] = min(255, int(base_color[0] + 120 * boost))
                base_color[1] = min(255, int(base_color[1] + 40 * boost))
            if is_placeholder:
                placeholder = fonts["body"].render("...", True, (200, 200, 200))
                surface.blit(
                    placeholder, (pos[0] - placeholder.get_width() // 2, pos[1] + 6)
                )
                continue
            shadow = tuple(max(0, c // 3) for c in base_color)
            pygame.draw.circle(surface, shadow, pos, 12)
            pygame.draw.circle(surface, tuple(base_color), pos, 8)
            pygame.draw.circle(surface, (255, 255, 255), pos, 3)

    label_base = net_top + net_height + 20
    for index, (name, count) in enumerate(zip(names, layers)):
        layer_x = int(margin + index * spacing_x)
        for offset, line in enumerate(name.split("\n")):
            text = fonts["body"].render(line, True, (190, 190, 210))
            rect = text.get_rect(center=(layer_x, int(label_base + offset * 18)))
            surface.blit(text, rect)
        count_text = fonts["body"].render(f"({count})", True, (120, 120, 150))
        surface.blit(
            count_text, count_text.get_rect(center=(layer_x, int(label_base + 44)))
        )

    return int(net_top + net_height + 70)


def _draw_losses(
    surface, fonts, histories: Dict[str, list], steps: list, top: int
) -> None:
    import pygame

    width, height = surface.get_size()
    margin = 40

    available_height = height - top - margin
    if available_height < 200:
        top = max(margin, height - margin - 200)
        available_height = height - top - margin

    area = pygame.Rect(margin, top, width - 2 * margin, max(200, available_height))
    pygame.draw.rect(surface, (25, 25, 35), area)
    pygame.draw.rect(surface, (60, 60, 80), area, 2)

    title = fonts["large"].render("Loss Function", True, (255, 150, 100))
    surface.blit(title, (area.x, area.y - 34))

    palette = {
        "policy": (255, 110, 110),
        "value": (110, 255, 150),
        "entropy": (150, 150, 255),
        "total": (255, 255, 120),
    }

    descriptions = {
        "policy": ["衡量當前策略與", "最佳策略的差距"],
        "value": ["評估狀態價值", "預測的準確性"],
        "entropy": ["動作選擇的", "隨機性程度"],
        "total": ["所有損失的加權總和", ""],
    }

    legend_width = min(360, area.width // 2)
    legend_x = area.x + 16
    legend_y = area.y + 18
    body_line = fonts["body"].get_linesize()

    max_points = max((len(values) for values in histories.values()), default=0)
    if max_points < 2:
        hint = fonts["body"].render("等待訓練數據中...", True, (160, 160, 160))
        surface.blit(hint, (area.x + 24, area.y + area.height // 2))
        return

    # 計算每個項目需要的高度（標題+值+說明(2行)+空行）
    item_spacing = body_line * 4 + 8  # 標題行 + 說明2行 + 空行

    current_y = legend_y
    for idx, key in enumerate(["policy", "value", "entropy", "total"]):
        color = palette[key]

        # 第一行：標題和值
        pygame.draw.rect(surface, color, (legend_x, current_y + 2, 14, 14))
        series = histories.get(key, [0])
        latest = series[-1] if series else 0.0
        label = fonts["body"].render(f"{key.title()}: {latest:+.4f}", True, color)
        surface.blit(label, (legend_x + 20, current_y))

        # 第二行和第三行：中文說明（每行最多12字）
        desc_lines = descriptions[key]
        desc_y = current_y + body_line + 2

        for line_idx, line_text in enumerate(desc_lines):
            line_surface = fonts["body"].render(line_text, True, (140, 140, 160))
            surface.blit(line_surface, (legend_x + 20, desc_y + line_idx * body_line))

        # 移動到下一個項目（包含空行）
        current_y += item_spacing

    plot = pygame.Rect(
        legend_x + legend_width,
        area.y + 14,
        max(120, area.width - legend_width - 30),
        area.height - 32,
    )
    pygame.draw.rect(surface, (18, 18, 26), plot)

    def normalise(series: Iterable[float]) -> Optional[np.ndarray]:
        values = np.asarray(list(series), dtype=np.float32)
        if values.size == 0:
            return None
        values = values[-plot.width :]
        minimum = float(values.min())
        maximum = float(values.max())
        if abs(maximum - minimum) < 1e-6:
            return values - minimum
        return (values - minimum) / (maximum - minimum)

    for key in ["policy", "value", "entropy", "total"]:
        normalised = normalise(histories.get(key, []))
        if normalised is None or normalised.size < 2:
            continue
        points = []
        for idx, val in enumerate(normalised):
            x = plot.x + int(idx * (plot.width - 2) / max(1, normalised.size - 1))
            y = plot.y + plot.height - int(val * (plot.height - 6)) - 3
            points.append((x, y))
        if len(points) > 1:
            pygame.draw.lines(surface, palette[key], False, points, 2)

    visible_count = min(len(steps), plot.width)
    if visible_count > 1:
        axis_steps = steps[-visible_count:]
        pygame.draw.line(
            surface,
            (90, 90, 120),
            (plot.x, plot.bottom - 2),
            (plot.right, plot.bottom - 2),
            1,
        )
        tick_indices = sorted({0, visible_count // 2, visible_count - 1})
        for idx in tick_indices:
            x = plot.x + int(idx * (plot.width - 2) / max(1, visible_count - 1))
            pygame.draw.line(
                surface, (130, 130, 150), (x, plot.bottom - 2), (x, plot.bottom - 8), 1
            )
            label = fonts["body"].render(str(axis_steps[idx]), True, (170, 170, 190))
            surface.blit(label, (x - label.get_width() // 2, plot.bottom + 4))


def _training_window_process(queue_, stop_event, width: int, height: int, title: str):
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

    import pygame

    pygame.init()
    pygame.font.init()
    min_width, min_height = 720, 520
    width = max(min_width, width)
    height = max(min_height, height)
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    base_title = title or "AI Training Visualization"
    pygame.display.set_caption(f"{base_title} ({width}x{height})")
    clock = pygame.time.Clock()

    fonts = _prepare_fonts(pygame)
    histories = {"policy": [], "value": [], "entropy": [], "total": []}
    iteration = 0
    weights = None
    history_steps: List[int] = []

    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
            elif event.type == pygame.VIDEORESIZE:
                width = max(min_width, event.w)
                height = max(min_height, event.h)
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                pygame.display.set_caption(f"{base_title} ({width}x{height})")

        try:
            while True:
                payload: _Payload = queue_.get_nowait()
                metrics = payload.metrics or {}
                iteration = int(metrics.get("it", iteration))
                _update_histories(histories, metrics, history_steps)
                if payload.weights is not None:
                    weights = payload.weights
        except queue.Empty:
            pass

        screen.fill((15, 15, 20))
        losses_top = _draw_network(screen, fonts, iteration, weights)
        _draw_losses(screen, fonts, histories, history_steps, losses_top + 20)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


class TrainingWindow:
    """Spawn a detached dashboard process to visualise live training metrics."""

    WIDTH = 960
    HEIGHT = 720

    def __init__(self, title: Optional[str] = None) -> None:
        self._queue = Queue(maxsize=10)
        self._stop_event = Event()
        self._process: Optional[Process] = None
        self._title = title or "AI Training Visualization"

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stop_event.clear()
        process = Process(
            target=_training_window_process,
            args=(self._queue, self._stop_event, self.WIDTH, self.HEIGHT, self._title),
            name="TrainingWindow",
            daemon=True,
        )
        process.start()
        self._process = process

    def update_data(self, metrics: Dict, weights: Optional[np.ndarray] = None) -> None:
        if self._process is None or not self._process.is_alive():
            return
        safe_weights = None
        if weights is not None:
            try:
                safe_weights = np.asarray(weights, dtype=np.float32)
            except Exception:
                safe_weights = None
        payload = _Payload(metrics=dict(metrics or {}), weights=safe_weights)
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                pass

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.is_alive():
            self._stop_event.set()
            self._process.join(timeout=2.0)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._process = None
        self._stop_event.clear()

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()
