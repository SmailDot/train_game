#!/usr/bin/env python3
"""Render PPO network as interconnected nodes for documentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


@dataclass
class LayerSpec:
    label: str
    full_size: int
    display_nodes: int


SHARED_LAYERS: List[LayerSpec] = [
    LayerSpec("Input (5 features)", 5, 5),
    LayerSpec("Shared Hidden 1 (256 ReLU)", 256, 6),
    LayerSpec("Shared Hidden 2 (256 ReLU)", 256, 6),
    LayerSpec("Shared Hidden 3 (256 ReLU)", 256, 6),
]

ACTOR_LAYER = LayerSpec("Actor logits (2 actions)", 2, 2)
CRITIC_LAYER = LayerSpec("Critic value (1 scalar)", 1, 1)


def _layer_positions(
    x: float, display_nodes: int, y_offset: float = 0.0
) -> List[Tuple[float, float]]:
    spacing = 0.7
    total_height = spacing * (display_nodes - 1)
    start = y_offset - total_height / 2
    return [(x, start + i * spacing) for i in range(display_nodes)]


def _draw_layer(ax, nodes: List[Tuple[float, float]], label: str):
    for x, y in nodes:
        circle = plt.Circle((x, y), 0.1, color="#2b6cb0", alpha=0.9)
        ax.add_patch(circle)
    ax.text(
        nodes[0][0], nodes[-1][1] + 0.5, label, ha="center", va="bottom", fontsize=9
    )


def _connect_layers(
    ax, left: List[Tuple[float, float]], right: List[Tuple[float, float]]
):
    for x1, y1 in left:
        for x2, y2 in right:
            ax.plot([x1, x2], [y1, y2], color="#a0aec0", linewidth=0.6, alpha=0.7)


def main() -> None:
    plt.rcParams.update({"figure.figsize": (11, 4.5)})
    fig, ax = plt.subplots()

    layer_nodes: List[List[Tuple[float, float]]] = []
    x = 0.0
    x_step = 2.0

    for spec in SHARED_LAYERS:
        nodes = _layer_positions(x, spec.display_nodes)
        _draw_layer(ax, nodes, spec.label)
        layer_nodes.append(nodes)
        x += x_step

    actor_nodes = _layer_positions(x, ACTOR_LAYER.display_nodes, y_offset=1.0)
    critic_nodes = _layer_positions(x, CRITIC_LAYER.display_nodes, y_offset=-1.0)
    _draw_layer(ax, actor_nodes, ACTOR_LAYER.label)
    _draw_layer(ax, critic_nodes, CRITIC_LAYER.label)

    for left, right in zip(layer_nodes[:-1], layer_nodes[1:]):
        _connect_layers(ax, left, right)

    last_shared = layer_nodes[-1]
    _connect_layers(ax, last_shared, actor_nodes)
    _connect_layers(ax, last_shared, critic_nodes)

    ax.set_xlim(-1, x + 1)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")
    ax.set_title(
        "PPO Policy / Value Network (connections shown between layers)", pad=18
    )

    out_path = Path("outputs/plots/PPO_policy_architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    print(f"🧠 Saved {out_path}")


if __name__ == "__main__":
    main()
