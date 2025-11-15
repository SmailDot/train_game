#!/usr/bin/env python3
"""Plot PPO scalar metrics exported from TensorBoard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

LOSS_TAGS = [
    "train/loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
]

PERF_TAGS = [
    "rollout/ep_rew_mean",
    "env/win_rate",
]

STYLE = {
    "figure.figsize": (10, 5.5),
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def load_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No scalar rows found in {csv_path}")
    table = df.pivot_table(index="step", columns="tag", values="value")
    table.sort_index(inplace=True)
    return table


def plot_lines(
    table: pd.DataFrame,
    tags: List[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots()

    for tag in tags:
        if tag not in table.columns:
            continue
        ax.plot(table.index, table[tag], label=tag)

    if not ax.lines:
        raise ValueError(f"None of the requested tags {tags} found in table")

    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(ylabel)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    try:
        rel_path = output_path.relative_to(Path.cwd())
    except ValueError:
        rel_path = output_path

    print(f"📈 Saved {rel_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO scalar metrics")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/metrics/PPO_15_scalars.csv"),
        help="CSV file created by export_scalars.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory to store generated figures",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="PPO_15",
        help="Filename prefix for the generated figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = load_table(args.csv)

    loss_path = args.out_dir / f"{args.prefix}_losses.png"
    plot_lines(
        table,
        LOSS_TAGS,
        title="PPO Training Loss Curves",
        ylabel="Loss",
        output_path=loss_path,
    )

    perf_path = args.out_dir / f"{args.prefix}_performance.png"
    plot_lines(
        table,
        PERF_TAGS,
        title="PPO Reward / Win Rate",
        ylabel="Value",
        output_path=perf_path,
    )


if __name__ == "__main__":
    main()
