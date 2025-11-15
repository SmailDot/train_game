#!/usr/bin/env python3
"""Utility to dump TensorBoard scalar summaries to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# NumPy 2 removed np.string_/np.unicode_. TensorBoard still references them.
if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

DEFAULT_TAGS: List[str] = [
    "train/loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "env/win_rate",
    "rollout/ep_rew_mean",
]


def collect_scalars(event_file: Path, tags: Iterable[str]) -> List[dict]:
    accumulator = event_accumulator.EventAccumulator(str(event_file))
    accumulator.Reload()

    available = set(accumulator.Tags().get("scalars", []))
    rows: List[dict] = []

    for tag in tags:
        if tag not in available:
            continue
        for scalar in accumulator.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": scalar.step,
                    "wall_time": scalar.wall_time,
                    "value": scalar.value,
                }
            )

    rows.sort(key=lambda row: (row["tag"], row["step"]))
    return rows


def resolve_event_file(run_dir: Path, event_file: Path | None) -> Path:
    if event_file:
        return event_file

    candidates = sorted(run_dir.glob("events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No event files found under {run_dir}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TensorBoard scalars to CSV")
    parser.add_argument(
        "--run",
        required=True,
        help="Name of the run directory under logs/tensorboard (e.g. PPO_15)",
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        help="Optional explicit path to an event file (overrides --run)",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=DEFAULT_TAGS,
        help="Scalar tags to export (defaults cover PPO losses + rewards)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV path (default: outputs/metrics/<run>_scalars.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path("logs/tensorboard") / args.run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    event_file = resolve_event_file(run_dir, args.event_file)
    tags = args.tags or DEFAULT_TAGS
    rows = collect_scalars(event_file, tags)

    output_path = (
        args.output
        if args.output
        else Path("outputs/metrics") / f"{args.run}_scalars.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"✅ Exported {len(rows)} scalar points from {event_file.name} to {output_path}"
    )


if __name__ == "__main__":
    main()
