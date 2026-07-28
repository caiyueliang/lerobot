#!/usr/bin/env python

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print unique LeRobot task names from episode metadata.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/datasets/G1_Dex1_merged_3cams"),
        help="LeRobot dataset root directory.",
    )
    return parser.parse_args()


def normalize_tasks(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(task) for task in value]


def main() -> None:
    args = parse_args()
    episodes_dir = args.dataset_root / "meta" / "episodes"
    episode_files = sorted(episodes_dir.rglob("*.parquet"))

    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found under {episodes_dir}")

    episodes = pd.concat((pd.read_parquet(path) for path in episode_files), ignore_index=True)
    if "tasks" not in episodes.columns:
        raise KeyError(f"'tasks' column not found in episode metadata under {episodes_dir}")

    task_counts: Counter[str] = Counter()
    task_order: list[str] = []

    for tasks in episodes["tasks"]:
        for task in normalize_tasks(tasks):
            if task not in task_counts:
                task_order.append(task)
            task_counts[task] += 1

    print(f"Dataset root: {args.dataset_root}")
    print(f"Episode count: {len(episodes)}")
    print(f"Unique task count: {len(task_order)}")
    print()
    print("Unique task names:")
    for index, task in enumerate(task_order, start=1):
        print(f"{index:03d}. {task}  episodes={task_counts[task]}")


if __name__ == "__main__":
    main()
