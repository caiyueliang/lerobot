#!/usr/bin/env python

"""Adapters for PI0.5 datasets that store robot state and action in split fields."""

from copy import deepcopy
from typing import Any

import torch

from lerobot.utils.constants import ACTION, OBS_STATE

# Keep this order aligned with the robot joint/action layout used by G1 Dex1.
PI05_SPLIT_STATE_KEYS = (
    "observation.state.left_arm",
    "observation.state.left_gripper",
    "observation.state.right_arm",
    "observation.state.right_gripper",
)

PI05_SPLIT_ACTION_KEYS = (
    "action.left_arm",
    "action.left_gripper",
    "action.right_arm",
    "action.right_gripper",
)


def _concatenate_batch_fields(batch: dict[str, Any], keys: tuple[str, ...], target_key: str) -> None:
    missing_keys = [key for key in keys if key not in batch]
    if missing_keys:
        raise KeyError(f"Cannot build '{target_key}'. Missing dataset fields: {missing_keys}")

    batch[target_key] = torch.cat([torch.as_tensor(batch[key]) for key in keys], dim=-1)


def adapt_pi05_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Return a batch containing the canonical PI0.5 state and action fields.

    Existing canonical fields are preserved, which keeps this helper safe for standard
    LeRobot datasets as well as split-field datasets.
    """

    adapted_batch = dict(batch)
    if OBS_STATE not in adapted_batch:
        _concatenate_batch_fields(adapted_batch, PI05_SPLIT_STATE_KEYS, OBS_STATE)
    if ACTION not in adapted_batch:
        _concatenate_batch_fields(adapted_batch, PI05_SPLIT_ACTION_KEYS, ACTION)
    return adapted_batch


def _concatenate_feature_stats(
    stats: dict[str, dict[str, Any]], source_keys: tuple[str, ...], target_key: str
) -> dict[str, Any]:
    missing_keys = [key for key in source_keys if key not in stats]
    if missing_keys:
        raise KeyError(f"Cannot build stats for '{target_key}'. Missing stats fields: {missing_keys}")

    common_stat_names = set(stats[source_keys[0]])
    for key in source_keys[1:]:
        common_stat_names.intersection_update(stats[key])

    concatenated_stats: dict[str, Any] = {}
    for stat_name in common_stat_names:
        values = [torch.as_tensor(stats[key][stat_name]) for key in source_keys]
        if stat_name == "count":
            concatenated_stats[stat_name] = values[0].clone()
        else:
            concatenated_stats[stat_name] = torch.cat([value.reshape(-1) for value in values])
    return concatenated_stats


def adapt_pi05_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Add canonical PI0.5 stats without mutating the dataset metadata."""

    adapted_stats = deepcopy(stats)
    if OBS_STATE not in adapted_stats:
        adapted_stats[OBS_STATE] = _concatenate_feature_stats(adapted_stats, PI05_SPLIT_STATE_KEYS, OBS_STATE)
    if ACTION not in adapted_stats:
        adapted_stats[ACTION] = _concatenate_feature_stats(adapted_stats, PI05_SPLIT_ACTION_KEYS, ACTION)
    return adapted_stats
