#!/usr/bin/env python

from types import SimpleNamespace

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.dataset_adapter import adapt_pi05_batch, adapt_pi05_stats
from lerobot.utils.constants import ACTION, OBS_STATE

STATE_KEYS = (
    "observation.state.left_arm",
    "observation.state.right_arm",
    "observation.state.left_gripper",
    "observation.state.right_gripper",
)

ACTION_KEYS = (
    "action.left_arm",
    "action.right_arm",
    "action.left_gripper",
    "action.right_gripper",
)


def test_adapt_pi05_batch_concatenates_split_fields_without_reordering_arm_joints():
    """验证只调整字段块顺序，左右手臂内部的 7 个关节保持数据集原始顺序。"""
    left_state = torch.arange(7, dtype=torch.float32).repeat(2, 1)
    right_state = torch.arange(10, 17, dtype=torch.float32).repeat(2, 1)
    left_action = torch.arange(40, 47, dtype=torch.float32).repeat(2, 3, 1)
    right_action = torch.arange(50, 57, dtype=torch.float32).repeat(2, 3, 1)
    batch = {
        STATE_KEYS[0]: left_state,
        STATE_KEYS[1]: right_state,
        STATE_KEYS[2]: torch.full((2, 1), 20.0),
        STATE_KEYS[3]: torch.full((2, 1), 30.0),
        ACTION_KEYS[0]: left_action,
        ACTION_KEYS[1]: right_action,
        ACTION_KEYS[2]: torch.full((2, 3, 1), 60.0),
        ACTION_KEYS[3]: torch.full((2, 3, 1), 70.0),
    }

    adapted = adapt_pi05_batch(batch)

    assert adapted[OBS_STATE].shape == (2, 16)
    assert adapted[ACTION].shape == (2, 3, 16)
    torch.testing.assert_close(
        adapted[OBS_STATE][0],
        torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0, 30.0]
        ),
    )
    torch.testing.assert_close(
        adapted[ACTION][0, 0],
        torch.tensor(
            [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 60.0, 70.0]
        ),
    )


def test_adapt_pi05_batch_restores_squeezed_gripper_dimension():
    """验证 DataLoader 压缩夹爪末维后，适配器会补回该维度再拼接。"""
    batch = {
        STATE_KEYS[0]: torch.full((2, 7), 1.0),
        STATE_KEYS[1]: torch.full((2, 7), 3.0),
        STATE_KEYS[2]: torch.full((2,), 2.0),
        STATE_KEYS[3]: torch.full((2,), 4.0),
        ACTION_KEYS[0]: torch.full((2, 3, 7), 5.0),
        ACTION_KEYS[1]: torch.full((2, 3, 7), 7.0),
        ACTION_KEYS[2]: torch.full((2, 3), 6.0),
        ACTION_KEYS[3]: torch.full((2, 3), 8.0),
    }

    adapted = adapt_pi05_batch(batch)

    assert adapted[OBS_STATE].shape == (2, 16)
    assert adapted[ACTION].shape == (2, 3, 16)
    torch.testing.assert_close(
        adapted[OBS_STATE][0],
        torch.tensor([1.0] * 7 + [3.0] * 7 + [2.0] + [4.0]),
    )
    torch.testing.assert_close(
        adapted[ACTION][0, 0],
        torch.tensor([5.0] * 7 + [7.0] * 7 + [6.0] + [8.0]),
    )


def test_adapt_pi05_batch_keeps_standard_fields_unchanged():
    """验证标准 observation.state/action 已存在时不会被兼容逻辑覆盖。"""
    state = torch.randn(2, 6)
    action = torch.randn(2, 3, 4)

    adapted = adapt_pi05_batch({OBS_STATE: state, ACTION: action})

    assert adapted[OBS_STATE] is state
    assert adapted[ACTION] is action


def test_adapt_pi05_stats_concatenates_vector_stats_and_keeps_count_scalar():
    """验证 stats 与 batch 都只拼接字段块，不重排单臂内部关节，count 保持单值。"""
    stats = {
        STATE_KEYS[0]: {
            "q01": torch.arange(7, dtype=torch.float32),
            "q99": torch.arange(100, 107, dtype=torch.float32),
            "count": torch.tensor([100]),
        },
        STATE_KEYS[1]: {
            "q01": torch.arange(10, 17, dtype=torch.float32),
            "q99": torch.arange(110, 117, dtype=torch.float32),
            "count": torch.tensor([100]),
        },
        STATE_KEYS[2]: {
            "q01": torch.tensor([20.0]),
            "q99": torch.tensor([120.0]),
            "count": torch.tensor([100]),
        },
        STATE_KEYS[3]: {
            "q01": torch.tensor([30.0]),
            "q99": torch.tensor([130.0]),
            "count": torch.tensor([100]),
        },
        ACTION_KEYS[0]: {
            "q01": torch.arange(40, 47, dtype=torch.float32),
            "q99": torch.arange(140, 147, dtype=torch.float32),
            "count": torch.tensor([100]),
        },
        ACTION_KEYS[1]: {
            "q01": torch.arange(50, 57, dtype=torch.float32),
            "q99": torch.arange(150, 157, dtype=torch.float32),
            "count": torch.tensor([100]),
        },
        ACTION_KEYS[2]: {
            "q01": torch.tensor([60.0]),
            "q99": torch.tensor([160.0]),
            "count": torch.tensor([100]),
        },
        ACTION_KEYS[3]: {
            "q01": torch.tensor([70.0]),
            "q99": torch.tensor([170.0]),
            "count": torch.tensor([100]),
        },
    }

    adapted = adapt_pi05_stats(stats)

    assert adapted[OBS_STATE]["q01"].shape == (16,)
    assert adapted[ACTION]["q99"].shape == (16,)
    torch.testing.assert_close(
        adapted[OBS_STATE]["q01"],
        torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0, 30.0]
        ),
    )
    torch.testing.assert_close(
        adapted[ACTION]["q99"],
        torch.tensor(
            [
                140.0,
                141.0,
                142.0,
                143.0,
                144.0,
                145.0,
                146.0,
                150.0,
                151.0,
                152.0,
                153.0,
                154.0,
                155.0,
                156.0,
                160.0,
                170.0,
            ]
        ),
    )
    torch.testing.assert_close(adapted[ACTION]["count"], torch.tensor([100]))
    assert OBS_STATE not in stats
    assert ACTION not in stats


def test_pi05_split_state_action_is_opt_in_and_uses_16_dimensional_features():
    """验证兼容模式默认关闭，显式开启后模型特征维度被设置为 16。"""
    standard_config = PI05Config(device="cpu")
    assert standard_config.use_split_state_action is False

    split_config = PI05Config(device="cpu", use_split_state_action=True)
    split_config.input_features = {
        key: PolicyFeature(type=FeatureType.STATE, shape=(width,))
        for key, width in zip(STATE_KEYS, (7, 7, 1, 1), strict=True)
    }
    split_config.output_features = {
        key: PolicyFeature(type=FeatureType.ACTION, shape=(width,))
        for key, width in zip(ACTION_KEYS, (7, 7, 1, 1), strict=True)
    }

    split_config.validate_features()

    assert split_config.input_features[OBS_STATE].shape == (16,)
    assert split_config.output_features[ACTION].shape == (16,)


def test_split_action_fields_receive_the_pi05_action_horizon():
    """验证四个拆分动作字段都能读取 PI0.5 所需的未来动作序列。"""
    config = PI05Config(device="cpu", chunk_size=3, n_action_steps=3, use_split_state_action=True)
    dataset_metadata = SimpleNamespace(features={key: {} for key in ACTION_KEYS}, fps=10)

    delta_timestamps = resolve_delta_timestamps(config, dataset_metadata)

    expected_timestamps = [0.0, 0.1, 0.2]
    assert delta_timestamps == dict.fromkeys(ACTION_KEYS, expected_timestamps)
