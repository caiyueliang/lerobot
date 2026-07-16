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
    "observation.state.left_gripper",
    "observation.state.right_arm",
    "observation.state.right_gripper",
)

ACTION_KEYS = (
    "action.left_arm",
    "action.left_gripper",
    "action.right_arm",
    "action.right_gripper",
)


def test_adapt_pi05_batch_concatenates_split_fields_in_robot_joint_order():
    """验证未压缩的拆分字段能够按左臂、左夹爪、右臂、右夹爪顺序拼接。"""
    batch = {
        STATE_KEYS[0]: torch.full((2, 7), 1.0),
        STATE_KEYS[1]: torch.full((2, 1), 2.0),
        STATE_KEYS[2]: torch.full((2, 7), 3.0),
        STATE_KEYS[3]: torch.full((2, 1), 4.0),
        ACTION_KEYS[0]: torch.full((2, 3, 7), 5.0),
        ACTION_KEYS[1]: torch.full((2, 3, 1), 6.0),
        ACTION_KEYS[2]: torch.full((2, 3, 7), 7.0),
        ACTION_KEYS[3]: torch.full((2, 3, 1), 8.0),
    }

    adapted = adapt_pi05_batch(batch)

    assert adapted[OBS_STATE].shape == (2, 16)
    assert adapted[ACTION].shape == (2, 3, 16)
    torch.testing.assert_close(
        adapted[OBS_STATE][0],
        torch.tensor([1.0] * 7 + [2.0] + [3.0] * 7 + [4.0]),
    )
    torch.testing.assert_close(
        adapted[ACTION][0, 0],
        torch.tensor([5.0] * 7 + [6.0] + [7.0] * 7 + [8.0]),
    )


def test_adapt_pi05_batch_restores_squeezed_gripper_dimension():
    """验证 DataLoader 压缩夹爪末维后，适配器会补回该维度再拼接。"""
    batch = {
        STATE_KEYS[0]: torch.full((2, 7), 1.0),
        STATE_KEYS[1]: torch.full((2,), 2.0),
        STATE_KEYS[2]: torch.full((2, 7), 3.0),
        STATE_KEYS[3]: torch.full((2,), 4.0),
        ACTION_KEYS[0]: torch.full((2, 3, 7), 5.0),
        ACTION_KEYS[1]: torch.full((2, 3), 6.0),
        ACTION_KEYS[2]: torch.full((2, 3, 7), 7.0),
        ACTION_KEYS[3]: torch.full((2, 3), 8.0),
    }

    adapted = adapt_pi05_batch(batch)

    assert adapted[OBS_STATE].shape == (2, 16)
    assert adapted[ACTION].shape == (2, 3, 16)
    torch.testing.assert_close(
        adapted[OBS_STATE][0],
        torch.tensor([1.0] * 7 + [2.0] + [3.0] * 7 + [4.0]),
    )
    torch.testing.assert_close(
        adapted[ACTION][0, 0],
        torch.tensor([5.0] * 7 + [6.0] + [7.0] * 7 + [8.0]),
    )


def test_adapt_pi05_batch_keeps_standard_fields_unchanged():
    """验证标准 observation.state/action 已存在时不会被兼容逻辑覆盖。"""
    state = torch.randn(2, 6)
    action = torch.randn(2, 3, 4)

    adapted = adapt_pi05_batch({OBS_STATE: state, ACTION: action})

    assert adapted[OBS_STATE] is state
    assert adapted[ACTION] is action


def test_adapt_pi05_stats_concatenates_vector_stats_and_keeps_count_scalar():
    """验证逐维统计量会拼接，而表示样本数的 count 仍保持单值。"""
    stats = {}
    for index, (state_key, action_key, width) in enumerate(
        zip(STATE_KEYS, ACTION_KEYS, (7, 1, 7, 1), strict=True),
        start=1,
    ):
        stats[state_key] = {
            "q01": torch.full((width,), float(index)),
            "q99": torch.full((width,), float(index + 10)),
            "count": torch.tensor([100]),
        }
        stats[action_key] = {
            "q01": torch.full((width,), float(index + 20)),
            "q99": torch.full((width,), float(index + 30)),
            "count": torch.tensor([100]),
        }

    adapted = adapt_pi05_stats(stats)

    assert adapted[OBS_STATE]["q01"].shape == (16,)
    assert adapted[ACTION]["q99"].shape == (16,)
    torch.testing.assert_close(
        adapted[OBS_STATE]["q01"],
        torch.tensor([1.0] * 7 + [2.0] + [3.0] * 7 + [4.0]),
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
        for key, width in zip(STATE_KEYS, (7, 1, 7, 1), strict=True)
    }
    split_config.output_features = {
        key: PolicyFeature(type=FeatureType.ACTION, shape=(width,))
        for key, width in zip(ACTION_KEYS, (7, 1, 7, 1), strict=True)
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
