#!/usr/bin/env python

"""将拆分保存的机器人状态和动作适配为 PI0.5 所需的标准字段格式。

标准 LeRobot 数据集通常直接提供 ``observation.state`` 和 ``action``。G1 Dex1
数据集则把左右手臂、左右夹爪分别保存在多个字段中，因此需要在送入 PI0.5 前，
按照固定顺序把这些字段拼接起来。同时，归一化所依赖的统计量也必须使用完全相同的
顺序拼接，否则每个维度会使用错误的统计量。
"""

from copy import deepcopy
from typing import Any

import torch

from lerobot.utils.constants import ACTION, OBS_STATE

# 字段块顺序必须与标准 16 维格式一致：左臂 7 维 + 右臂 7 维 + 左夹爪 1 维
# + 右夹爪 1 维。最终生成的 ``observation.state`` 可以直接替代标准数据集中的同名字段。
PI05_SPLIT_STATE_KEYS = (
    "observation.state.left_arm",
    "observation.state.right_arm",
    "observation.state.left_gripper",
    "observation.state.right_gripper",
)

# 动作字段使用与状态完全相同的身体部位顺序，保证 state/action 的每个维度语义对齐。
PI05_SPLIT_ACTION_KEYS = (
    "action.left_arm",
    "action.right_arm",
    "action.left_gripper",
    "action.right_gripper",
)


def _concatenate_batch_fields(batch: dict[str, Any], keys: tuple[str, ...], target_key: str) -> None:
    """把一个 batch 中的多个拆分字段沿最后一维拼接为标准字段。

    这里同时兼容两种夹爪张量格式：

    - 状态：手臂通常是 ``[B, 7]``，夹爪可能是 ``[B]`` 或 ``[B, 1]``；
    - 动作：手臂通常是 ``[B, T, 7]``，夹爪可能是 ``[B, T]`` 或 ``[B, T, 1]``。

    当夹爪的最后一个 1 维被 DataLoader 压缩掉时，会先自动补回该维度，再进行拼接。
    ``B`` 表示 batch 大小，``T`` 表示 PI0.5 的动作预测时间长度。
    """

    # 在真正拼接前一次性检查字段是否齐全，以便报错时直接指出缺失字段，而不是让
    # ``torch.cat`` 给出难以定位的数据形状错误。
    missing_keys = [key for key in keys if key not in batch]
    if missing_keys:
        raise KeyError(f"Cannot build '{target_key}'. Missing dataset fields: {missing_keys}")

    # Dataset/DataLoader 返回值可能是 Tensor、NumPy 数组或 Python 数值，因此统一转为
    # Tensor。这里只调整字段块顺序，不改变左臂和右臂各自内部的 7 个关节顺序。
    values = [torch.as_tensor(batch[key]) for key in keys]

    # 以手臂字段的维数作为正常维数。夹爪字段只有一个数，经过数据加载后可能比手臂
    # 少最后一维，例如 ``[B]`` 对 ``[B, 7]``、``[B, T]`` 对 ``[B, T, 7]``。
    max_ndim = max(value.ndim for value in values)
    normalized_values = []
    for key, value in zip(keys, values, strict=True):
        if value.ndim == max_ndim - 1:
            # 只在末尾补维度，不改变 batch 维和时间维：
            # ``[B] -> [B, 1]``，``[B, T] -> [B, T, 1]``。
            value = value.unsqueeze(-1)
        elif value.ndim != max_ndim:
            # 如果相差不止一维，说明数据并非预期的“标量夹爪字段被压缩”情况。
            # 此时停止训练并打印所有源字段形状，避免静默拼出错误数据。
            shapes = {
                source_key: tuple(source_value.shape)
                for source_key, source_value in zip(keys, values, strict=True)
            }
            raise ValueError(
                f"Cannot build '{target_key}' from incompatible field shapes: {shapes}. "
                f"Field '{key}' has {value.ndim} dimensions, expected {max_ndim} or {max_ndim - 1}."
            )
        normalized_values.append(value)

    # ``dim=-1`` 只合并特征维，batch 维和动作时间维保持不变。
    batch[target_key] = torch.cat(normalized_values, dim=-1)


def adapt_pi05_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """返回包含 PI0.5 标准状态和动作字段的新 batch。

    函数只做浅拷贝，不会向原始 batch 字典写入新字段。如果数据本身已经包含
    ``observation.state`` 或 ``action``，则原值优先，不重复拼接。这样既兼容 G1 Dex1
    拆分格式，也不会改变标准 LeRobot 数据集的原有训练行为。
    """

    adapted_batch = dict(batch)
    # 仅在标准状态字段不存在时，按固定顺序生成 16 维状态。
    if OBS_STATE not in adapted_batch:
        _concatenate_batch_fields(adapted_batch, PI05_SPLIT_STATE_KEYS, OBS_STATE)
    # 仅在标准动作字段不存在时，按固定顺序生成 16 维动作序列。
    if ACTION not in adapted_batch:
        _concatenate_batch_fields(adapted_batch, PI05_SPLIT_ACTION_KEYS, ACTION)
    return adapted_batch


def _concatenate_feature_stats(
    stats: dict[str, dict[str, Any]], source_keys: tuple[str, ...], target_key: str
) -> dict[str, Any]:
    """按照 batch 的字段顺序，拼接一个标准字段所需的归一化统计量。

    PI0.5 默认用 q01/q99 分位数归一化 state 和 action。均值、标准差、最小值、最大值
    等其他逐维统计量也采用相同规则拼接，便于切换归一化方式时继续使用。
    """

    # stats.json 中每一个参与拼接的源字段都必须存在，否则无法保证目标字段的每个
    # 维度都有对应统计量。
    missing_keys = [key for key in source_keys if key not in stats]
    if missing_keys:
        raise KeyError(f"Cannot build stats for '{target_key}'. Missing stats fields: {missing_keys}")

    # 只处理所有源字段共同拥有的统计项，防止某个字段缺少可选统计项时直接 KeyError。
    common_stat_names = set(stats[source_keys[0]])
    for key in source_keys[1:]:
        common_stat_names.intersection_update(stats[key])

    concatenated_stats: dict[str, Any] = {}
    for stat_name in common_stat_names:
        values = [torch.as_tensor(stats[key][stat_name]) for key in source_keys]
        if stat_name == "count":
            # count 表示样本总数，各字段来自同一批帧，数值相同，因此保留一份即可；
            # 如果把四个 count 拼起来，会错误地产生一个 4 维 count。
            concatenated_stats[stat_name] = values[0].clone()
        else:
            # q01/q99/mean/std/min/max 等都是逐特征维统计量。与 batch 一样，只按照
            # 左臂、右臂、左夹爪、右夹爪的字段块顺序拼接，不改变单臂内部关节顺序。
            concatenated_stats[stat_name] = torch.cat([value.reshape(-1) for value in values])
    return concatenated_stats


def adapt_pi05_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """补充 PI0.5 标准字段的统计量，同时保持原始数据集元数据不变。

    只有启用拆分字段兼容参数时，训练入口才会调用本函数。已有标准统计量时直接保留，
    因此该逻辑不会覆盖标准数据集自己的 ``observation.state`` 或 ``action`` 统计信息。
    """

    # 深拷贝可避免把临时生成的 PI0.5 统计量写回 dataset.meta.stats。
    adapted_stats = deepcopy(stats)
    if OBS_STATE not in adapted_stats:
        adapted_stats[OBS_STATE] = _concatenate_feature_stats(adapted_stats, PI05_SPLIT_STATE_KEYS, OBS_STATE)
    if ACTION not in adapted_stats:
        adapted_stats[ACTION] = _concatenate_feature_stats(adapted_stats, PI05_SPLIT_ACTION_KEYS, ACTION)
    return adapted_stats
