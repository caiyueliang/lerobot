#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RandomErasingConfig:
    """
    局部区域 Mask 的命令行友好配置。

    这里不把 RandomErasing 放进默认的 `tfs` 字典里，原因是 `tfs` 主要服务于已有的
    ColorJitter/SharpnessJitter 等“参数都是范围二元组”的增强；可视化脚本也会按这个假设
    遍历 `tfs`。RandomErasing 同时包含概率、面积范围、宽高比和值，单独建 dataclass
    可以让训练命令使用 `--dataset.image_transforms.random_erasing.xxx=...` 这种稳定路径。
    """

    # 总开关。默认关闭，避免现有训练命令在未显式指定时改变数据分布。
    enable: bool = False
    # 参与 RandomSubsetApply 采样时的权重；只有 enable=True 且 weight>0 时才会加入增强池。
    weight: float = 1.0
    # 对单张图片实际执行擦除的概率。即使该增强被采样到，RandomErasing 内部也会再按 p 判定。
    p: float = 0.1
    # 被擦除矩形区域占整张图片面积的比例范围。默认较保守，降低遮掉关键目标的概率。
    scale: tuple[float, float] = (0.02, 0.08)
    # 被擦除矩形的宽高比范围，沿用 torchvision RandomErasing 的常见默认范围。
    ratio: tuple[float, float] = (0.3, 3.3)
    # 被擦除区域的填充值。图片在 dataset 阶段通常是 [0, 1]，0.0 表示黑块，"random" 表示随机值。
    value: float | str = 0.0


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    # 局部区域 Mask 的显式配置入口。它默认关闭，但可以通过命令行单独控制 p/scale/ratio/value。
    random_erasing: RandomErasingConfig = field(default_factory=RandomErasingConfig)
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )


@dataclass
class CameraDropoutConfig:
    """
    整路摄像头 Mask 的配置。

    这个增强必须在 dataset 层做，而不是放进 `ImageTransforms`，因为单图 transform 每次只能
    看到一路 camera tensor，无法知道同一个样本里还有哪些摄像头，也就无法保证“至少保留几路”
    或“最多 Mask 几路”。dataset 层已经拿到了 `meta.camera_keys` 和样本中的所有图像，适合
    在每个样本上统一随机选择整路摄像头进行 Mask。
    """

    # 总开关。默认关闭，保证旧训练命令完全不受影响。
    enable: bool = False
    # 对一个训练样本执行整路摄像头 Mask 的概率。
    p: float = 0.05
    # 单个样本里最多 Mask 掉几路摄像头。四路相机任务默认建议保持为 1。
    max_num_cameras: int = 1
    # 单个样本至少保留几路摄像头，防止一次 Mask 太多导致训练信号过弱。
    min_num_cameras_to_keep: int = 3
    # 整路摄像头被 Mask 后的填充值；支持数字和 "random"。
    value: float | str = 0.0
    # 允许被 Mask 的摄像头 key。为 None 时表示当前样本中所有 camera key 都有资格参与随机选择。
    eligible_camera_keys: list[str] | None = None


def _make_camera_dropout_value(image: torch.Tensor, value: float | str) -> torch.Tensor:
    """根据配置生成和原图同形状、同 dtype/device 的整路 Mask 图像。"""
    if value == "random":
        return torch.rand_like(image)
    if isinstance(value, (int, float)):
        return torch.full_like(image, float(value))
    raise ValueError(f"Camera dropout value must be a number or 'random', got {value!r}.")


def apply_camera_dropout(
    item: dict[str, Any],
    camera_keys: Sequence[str],
    cfg: CameraDropoutConfig,
) -> dict[str, Any]:
    """
    对一个 dataset 样本执行整路摄像头 Mask。

    返回值保持 LeRobotDataset item 的字典结构不变，只替换被选中 camera key 对应的 tensor。
    注意这里不修改 PI0.5 后续生成的 `img_masks`：整路 camera dropout 模拟的是“图像内容被遮挡
    或不可用”，不是“这个摄像头字段从 batch 中消失”。如果字段真的缺失，PI0.5 现有预处理逻辑
    会自己把对应 `img_masks` 置为 false。
    """
    # 快速退出路径保持原 item 对象，避免默认关闭时引入额外拷贝成本。
    if not cfg.enable or cfg.p <= 0.0 or cfg.max_num_cameras <= 0:
        return item
    # 对每个样本独立抽样，因此同一个 batch 中不同样本可能 Mask 不同摄像头。
    if torch.rand(()) >= cfg.p:
        return item

    # 只在样本实际存在的 camera key 中选择；有些数据集或策略可能只提供部分相机。
    present_camera_keys = [key for key in camera_keys if key in item]
    eligible = list(cfg.eligible_camera_keys) if cfg.eligible_camera_keys else list(camera_keys)
    available_keys = [key for key in eligible if key in present_camera_keys]
    if not available_keys:
        return item

    # min_num_cameras_to_keep 按样本中“实际存在的全部 camera 数量”计算，而不是按 eligible 子集计算。
    # 这样可以支持“只允许 wrist_left 被 Mask，但仍要求四路相机至少保留三路”的用法。
    max_allowed = len(present_camera_keys) - cfg.min_num_cameras_to_keep
    num_to_mask = min(cfg.max_num_cameras, len(available_keys), max_allowed)
    if num_to_mask <= 0:
        return item

    # 使用 torch 随机数，和现有图像增强一样受 torch seed/seeded_context 控制。
    selected_indices = torch.randperm(len(available_keys))[:num_to_mask]
    selected_keys = [available_keys[i] for i in selected_indices.tolist()]

    # 只在真正需要替换图片时浅拷贝字典，未被 Mask 的字段继续复用原对象。
    output = dict(item)
    for key in selected_keys:
        image = output[key]
        # 防御式处理：camera key 正常应为 Tensor；如果调用方传入非 Tensor，则跳过该 key。
        if not isinstance(image, torch.Tensor):
            continue
        output[key] = _make_camera_dropout_value(image, cfg.value)

    return output


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    elif cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    elif cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    elif cfg.type == "RandomAffine":
        return v2.RandomAffine(**cfg.kwargs)
    elif cfg.type == "RandomErasing":
        return v2.RandomErasing(**cfg.kwargs)
    else:
        raise ValueError(f"Transform '{cfg.type}' is not valid.")


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        # RandomErasing 作为独立显式配置接入，方便 CLI 调参，也避免默认 `tfs` 遍历逻辑误处理
        # `p/value` 这类不是范围二元组的参数。
        if cfg.random_erasing.enable and cfg.random_erasing.weight > 0.0:
            self.transforms["random_erasing"] = v2.RandomErasing(
                p=cfg.random_erasing.p,
                scale=cfg.random_erasing.scale,
                ratio=cfg.random_erasing.ratio,
                value=cfg.random_erasing.value,
            )
            self.weights.append(cfg.random_erasing.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
