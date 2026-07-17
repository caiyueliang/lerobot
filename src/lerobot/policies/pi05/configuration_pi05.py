#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.pi05.dataset_adapter import PI05_SPLIT_ACTION_KEYS, PI05_SPLIT_STATE_KEYS
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    # 数据集工厂通过该类变量识别需要按 action horizon 读取的拆分动作字段。
    # ClassVar 不属于 dataclass 的命令行配置项，因此用户只需要控制下面的布尔开关。
    split_action_keys: ClassVar[tuple[str, ...]] = PI05_SPLIT_ACTION_KEYS

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # 是否启用 G1 Dex1 拆分字段兼容模式，默认关闭以完整保留原来的训练方式。
    # 命令行传入 ``--policy.use_split_state_action=true`` 后，训练流程才会把左右手臂、
    # 左右夹爪拼成标准的 ``observation.state`` 和 ``action``。
    use_split_state_action: bool = False

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if self.use_split_state_action:
            # 开关打开时，先验证数据集元信息中是否同时存在四个状态字段和四个动作字段。
            # 这里提前失败可以避免训练到第一个 batch 时才发现字段缺失。
            missing_state_keys = [key for key in PI05_SPLIT_STATE_KEYS if key not in self.input_features]
            missing_action_keys = [key for key in PI05_SPLIT_ACTION_KEYS if key not in self.output_features]
            if missing_state_keys or missing_action_keys:
                raise ValueError(
                    "PI05 split state/action compatibility is enabled, but source features are missing. "
                    f"Missing state features: {missing_state_keys}; missing action features: {missing_action_keys}"
                )

            # 根据数据集元信息动态计算拼接后的真实维度。当前 G1 Dex1 为
            # 7 + 7 + 1 + 1 = 16 维，但不在代码中硬编码 16，便于发现元信息变化。
            state_dim = sum(self.input_features[key].shape[-1] for key in PI05_SPLIT_STATE_KEYS)
            action_dim = sum(self.output_features[key].shape[-1] for key in PI05_SPLIT_ACTION_KEYS)
            # PI0.5 内部会把较短向量补零到 max_state_dim/max_action_dim，但不能接受
            # 超过最大维度的输入，因此在创建模型前进行明确检查。
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"Concatenated state dimension ({state_dim}) exceeds max_state_dim ({self.max_state_dim})"
                )
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Concatenated action dimension ({action_dim}) exceeds max_action_dim ({self.max_action_dim})"
                )

            # 把动态计算出的标准字段定义写入 policy config。模型据此知道真实输出只有
            # 16 维，内部补零到 32 维后，推理时仍会正确裁回真实动作维度。
            self.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
            self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
