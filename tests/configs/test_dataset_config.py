import logging

import draccus

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import log_image_augmentation_config


def test_dataset_config_parses_mask_augmentation_cli_flags():
    cfg = draccus.parse(
        DatasetConfig,
        args=[
            "--repo_id=/tmp/example",
            "--image_transforms.enable=true",
            "--image_transforms.random_erasing.enable=true",
            "--image_transforms.random_erasing.weight=1.0",
            "--image_transforms.random_erasing.p=0.15",
            "--image_transforms.random_erasing.scale=[0.02,0.08]",
            "--image_transforms.random_erasing.ratio=[0.3,3.3]",
            "--image_transforms.random_erasing.value=0.0",
            "--camera_dropout.enable=true",
            "--camera_dropout.p=0.05",
            "--camera_dropout.max_num_cameras=1",
            "--camera_dropout.min_num_cameras_to_keep=3",
            "--camera_dropout.value=0.0",
            "--camera_dropout.eligible_camera_keys=[observation.images.head_stereo_left,observation.images.wrist_left]",
        ],
    )

    assert cfg.image_transforms.enable is True
    assert cfg.image_transforms.random_erasing.enable is True
    assert cfg.image_transforms.random_erasing.weight == 1.0
    assert cfg.image_transforms.random_erasing.p == 0.15
    assert cfg.image_transforms.random_erasing.scale == (0.02, 0.08)
    assert cfg.camera_dropout.enable is True
    assert cfg.camera_dropout.p == 0.05
    assert cfg.camera_dropout.max_num_cameras == 1
    assert cfg.camera_dropout.min_num_cameras_to_keep == 3
    assert cfg.camera_dropout.value == 0.0
    assert cfg.camera_dropout.eligible_camera_keys == [
        "observation.images.head_stereo_left",
        "observation.images.wrist_left",
    ]


def test_dataset_config_mask_defaults_are_disabled():
    cfg = draccus.parse(DatasetConfig, args=["--repo_id=/tmp/example"])

    assert cfg.camera_dropout.enable is False
    assert cfg.image_transforms.random_erasing.enable is False
    assert "random_erasing" not in cfg.image_transforms.tfs


def test_log_image_augmentation_config_reports_mask_parameters(caplog):
    caplog.set_level(logging.INFO)
    cfg = draccus.parse(
        DatasetConfig,
        args=[
            "--repo_id=/tmp/example",
            "--image_transforms.enable=true",
            "--image_transforms.max_num_transforms=2",
            "--image_transforms.random_order=true",
            "--image_transforms.random_erasing.enable=true",
            "--image_transforms.random_erasing.weight=0.5",
            "--image_transforms.random_erasing.p=0.15",
            "--image_transforms.random_erasing.scale=[0.02,0.08]",
            "--image_transforms.random_erasing.ratio=[0.3,3.3]",
            "--image_transforms.random_erasing.value=0.0",
            "--camera_dropout.enable=true",
            "--camera_dropout.p=0.05",
            "--camera_dropout.max_num_cameras=1",
            "--camera_dropout.min_num_cameras_to_keep=3",
            "--camera_dropout.value=0.0",
            "--camera_dropout.eligible_camera_keys=[observation.images.head_stereo_left,observation.images.wrist_left]",
        ],
    )

    log_image_augmentation_config(cfg)

    log_text = caplog.text
    assert "Image transforms: enable=True, max_num_transforms=2, random_order=True" in log_text
    assert "Random erasing: enable=True, weight=0.5, p=0.15, scale=(0.02, 0.08)" in log_text
    assert "ratio=(0.3, 3.3), value=0.0" in log_text
    assert "Camera dropout: enable=True, p=0.05, max_num_cameras=1, min_num_cameras_to_keep=3" in log_text
    assert "eligible_camera_keys=['observation.images.head_stereo_left', 'observation.images.wrist_left']" in log_text
