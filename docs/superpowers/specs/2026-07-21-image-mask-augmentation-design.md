# 图片 Mask 增强设计

## 背景

当前图片增强发生在数据集读取阶段，而不是模型内部。训练配置通过 `make_dataset(cfg)` 创建 `ImageTransforms`，再传入 `LeRobotDataset` 或 `StreamingLeRobotDataset`。当 `DataLoader` 调用 `dataset.__getitem__(idx)` 时，数据集读取样本、解码摄像头画面，并遍历 `meta.camera_keys` 对每路 camera tensor 执行 `image_transforms`。

现有增强位于 `src/lerobot/datasets/transforms.py`，支持 `ColorJitter`、`SharpnessJitter`、`RandomAffine` 和 `Identity`。当前没有 Mask、Cutout、RandomErasing 或整路摄像头 dropout 类增强。

目标是在不改变模型内部逻辑的前提下，增加两类 Mask 增强：

- 局部区域 Mask：对单张摄像头图片随机擦除一个矩形区域。
- 整路摄像头 Mask：在一个训练样本中随机将某一路或某几路摄像头图像整体置为 Mask。

两类功能都必须默认关闭，通过训练配置显式开启，并提供中文参数说明。

## 设计原则

- 保守默认值：新增功能默认不开启，避免影响现有训练收敛和回归测试。
- 复用现有管线：局部区域 Mask 复用当前 `ImageTransforms` 的单图增强机制。
- 分层实现：整路摄像头 Mask 需要看到所有 camera key，因此放在 dataset 层，而不是伪装成单图 transform。
- 可配置：`p`、`scale`、`ratio`、`value` 等参数均可从配置传入。
- 可测试：局部 Mask 和整路摄像头 Mask 分别单元测试，数据集调用路径增加覆盖。

## 推荐方案

采用方案 B：

1. 在 `src/lerobot/datasets/transforms.py` 中扩展 `make_transform_from_config()`，支持 `torchvision.transforms.v2.RandomErasing`。
2. 在 `ImageTransformsConfig.tfs` 的默认配置里预留 `random_erasing`，但默认 `weight=0.0`，因此不会参与随机采样。
3. 新增整路摄像头 Mask 配置 `CameraDropoutConfig`，挂在 `DatasetConfig.camera_dropout`，使其可以通过 `--dataset.camera_dropout.xxx=...` 形式从训练命令控制。
4. 在 `LeRobotDataset.__getitem__()` 和 `StreamingLeRobotDataset` 的读取路径中，完成所有 camera image 加载和 per-image transforms 后，再对所有 camera keys 执行 camera dropout。

## 局部区域 Mask

局部区域 Mask 使用 `torchvision.transforms.v2.RandomErasing`。该增强会随机选择图片中的一个矩形区域，并将区域像素替换为指定值或随机值。它适合模拟手、物体、环境遮挡，也能降低模型对单一局部视觉特征的依赖。

配置示例：

```yaml
dataset:
  image_transforms:
    enable: true
    max_num_transforms: 3
    tfs:
      random_erasing:
        weight: 1.0
        type: RandomErasing
        kwargs:
          p: 0.15
          scale: [0.02, 0.08]
          ratio: [0.3, 3.3]
          value: 0.0
```

参数说明：

- `p`：对单张图片执行局部 Mask 的概率。
- `scale`：Mask 区域面积占整张图片面积的比例范围。建议从 `[0.02, 0.08]` 或 `[0.02, 0.12]` 开始。
- `ratio`：Mask 矩形的宽高比范围。默认可沿用 `RandomErasing` 的 `[0.3, 3.3]`。
- `value`：Mask 区域填充值。图片在当前 dataset 阶段通常为 `[0, 1]`，`0.0` 表示黑色区域；也可配置为 `"random"` 使用随机像素值。

局部区域 Mask 由现有 `RandomSubsetApply` 采样，因此每个 batch、每个 sample、每路 camera 的 Mask 位置都可以不同。

训练命令示例：

```bash
--dataset.image_transforms.enable=true \
--dataset.image_transforms.max_num_transforms=3 \
--dataset.image_transforms.random_order=false \
--dataset.image_transforms.tfs.random_erasing.weight=1.0 \
--dataset.image_transforms.tfs.random_erasing.type=RandomErasing \
--dataset.image_transforms.tfs.random_erasing.kwargs.p=0.15 \
--dataset.image_transforms.tfs.random_erasing.kwargs.scale='[0.02,0.08]' \
--dataset.image_transforms.tfs.random_erasing.kwargs.ratio='[0.3,3.3]' \
--dataset.image_transforms.tfs.random_erasing.kwargs.value=0.0
```

## 整路摄像头 Mask

整路摄像头 Mask 用于模拟某一路摄像头失效、被遮挡或画面不可用。它不能放在当前 `ImageTransforms` 内，因为 `ImageTransforms` 每次只接收单路图片，不知道其它摄像头是否已经被 Mask，也无法保证最少保留几路摄像头。

新增配置建议：

```yaml
dataset:
  camera_dropout:
    enable: false
    p: 0.05
    max_num_cameras: 1
    min_num_cameras_to_keep: 3
    value: 0.0
    eligible_camera_keys:
      - observation.images.head_stereo_left
      - observation.images.head_stereo_right
      - observation.images.wrist_left
      - observation.images.wrist_right
```

参数说明：

- `enable`：是否开启整路摄像头 Mask。默认 `false`。
- `p`：对一个训练样本执行整路摄像头 Mask 的概率。
- `max_num_cameras`：一次最多 Mask 掉几路摄像头。保守默认建议为 `1`。
- `min_num_cameras_to_keep`：每个样本至少保留几路摄像头。四路摄像头任务建议默认 `3`。
- `value`：整路摄像头被 Mask 后的填充值。当前阶段建议默认 `0.0`。
- `eligible_camera_keys`：允许被整路 Mask 的摄像头列表。未列入的摄像头永远不会被该增强 Mask。

默认配置中，整路摄像头 Mask 不开启。开启后，每个样本独立随机决策，batch 内不同样本可以 Mask 不同摄像头。采样时必须满足 `min_num_cameras_to_keep`，避免训练样本丢失过多视觉信息。

训练命令示例：

```bash
--dataset.camera_dropout.enable=true \
--dataset.camera_dropout.p=0.05 \
--dataset.camera_dropout.max_num_cameras=1 \
--dataset.camera_dropout.min_num_cameras_to_keep=3 \
--dataset.camera_dropout.value=0.0 \
--dataset.camera_dropout.eligible_camera_keys='["observation.images.head_stereo_left","observation.images.head_stereo_right","observation.images.wrist_left","observation.images.wrist_right"]'
```

完整训练命令可以只增加需要开启的 Mask 参数。两类 Mask 默认都关闭，因此不添加这些参数时，现有训练命令行为保持不变。

## 数据流

```text
train()
  ↓
make_dataset(cfg)
  ↓
创建 ImageTransforms；读取 camera_dropout 配置
  ↓
创建 LeRobotDataset / StreamingLeRobotDataset
  ↓
DataLoader 调用 dataset.__getitem__(idx)
  ↓
读取样本并解码摄像头画面
  ↓
遍历 meta.camera_keys，执行 ImageTransforms
  ↓
如果开启 camera_dropout，对样本级 camera dict 执行整路 Mask
  ↓
DataLoader collate 成 batch
  ↓
PI0.5 adapter / preprocessor
  ↓
模型训练
```

局部区域 Mask 在 per-image transform 阶段发生。整路摄像头 Mask 在所有 camera image 都已经加载并完成 per-image transform 后发生。

## 错误处理和边界

- 如果 `RandomErasing` 参数非法，沿用 Torchvision 的参数校验错误。
- 如果 `camera_dropout.enable=true` 但 `eligible_camera_keys` 为空，则默认使用当前 dataset 的 `meta.camera_keys`。
- 如果 `max_num_cameras <= 0`，则不执行整路 Mask。
- 如果 `min_num_cameras_to_keep` 大于可参与摄像头数量，则不执行整路 Mask，并保持样本不变。
- 如果样本中缺少某个 eligible camera key，则仅在实际存在的 camera key 中采样。
- 整路 Mask 不修改 PI0.5 的 `img_masks`，因为该增强模拟的是“图像内容不可用/被遮挡”，不是“摄像头特征缺失”。真正缺失的摄像头仍由 PI0.5 现有逻辑生成 `img_masks=False`。

## 测试计划

- `tests/datasets/test_image_transforms.py` 增加 `RandomErasing` 构造测试。
- 测试 `weight=0.0` 时 `random_erasing` 不参与采样，确保默认行为不变。
- 测试固定随机种子下，局部区域 Mask 会改变图片但保持 shape。
- 为整路摄像头 Mask 增加单元测试：默认关闭时不改图片。
- 测试 `p=1.0`、`max_num_cameras=1`、`min_num_cameras_to_keep=3` 时，四路输入中恰好一路被整体 Mask。
- 测试 `eligible_camera_keys` 只限制指定摄像头，不影响未列入摄像头。
- 测试 `min_num_cameras_to_keep` 约束生效，避免过度 Mask。
- 覆盖 `LeRobotDataset` 和 `StreamingLeRobotDataset` 的调用位置，至少通过轻量 helper 或 mock 样本验证 camera dropout 被调用。

## 文档计划

在 `docs/source/lerobot-dataset-v3.mdx` 的 Image transforms 章节补充中文说明，至少包括：

- 局部区域 Mask 与整路摄像头 Mask 的区别。
- 两类 Mask 默认关闭。
- 推荐保守起步参数。
- `p`、`scale`、`ratio`、`value`、`max_num_cameras`、`min_num_cameras_to_keep`、`eligible_camera_keys` 的含义。
- 说明整路 Mask 不等同于 PI0.5 的 `img_masks=False`。

## 参考经验

- Torchvision `RandomErasing` 是成熟的局部遮挡增强，支持 `p`、`scale`、`ratio`、`value` 参数。
- Random Erasing / Cutout 类方法通常用于提升对遮挡的鲁棒性，但过强的擦除可能抹掉任务关键区域，因此机器人操作任务应从低概率、小面积开始。
- 多摄像头机器人训练中，整路 camera dropout 更像传感器或视角缺失模拟，应比局部 Mask 更保守，默认最多 Mask 一路，并至少保留大部分摄像头。

参考资料：

- https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomErasing.html
- https://paperswithcode.com/paper/random-erasing-data-augmentation
- https://doi.org/10.1186/s40537-019-0197-0
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9966095/
