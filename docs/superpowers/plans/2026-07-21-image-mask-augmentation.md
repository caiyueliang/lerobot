# Image Mask Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in local image mask augmentation and opt-in whole-camera mask augmentation for LeRobot training datasets.

**Architecture:** Local region masking extends the existing per-image `ImageTransforms` pipeline with `torchvision.transforms.v2.RandomErasing`. Whole-camera masking is implemented as a dataset-level helper because it needs access to all camera keys in a sample before deciding which complete camera tensors to mask.

**Tech Stack:** Python dataclasses, PyTorch tensors, Torchvision transforms v2, draccus CLI parsing, pytest.

## Global Constraints

- Do not change model internals; masking happens during dataset reading.
- Both local region mask and whole-camera mask default to disabled.
- CLI control must use the existing nested `--dataset.xxx` style.
- Local region mask supports configurable `p`, `scale`, `ratio`, and `value`.
- Whole-camera mask supports configurable `enable`, `p`, `max_num_cameras`, `min_num_cameras_to_keep`, `value`, and `eligible_camera_keys`.
- Whole-camera mask must not modify PI0.5 `img_masks`; it only changes image tensor content.
- Existing training commands without new mask flags must behave the same.

---

## File Structure

- Modify `src/lerobot/datasets/transforms.py`
  - Add `RandomErasing` support to `make_transform_from_config()`.
  - Add a default disabled `random_erasing` entry with `weight=0.0`.
  - Define `CameraDropoutConfig` and `apply_camera_dropout()` near image transform utilities.
- Modify `src/lerobot/configs/default.py`
  - Add `camera_dropout: CameraDropoutConfig` to `DatasetConfig`.
- Modify `src/lerobot/datasets/factory.py`
  - Pass `cfg.dataset.camera_dropout` into regular and streaming datasets.
- Modify `src/lerobot/datasets/lerobot_dataset.py`
  - Accept and store `camera_dropout`.
  - Apply whole-camera dropout after per-image transforms in `__getitem__()`.
  - Preserve dataset copying/aggregation paths that carry dataset transform state.
- Modify `src/lerobot/datasets/streaming_dataset.py`
  - Accept and store `camera_dropout`.
  - Apply whole-camera dropout after per-image transforms in the streaming sample path.
- Modify `tests/datasets/test_image_transforms.py`
  - Add unit tests for `RandomErasing` and `apply_camera_dropout()`.
- Create `tests/configs/test_dataset_config.py`
  - Add draccus parse tests for representative CLI flags and disabled defaults.
- Modify `docs/source/lerobot-dataset-v3.mdx`
  - Add Chinese parameter documentation and CLI examples.

---

### Task 1: Add Local RandomErasing Transform Support

**Files:**
- Modify: `src/lerobot/datasets/transforms.py`
- Test: `tests/datasets/test_image_transforms.py`

**Interfaces:**
- Consumes: `ImageTransformConfig(type: str, kwargs: dict[str, Any])`
- Produces: `make_transform_from_config(cfg: ImageTransformConfig) -> Transform` supports `cfg.type == "RandomErasing"`.
- Produces: `ImageTransformsConfig.tfs["random_erasing"]` exists with `weight=0.0` and `type="RandomErasing"`.

- [ ] **Step 1: Write the failing RandomErasing construction test**

Add this import if missing:

```python
from lerobot.datasets.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig
```

Add this test to `tests/datasets/test_image_transforms.py` near other transform construction tests:

```python
def test_get_image_transforms_random_erasing(img_tensor_factory):
    img_tensor = img_tensor_factory()
    tf_cfg = ImageTransformsConfig(
        enable=True,
        tfs={
            "random_erasing": ImageTransformConfig(
                type="RandomErasing",
                kwargs={"p": 1.0, "scale": (0.2, 0.2), "ratio": (1.0, 1.0), "value": 0.0},
            )
        },
    )

    tf = ImageTransforms(tf_cfg)
    output = tf(img_tensor)

    assert output.shape == img_tensor.shape
    assert isinstance(tf.transforms["random_erasing"], v2.RandomErasing)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(output, img_tensor)
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
pytest tests/datasets/test_image_transforms.py::test_get_image_transforms_random_erasing -v
```

Expected: FAIL with `ValueError: Transform 'RandomErasing' is not valid.`

- [ ] **Step 3: Implement RandomErasing support**

In `src/lerobot/datasets/transforms.py`, update `ImageTransformsConfig.tfs` with this disabled default entry after `affine`:

```python
"random_erasing": ImageTransformConfig(
    weight=0.0,
    type="RandomErasing",
    kwargs={"p": 0.1, "scale": (0.02, 0.08), "ratio": (0.3, 3.3), "value": 0.0},
),
```

Update `make_transform_from_config()`:

```python
elif cfg.type == "RandomErasing":
    return v2.RandomErasing(**cfg.kwargs)
```

- [ ] **Step 4: Verify local transform tests pass**

Run:

```bash
pytest tests/datasets/test_image_transforms.py::test_get_image_transforms_random_erasing tests/datasets/test_image_transforms.py::test_get_image_transforms_no_transform_enable_false -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/datasets/transforms.py tests/datasets/test_image_transforms.py
git commit -m "feat: add random erasing image transform"
```

---

### Task 2: Add Whole-Camera Dropout Helper

**Files:**
- Modify: `src/lerobot/datasets/transforms.py`
- Test: `tests/datasets/test_image_transforms.py`

**Interfaces:**
- Produces: `CameraDropoutConfig(enable: bool = False, p: float = 0.05, max_num_cameras: int = 1, min_num_cameras_to_keep: int = 3, value: float | str = 0.0, eligible_camera_keys: list[str] | None = None)`
- Produces: `apply_camera_dropout(item: dict[str, Any], camera_keys: Sequence[str], cfg: CameraDropoutConfig) -> dict[str, Any]`
- Later tasks call `apply_camera_dropout(item, self.meta.camera_keys, self.camera_dropout)`.

- [ ] **Step 1: Write failing tests for disabled and enabled camera dropout**

Add these imports:

```python
from lerobot.datasets.transforms import CameraDropoutConfig, apply_camera_dropout
```

Add these tests:

```python
def test_apply_camera_dropout_disabled_keeps_images():
    item = {
        "observation.images.head_stereo_left": torch.ones(3, 8, 8),
        "observation.images.head_stereo_right": torch.ones(3, 8, 8) * 2,
    }
    original = {key: value.clone() for key, value in item.items()}

    output = apply_camera_dropout(
        item,
        list(item),
        CameraDropoutConfig(enable=False, p=1.0, max_num_cameras=1, min_num_cameras_to_keep=1),
    )

    assert output is item
    for key in original:
        torch.testing.assert_close(output[key], original[key])
```

```python
def test_apply_camera_dropout_masks_one_eligible_camera_with_keep_constraint():
    item = {
        "observation.images.head_stereo_left": torch.ones(3, 8, 8),
        "observation.images.head_stereo_right": torch.ones(3, 8, 8) * 2,
        "observation.images.wrist_left": torch.ones(3, 8, 8) * 3,
        "observation.images.wrist_right": torch.ones(3, 8, 8) * 4,
    }
    cfg = CameraDropoutConfig(
        enable=True,
        p=1.0,
        max_num_cameras=1,
        min_num_cameras_to_keep=3,
        value=0.0,
        eligible_camera_keys=list(item),
    )

    with seeded_context(1234):
        output = apply_camera_dropout(item, list(item), cfg)

    masked_keys = [key for key, image in output.items() if torch.count_nonzero(image) == 0]
    assert len(masked_keys) == 1
    assert masked_keys[0] in cfg.eligible_camera_keys
```

```python
def test_apply_camera_dropout_respects_eligible_camera_keys():
    item = {
        "observation.images.head_stereo_left": torch.ones(3, 8, 8),
        "observation.images.head_stereo_right": torch.ones(3, 8, 8) * 2,
        "observation.images.wrist_left": torch.ones(3, 8, 8) * 3,
        "observation.images.wrist_right": torch.ones(3, 8, 8) * 4,
    }
    cfg = CameraDropoutConfig(
        enable=True,
        p=1.0,
        max_num_cameras=1,
        min_num_cameras_to_keep=3,
        value=0.0,
        eligible_camera_keys=["observation.images.wrist_left"],
    )

    output = apply_camera_dropout(item, list(item), cfg)

    assert torch.count_nonzero(output["observation.images.wrist_left"]) == 0
    assert torch.count_nonzero(output["observation.images.head_stereo_left"]) > 0
    assert torch.count_nonzero(output["observation.images.head_stereo_right"]) > 0
    assert torch.count_nonzero(output["observation.images.wrist_right"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/datasets/test_image_transforms.py::test_apply_camera_dropout_disabled_keeps_images tests/datasets/test_image_transforms.py::test_apply_camera_dropout_masks_one_eligible_camera_with_keep_constraint tests/datasets/test_image_transforms.py::test_apply_camera_dropout_respects_eligible_camera_keys -v
```

Expected: FAIL with import error for `CameraDropoutConfig` or `apply_camera_dropout`.

- [ ] **Step 3: Implement CameraDropoutConfig and helper**

In `src/lerobot/datasets/transforms.py`, add `Sequence` is already imported. Add this dataclass after `ImageTransformsConfig`:

```python
@dataclass
class CameraDropoutConfig:
    """Configuration for masking complete camera image tensors during dataset reads."""

    enable: bool = False
    p: float = 0.05
    max_num_cameras: int = 1
    min_num_cameras_to_keep: int = 3
    value: float | str = 0.0
    eligible_camera_keys: list[str] | None = None
```

Add this helper below `CameraDropoutConfig`:

```python
def _make_camera_dropout_value(image: torch.Tensor, value: float | str) -> torch.Tensor:
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
    if not cfg.enable or cfg.p <= 0.0 or cfg.max_num_cameras <= 0:
        return item
    if torch.rand(()) >= cfg.p:
        return item

    eligible = list(cfg.eligible_camera_keys) if cfg.eligible_camera_keys else list(camera_keys)
    available_keys = [key for key in eligible if key in item and key in camera_keys]
    if not available_keys:
        return item

    max_allowed = len(available_keys) - cfg.min_num_cameras_to_keep
    num_to_mask = min(cfg.max_num_cameras, max_allowed)
    if num_to_mask <= 0:
        return item

    selected_indices = torch.randperm(len(available_keys))[:num_to_mask]
    selected_keys = [available_keys[i] for i in selected_indices.tolist()]

    output = dict(item)
    for key in selected_keys:
        image = output[key]
        if not isinstance(image, torch.Tensor):
            continue
        output[key] = _make_camera_dropout_value(image, cfg.value)

    return output
```

- [ ] **Step 4: Verify helper tests pass**

Run:

```bash
pytest tests/datasets/test_image_transforms.py::test_apply_camera_dropout_disabled_keeps_images tests/datasets/test_image_transforms.py::test_apply_camera_dropout_masks_one_eligible_camera_with_keep_constraint tests/datasets/test_image_transforms.py::test_apply_camera_dropout_respects_eligible_camera_keys -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/datasets/transforms.py tests/datasets/test_image_transforms.py
git commit -m "feat: add camera dropout transform helper"
```

---

### Task 3: Wire Camera Dropout Through Dataset Config and Dataset Reads

**Files:**
- Modify: `src/lerobot/configs/default.py`
- Modify: `src/lerobot/datasets/factory.py`
- Modify: `src/lerobot/datasets/lerobot_dataset.py`
- Modify: `src/lerobot/datasets/streaming_dataset.py`
- Modify: `src/lerobot/datasets/dataset_tools.py`
- Test: `tests/datasets/test_image_transforms.py`

**Interfaces:**
- Consumes: `CameraDropoutConfig` and `apply_camera_dropout()` from Task 2.
- Produces: `DatasetConfig.camera_dropout: CameraDropoutConfig`.
- Produces: `LeRobotDataset(..., camera_dropout: CameraDropoutConfig | None = None)`.
- Produces: `StreamingLeRobotDataset(..., camera_dropout: CameraDropoutConfig | None = None)`.

- [ ] **Step 1: Write a failing dataset-level wiring test with a lightweight fake dataset**

Add this test to `tests/datasets/test_image_transforms.py`:

```python
def test_dataset_item_applies_camera_dropout_after_image_transforms(monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = object.__new__(LeRobotDataset)
    dataset.image_transforms = lambda image: image + 1
    dataset.camera_dropout = CameraDropoutConfig(
        enable=True,
        p=1.0,
        max_num_cameras=1,
        min_num_cameras_to_keep=1,
        value=0.0,
        eligible_camera_keys=["observation.images.head_stereo_left"],
    )
    dataset.delta_indices = None
    dataset._ensure_hf_dataset_loaded = lambda: None
    dataset._query_videos = lambda query_timestamps, ep_idx: {}
    dataset._get_query_timestamps = lambda current_ts, query_indices: []
    dataset.hf_dataset = [
        {
            "episode_index": torch.tensor(0),
            "timestamp": torch.tensor(0.0),
            "task_index": torch.tensor(0),
            "observation.images.head_stereo_left": torch.ones(3, 4, 4),
            "observation.images.head_stereo_right": torch.ones(3, 4, 4) * 2,
        }
    ]
    dataset.meta = type(
        "Meta",
        (),
        {
            "video_keys": [],
            "camera_keys": [
                "observation.images.head_stereo_left",
                "observation.images.head_stereo_right",
            ],
            "tasks": type("Tasks", (), {"iloc": {0: type("Task", (), {"name": "task"})()}})(),
        },
    )()

    item = dataset[0]

    assert torch.count_nonzero(item["observation.images.head_stereo_left"]) == 0
    torch.testing.assert_close(item["observation.images.head_stereo_right"], torch.ones(3, 4, 4) * 3)
```

- [ ] **Step 2: Run the wiring test to verify it fails**

Run:

```bash
pytest tests/datasets/test_image_transforms.py::test_dataset_item_applies_camera_dropout_after_image_transforms -v
```

Expected: FAIL because `LeRobotDataset.__getitem__()` does not call camera dropout.

- [ ] **Step 3: Add camera_dropout config to DatasetConfig**

In `src/lerobot/configs/default.py`, update imports:

```python
from lerobot.datasets.transforms import CameraDropoutConfig, ImageTransformsConfig
```

Add field:

```python
camera_dropout: CameraDropoutConfig = field(default_factory=CameraDropoutConfig)
```

- [ ] **Step 4: Pass camera_dropout from factory to datasets**

In `src/lerobot/datasets/factory.py`, add `camera_dropout=cfg.dataset.camera_dropout` to both dataset constructors:

```python
dataset = LeRobotDataset(
    cfg.dataset.repo_id,
    root=cfg.dataset.root,
    episodes=cfg.dataset.episodes,
    delta_timestamps=delta_timestamps,
    image_transforms=image_transforms,
    camera_dropout=cfg.dataset.camera_dropout,
    revision=cfg.dataset.revision,
    video_backend=cfg.dataset.video_backend,
)
```

```python
dataset = StreamingLeRobotDataset(
    cfg.dataset.repo_id,
    root=cfg.dataset.root,
    episodes=cfg.dataset.episodes,
    delta_timestamps=delta_timestamps,
    image_transforms=image_transforms,
    camera_dropout=cfg.dataset.camera_dropout,
    revision=cfg.dataset.revision,
    max_num_shards=cfg.num_workers,
)
```

- [ ] **Step 5: Update LeRobotDataset constructor and __getitem__**

In `src/lerobot/datasets/lerobot_dataset.py`, update imports:

```python
from lerobot.datasets.transforms import CameraDropoutConfig, apply_camera_dropout
```

Add constructor parameter after `image_transforms`:

```python
camera_dropout: CameraDropoutConfig | None = None,
```

Store it:

```python
self.camera_dropout = camera_dropout or CameraDropoutConfig()
```

In `__getitem__()`, after the per-camera `image_transforms` loop, add:

```python
item = apply_camera_dropout(item, self.meta.camera_keys, self.camera_dropout)
```

Update any dataset clone/copy constructors in `dataset_tools.py` or `lerobot_dataset.py` that already pass `image_transforms=...` to also pass `camera_dropout=...` if the constructor call is for `LeRobotDataset` or `StreamingLeRobotDataset`.

In `src/lerobot/datasets/lerobot_dataset.py`, update `create()` deserialization setup to initialize disabled camera dropout:

```python
obj.camera_dropout = CameraDropoutConfig()
```

In `MultiLeRobotDataset.__init__()`, add `camera_dropout=CameraDropoutConfig()` to the inner `LeRobotDataset(...)` calls unless a later implementation chooses to expose a `camera_dropout` parameter on `MultiLeRobotDataset`.

In `src/lerobot/datasets/dataset_tools.py`, update all four `LeRobotDataset(...)` calls that copy `image_transforms=dataset.image_transforms` or `image_transforms=datasets[0].image_transforms`:

```python
camera_dropout=dataset.camera_dropout,
```

For the merge path that uses `datasets[0]`, add:

```python
camera_dropout=datasets[0].camera_dropout,
```

- [ ] **Step 6: Update StreamingLeRobotDataset constructor and sample path**

In `src/lerobot/datasets/streaming_dataset.py`, update imports:

```python
from lerobot.datasets.transforms import CameraDropoutConfig, apply_camera_dropout
```

Add constructor parameter after `image_transforms`:

```python
camera_dropout: CameraDropoutConfig | None = None,
```

Store it:

```python
self.camera_dropout = camera_dropout or CameraDropoutConfig()
```

After the per-camera `image_transforms` loop in the sample path, add:

```python
video_frames = apply_camera_dropout(video_frames, self.meta.camera_keys, self.camera_dropout)
```

- [ ] **Step 7: Verify dataset wiring test passes**

Run:

```bash
pytest tests/datasets/test_image_transforms.py::test_dataset_item_applies_camera_dropout_after_image_transforms -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/lerobot/configs/default.py src/lerobot/datasets/factory.py src/lerobot/datasets/lerobot_dataset.py src/lerobot/datasets/streaming_dataset.py src/lerobot/datasets/dataset_tools.py tests/datasets/test_image_transforms.py
git commit -m "feat: wire camera dropout through datasets"
```

---

### Task 4: Verify CLI Parsing and Saved Config Shape

**Files:**
- Create: `tests/configs/test_dataset_config.py`
- Modify: `src/lerobot/datasets/transforms.py` if draccus needs a narrower type than `float | str` for `value`

**Interfaces:**
- Consumes: `DatasetConfig.camera_dropout`.
- Produces: CLI-compatible nested parameters for `dataset.camera_dropout` and `dataset.image_transforms.random_erasing`.

- [ ] **Step 1: Write a failing draccus parse test**

Create `tests/configs/test_dataset_config.py`:

```python
import draccus

from lerobot.configs.default import DatasetConfig


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
    assert cfg.image_transforms.tfs["random_erasing"].weight == 1.0
    assert cfg.image_transforms.tfs["random_erasing"].type == "RandomErasing"
    assert cfg.image_transforms.tfs["random_erasing"].kwargs["p"] == 0.15
    assert cfg.image_transforms.tfs["random_erasing"].kwargs["scale"] == [0.02, 0.08]
    assert cfg.camera_dropout.enable is True
    assert cfg.camera_dropout.p == 0.05
    assert cfg.camera_dropout.max_num_cameras == 1
    assert cfg.camera_dropout.min_num_cameras_to_keep == 3
    assert cfg.camera_dropout.value == 0.0
    assert cfg.camera_dropout.eligible_camera_keys == [
        "observation.images.head_stereo_left",
        "observation.images.wrist_left",
    ]
```

- [ ] **Step 2: Run parse test to verify the CLI contract**

Run:

```bash
pytest tests/configs/test_dataset_config.py::test_dataset_config_parses_mask_augmentation_cli_flags -v
```

Expected: PASS. If it fails because draccus cannot parse `float | str`, change `CameraDropoutConfig.value` to `Any = 0.0` and keep `_make_camera_dropout_value()` validation from Task 2.

- [ ] **Step 3: Add a regression assertion for disabled defaults**

Append this test:

```python
def test_dataset_config_mask_defaults_are_disabled():
    cfg = draccus.parse(DatasetConfig, args=["--repo_id=/tmp/example"])

    assert cfg.camera_dropout.enable is False
    assert cfg.image_transforms.tfs["random_erasing"].weight == 0.0
```

- [ ] **Step 4: Run config tests**

Run:

```bash
pytest tests/configs/test_dataset_config.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/configs/test_dataset_config.py src/lerobot/datasets/transforms.py
git commit -m "test: cover mask augmentation cli parsing"
```

---

### Task 5: Document Mask Parameters and Run Focused Regression

**Files:**
- Modify: `docs/source/lerobot-dataset-v3.mdx`
- Test: documentation is verified by reading the rendered source text and running focused tests.

**Interfaces:**
- Consumes: CLI flags and config names from Tasks 1-4.
- Produces: Chinese documentation for local region mask and whole-camera mask.

- [ ] **Step 1: Add Chinese documentation under the Image transforms section**

In `docs/source/lerobot-dataset-v3.mdx`, add this subsection after the existing transform examples:

```md
### Mask 增强参数说明

LeRobot 支持两类可选 Mask 增强，二者默认都不开启：

- **局部区域 Mask**：通过 `dataset.image_transforms.random_erasing` 配置，使用 `torchvision.transforms.v2.RandomErasing` 对单张摄像头图片随机擦除一个矩形区域。
- **整路摄像头 Mask**：通过 `dataset.camera_dropout` 配置，在一个训练样本中随机将某一路或某几路摄像头图像整体置为指定值。

局部区域 Mask 常用参数：

- `p`：对单张图片执行局部 Mask 的概率。
- `scale`：Mask 区域面积占整张图片面积的比例范围，建议从 `[0.02, 0.08]` 或 `[0.02, 0.12]` 开始。
- `ratio`：Mask 矩形宽高比范围，默认可使用 `[0.3, 3.3]`。
- `value`：Mask 区域填充值。图片在 dataset 阶段通常为 `[0, 1]`，`0.0` 表示黑色区域，`"random"` 表示随机像素值。

整路摄像头 Mask 常用参数：

- `enable`：是否开启整路摄像头 Mask，默认 `false`。
- `p`：对一个训练样本执行整路摄像头 Mask 的概率。
- `max_num_cameras`：一次最多 Mask 掉几路摄像头。四路摄像头任务建议从 `1` 开始。
- `min_num_cameras_to_keep`：每个样本至少保留几路摄像头。四路摄像头任务建议从 `3` 开始。
- `value`：整路摄像头被 Mask 后的填充值。
- `eligible_camera_keys`：允许被整路 Mask 的摄像头列表。未列入的摄像头不会被该增强 Mask。

整路摄像头 Mask 不会把 PI0.5 的 `img_masks` 改成 `false`。它模拟的是图像内容被遮挡或不可用，而不是摄像头字段缺失。
```

- [ ] **Step 2: Add CLI examples**

Add this example below the parameter list:

````md
局部区域 Mask 命令行示例：

```bash
--dataset.image_transforms.enable=true \
--dataset.image_transforms.random_erasing.enable=true \
--dataset.image_transforms.random_erasing.weight=1.0 \
--dataset.image_transforms.random_erasing.p=0.15 \
--dataset.image_transforms.random_erasing.scale='[0.02,0.08]' \
--dataset.image_transforms.random_erasing.ratio='[0.3,3.3]' \
--dataset.image_transforms.random_erasing.value=0.0
```

整路摄像头 Mask 命令行示例：

```bash
--dataset.camera_dropout.enable=true \
--dataset.camera_dropout.p=0.05 \
--dataset.camera_dropout.max_num_cameras=1 \
--dataset.camera_dropout.min_num_cameras_to_keep=3 \
--dataset.camera_dropout.value=0.0 \
--dataset.camera_dropout.eligible_camera_keys='["observation.images.head_stereo_left","observation.images.head_stereo_right","observation.images.wrist_left","observation.images.wrist_right"]'
```
````

- [ ] **Step 3: Run focused regression tests**

Run:

```bash
pytest tests/datasets/test_image_transforms.py tests/configs/test_dataset_config.py -v
```

Expected: all selected tests PASS.

- [ ] **Step 4: Run formatting/lint on touched Python files if available**

Run:

```bash
ruff check src/lerobot/datasets/transforms.py src/lerobot/configs/default.py src/lerobot/datasets/factory.py src/lerobot/datasets/lerobot_dataset.py src/lerobot/datasets/streaming_dataset.py tests/datasets/test_image_transforms.py tests/configs/test_dataset_config.py
```

Expected: PASS. If `ruff` is unavailable, record the command failure in the final implementation notes and rely on pytest results.

- [ ] **Step 5: Commit**

```bash
git add docs/source/lerobot-dataset-v3.mdx
git commit -m "docs: explain mask augmentation options"
```

---

## Final Verification

- [ ] Run focused tests:

```bash
pytest tests/datasets/test_image_transforms.py tests/configs/test_dataset_config.py -v
```

Expected: PASS.

- [ ] Run final git status:

```bash
git status --short
```

Expected: no uncommitted changes unless the user has unrelated local edits.

## Self-Review Notes

- Spec coverage: local RandomErasing is covered by Task 1; whole-camera dropout helper by Task 2; dataset wiring by Task 3; CLI control by Task 4; Chinese docs by Task 5.
- Placeholder scan: no placeholder sections remain; edge cases are specified in helper tests and implementation steps.
- Type consistency: `CameraDropoutConfig`, `apply_camera_dropout()`, and `DatasetConfig.camera_dropout` names are consistent across all tasks.
