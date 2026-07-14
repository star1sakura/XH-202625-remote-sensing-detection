# SPH-YOLOv5 Small-Object Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build SPH-YOLOv5-inspired P2/NAM/Swin YOLO experiment configs for XH25, starting with a P2 small-object head aimed at FSC vehicle recall.

**Architecture:** Keep the existing Ultralytics YOLO26-style training, tiled inference, and evaluation pipeline. Add SPH modules as channel-preserving custom PyTorch modules, then create three YAML model variants: `sph-p2`, `sph-p2-nam`, and `sph-full`.

**Tech Stack:** Python 3.12, PyTorch, Ultralytics YOLO YAML parsing, pytest, repository CLI `xh-detect`.

## Global Constraints

- Keep `xh25-yolo26s-e80` as the safety baseline.
- First trainable candidate is `sph-p2`; NAM and Swin are later ablations.
- Do not combine SPH with MKSNet full backbone replacement.
- `sph-p2` Detect must use four scales: P2, P3, P4, P5.
- Preferred candidate criteria: Vehicle Recall greater than `0.705128`, Vehicle FDR at most `0.202899` after raw or thresholded evaluation, Ship Recall drop no more than `0.02`, Aircraft Recall drop no more than `0.005`.
- Keep 10000 x 10000 tiled inference within the RTX3090 timing budget from the competition scoring scheme.

---

## File Structure

- `src/xh_detect/models/sph_yolo.py`
  - Owns SPH custom modules only: `NAMBlock` and `SwinPredictionBlock`.
  - No training or evaluation code belongs here.
- `src/xh_detect/models/__init__.py`
  - Re-exports SPH modules beside existing MKS modules.
- `src/xh_detect/models/ultralytics.py`
  - Registers SPH modules with Ultralytics YAML parsing and checkpoint loading.
- `configs/models/xh25-yolo26s-sph-p2.yaml`
  - YOLO26-style model with added P2 small-object path and four-scale Detect.
- `configs/models/xh25-yolo26s-sph-p2-nam.yaml`
  - `sph-p2` plus NAM after P2 and P3 fusion outputs.
- `configs/models/xh25-yolo26s-sph-full.yaml`
  - `sph-p2-nam` plus Swin prediction blocks on P2/P3/P4/P5 outputs.
- `configs/xh25-sph-p2.yaml`
  - Runtime inference config pointing to `runs/train/xh25-sph-p2/weights/best.pt`.
- `configs/xh25-sph-p2-nam.yaml`
  - Runtime inference config pointing to `runs/train/xh25-sph-p2-nam/weights/best.pt`.
- `configs/xh25-sph-full.yaml`
  - Runtime inference config pointing to `runs/train/xh25-sph-full/weights/best.pt`.
- `tests/test_sph_yolo.py`
  - Unit tests for SPH module shape preservation, validation, gradients, and registration.
- `tests/test_sph_configs.py`
  - YAML structure, `PipelineConfig`, and YOLO smoke-load tests.
- `docs/experiments/sph-yolov5-small-object.md`
  - Runbook and result table for main vs SPH candidates.

---

### Task 1: Add SPH Module Tests And Implementation

**Files:**
- Create: `tests/test_sph_yolo.py`
- Create: `src/xh_detect/models/sph_yolo.py`
- Modify: `src/xh_detect/models/__init__.py`
- Modify: `src/xh_detect/models/ultralytics.py`

**Interfaces:**
- Consumes: existing `register_custom_modules()` pattern in `src/xh_detect/models/ultralytics.py`.
- Produces:
  - `NAMBlock(channels: int, use_spatial: bool = True, eps: float = 1e-6) -> nn.Module`
  - `SwinPredictionBlock(channels: int, num_heads: int = 4, window_size: int = 7, mlp_ratio: float = 4.0) -> nn.Module`
  - `register_custom_modules()` exposes `tasks.NAMBlock` and `tasks.SwinPredictionBlock`.

- [ ] **Step 1: Write failing SPH module tests**

Create `tests/test_sph_yolo.py` with:

```python
from __future__ import annotations

import pytest
import torch
import ultralytics.nn.tasks as tasks

from xh_detect.models.sph_yolo import NAMBlock, SwinPredictionBlock
from xh_detect.models.ultralytics import register_custom_modules


def test_nam_block_preserves_shape_and_allows_gradients() -> None:
    block = NAMBlock(16)
    x = torch.randn(2, 16, 12, 10, requires_grad=True)

    y = block(x)
    y.mean().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(y).all()


def test_nam_block_can_disable_spatial_attention() -> None:
    block = NAMBlock(8, use_spatial=False)
    x = torch.randn(1, 8, 6, 6)

    y = block(x)

    assert y.shape == x.shape


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channels": 0}, "channels must be a positive integer"),
        ({"channels": True}, "channels must be a positive integer"),
        ({"channels": 8, "eps": 0.0}, "eps must be positive"),
    ],
)
def test_nam_block_validates_arguments(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        NAMBlock(**kwargs)


def test_swin_prediction_block_preserves_shape_when_padding_is_needed() -> None:
    block = SwinPredictionBlock(24, num_heads=4, window_size=7, mlp_ratio=2.0)
    x = torch.randn(2, 24, 15, 13, requires_grad=True)

    y = block(x)
    y.sum().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channels": 0}, "channels must be a positive integer"),
        ({"channels": 16, "num_heads": 0}, "num_heads must be a positive integer"),
        ({"channels": 16, "num_heads": 3}, "channels must be divisible by num_heads"),
        ({"channels": 16, "window_size": 0}, "window_size must be a positive integer"),
        ({"channels": 16, "mlp_ratio": 0.0}, "mlp_ratio must be positive"),
    ],
)
def test_swin_prediction_block_validates_arguments(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SwinPredictionBlock(**kwargs)


def test_register_custom_modules_exposes_sph_blocks_to_ultralytics() -> None:
    original_nam = getattr(tasks, "NAMBlock", None)
    original_swin = getattr(tasks, "SwinPredictionBlock", None)
    try:
        if hasattr(tasks, "NAMBlock"):
            delattr(tasks, "NAMBlock")
        if hasattr(tasks, "SwinPredictionBlock"):
            delattr(tasks, "SwinPredictionBlock")

        register_custom_modules()

        assert tasks.NAMBlock is NAMBlock
        assert tasks.SwinPredictionBlock is SwinPredictionBlock
    finally:
        if original_nam is not None:
            tasks.NAMBlock = original_nam
        elif hasattr(tasks, "NAMBlock"):
            delattr(tasks, "NAMBlock")
        if original_swin is not None:
            tasks.SwinPredictionBlock = original_swin
        elif hasattr(tasks, "SwinPredictionBlock"):
            delattr(tasks, "SwinPredictionBlock")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_sph_yolo.py -q
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'xh_detect.models.sph_yolo'`.

- [ ] **Step 3: Implement SPH modules**

Create `src/xh_detect/models/sph_yolo.py` with:

```python
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


class NAMBlock(nn.Module):
    """Shape-preserving normalization-based attention block for SPH-YOLO heads."""

    def __init__(self, channels: int, use_spatial: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        eps = _positive_float(eps, "eps")
        if not isinstance(use_spatial, bool):
            raise TypeError("use_spatial must be a boolean")

        self.channels = channels
        self.use_spatial = use_spatial
        self.eps = eps
        self.channel_norm = nn.BatchNorm2d(channels, affine=True)
        self.spatial_norm = nn.BatchNorm2d(1, affine=True) if use_spatial else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_scale = self.channel_norm.weight.abs()
        channel_scale = channel_scale / (channel_scale.sum() + self.eps)
        channel_gate = torch.sigmoid(
            self.channel_norm(x) * channel_scale.view(1, self.channels, 1, 1) * self.channels
        )
        y = x * channel_gate
        if self.spatial_norm is not None:
            spatial = y.mean(dim=1, keepdim=True)
            y = y * torch.sigmoid(self.spatial_norm(spatial))
        return x + y


class SwinPredictionBlock(nn.Module):
    """Window-attention prediction block that preserves BCHW tensor shape."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        num_heads = _positive_int(num_heads, "num_heads")
        window_size = _positive_int(window_size, "window_size")
        mlp_ratio = _positive_float(mlp_ratio, "mlp_ratio")
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        hidden_channels = max(channels, int(math.ceil(channels * mlp_ratio)))
        self.channels = channels
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    def _partition_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        b, c, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h = h + pad_h
        padded_w = w + pad_w
        x = x.permute(0, 2, 3, 1).contiguous()
        windows = x.view(
            b,
            padded_h // self.window_size,
            self.window_size,
            padded_w // self.window_size,
            self.window_size,
            c,
        )
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, self.window_size * self.window_size, c), padded_h, padded_w

    def _merge_windows(
        self,
        windows: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        padded_h: int,
        padded_w: int,
    ) -> torch.Tensor:
        x = windows.view(
            batch_size,
            padded_h // self.window_size,
            padded_w // self.window_size,
            self.window_size,
            self.window_size,
            self.channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, padded_h, padded_w, self.channels)
        x = x[:, :height, :width, :]
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        windows, padded_h, padded_w = self._partition_windows(x)
        attn_input = self.norm1(windows)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        windows = windows + attn_output
        windows = windows + self.mlp(self.norm2(windows))
        return self._merge_windows(windows, b, h, w, padded_h, padded_w)
```

- [ ] **Step 4: Export and register SPH modules**

Modify `src/xh_detect/models/__init__.py` so it contains these imports and exports:

```python
from xh_detect.models.mksnet_lite import MKSNetLiteBlock
from xh_detect.models.mksnet_v2 import (
    MKSBlock,
    MKSChannelAttention,
    MKSSpatialAttention,
    MKSStage,
)
from xh_detect.models.sph_yolo import NAMBlock, SwinPredictionBlock

__all__ = [
    "MKSNetLiteBlock",
    "MKSChannelAttention",
    "MKSSpatialAttention",
    "MKSBlock",
    "MKSStage",
    "NAMBlock",
    "SwinPredictionBlock",
]
```

Modify `src/xh_detect/models/ultralytics.py` so it imports SPH modules:

```python
from xh_detect.models.sph_yolo import NAMBlock, SwinPredictionBlock
```

and add these assignments inside `register_custom_modules()`:

```python
    tasks.NAMBlock = NAMBlock
    tasks.SwinPredictionBlock = SwinPredictionBlock
```

- [ ] **Step 5: Run SPH module tests**

Run:

```bash
python -m pytest tests/test_sph_yolo.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/xh_detect/models/sph_yolo.py src/xh_detect/models/__init__.py src/xh_detect/models/ultralytics.py tests/test_sph_yolo.py
git commit -m "feat: add sph yolov5 modules"
```

---

### Task 2: Add P2 SPH Model And Runtime Config

**Files:**
- Create: `configs/models/xh25-yolo26s-sph-p2.yaml`
- Create: `configs/xh25-sph-p2.yaml`
- Create: `tests/test_sph_configs.py`

**Interfaces:**
- Consumes: existing Ultralytics built-in layers `Conv`, `C3k2`, `SPPF`, `C2PSA`, `Concat`, `Detect`.
- Produces:
  - model YAML `configs/models/xh25-yolo26s-sph-p2.yaml`
  - pipeline YAML `configs/xh25-sph-p2.yaml`
  - Detect layer `[[19, 22, 25, 28], 1, Detect, [nc]]`.

- [ ] **Step 1: Write failing P2 config tests**

Create `tests/test_sph_configs.py` with:

```python
from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO

from xh_detect.config import PipelineConfig
from xh_detect.models.ultralytics import register_custom_modules


def _load_model_yaml(path: str) -> dict[str, object]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _layers(model: dict[str, object]) -> list[list[object]]:
    return list(model["backbone"]) + list(model["head"])


def test_sph_p2_model_yaml_adds_four_scale_detect() -> None:
    model = _load_model_yaml("configs/models/xh25-yolo26s-sph-p2.yaml")
    layers = _layers(model)

    assert model["nc"] == 25
    assert model["scale"] == "s"
    assert model["end2end"] is True
    assert model["reg_max"] == 1
    assert layers[17] == [-1, 1, "nn.Upsample", [None, 2, "nearest"]]
    assert layers[18] == [[-1, 2], 1, "Concat", [1]]
    assert layers[19] == [-1, 2, "C3k2", [128, True]]
    assert layers[-1] == [[19, 22, 25, 28], 1, "Detect", ["nc"]]


def test_sph_p2_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-sph-p2.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-sph-p2/weights/best.pt"
    assert config.image_size == 1024
    assert config.batch_size == 8
    assert set(config.class_thresholds) == set(range(25))


def test_sph_p2_model_smoke_loads_with_detection_model() -> None:
    register_custom_modules()

    model = YOLO("configs/models/xh25-yolo26s-sph-p2.yaml")

    assert model.model.__class__.__name__ == "DetectionModel"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Expected: FAIL because `configs/models/xh25-yolo26s-sph-p2.yaml` does not exist.

- [ ] **Step 3: Create P2 model YAML**

Create `configs/models/xh25-yolo26s-sph-p2.yaml` with:

```yaml
# XH25 YOLO26s-style HBB detector with SPH-YOLOv5-inspired P2 small-object head.
nc: 25
end2end: true
reg_max: 1
scale: s
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, false, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, false, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, true]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, true]]
  - [-1, 1, SPPF, [1024, 5, 3, true]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, true]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, true]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, true]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, true]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, true]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, true, 0.5, true]]

  - [[19, 22, 25, 28], 1, Detect, [nc]]
```

- [ ] **Step 4: Create P2 runtime config**

Create `configs/xh25-sph-p2.yaml` with:

```yaml
task: detect
taxonomy: xh25
model_path: runs/train/xh25-sph-p2/weights/best.pt
device: "0"
image_size: 1024
tile_size: 1024
overlap: 0.2
batch_size: 8
merge_iou: 0.3
edge_margin: 16
half: true
class_thresholds:
  0: 0.25
  1: 0.25
  2: 0.25
  3: 0.25
  4: 0.25
  5: 0.25
  6: 0.25
  7: 0.25
  8: 0.25
  9: 0.25
  10: 0.25
  11: 0.25
  12: 0.25
  13: 0.25
  14: 0.25
  15: 0.25
  16: 0.25
  17: 0.25
  18: 0.25
  19: 0.25
  20: 0.25
  21: 0.25
  22: 0.25
  23: 0.25
  24: 0.25
```

- [ ] **Step 5: Run P2 config tests**

Run:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add configs/models/xh25-yolo26s-sph-p2.yaml configs/xh25-sph-p2.yaml tests/test_sph_configs.py
git commit -m "config: add sph p2 model"
```

---

### Task 3: Add NAM And Full SPH Config Variants

**Files:**
- Modify: `tests/test_sph_configs.py`
- Create: `configs/models/xh25-yolo26s-sph-p2-nam.yaml`
- Create: `configs/models/xh25-yolo26s-sph-full.yaml`
- Create: `configs/xh25-sph-p2-nam.yaml`
- Create: `configs/xh25-sph-full.yaml`

**Interfaces:**
- Consumes:
  - `NAMBlock(channels: int, use_spatial: bool = True, eps: float = 1e-6)`
  - `SwinPredictionBlock(channels: int, num_heads: int = 4, window_size: int = 7, mlp_ratio: float = 4.0)`
- Produces:
  - `sph-p2-nam` Detect layer `[[20, 24, 27, 30], 1, Detect, [nc]]`
  - `sph-full` Detect layer `[[21, 26, 30, 34], 1, Detect, [nc]]`

- [ ] **Step 1: Extend config tests for NAM and full variants**

Append these tests to `tests/test_sph_configs.py`:

```python
def test_sph_p2_nam_model_yaml_adds_nam_blocks() -> None:
    model = _load_model_yaml("configs/models/xh25-yolo26s-sph-p2-nam.yaml")
    layers = _layers(model)

    nam_layers = [layer for layer in layers if layer[2] == "NAMBlock"]

    assert len(nam_layers) == 2
    assert layers[20] == [-1, 1, "NAMBlock", [128]]
    assert layers[24] == [-1, 1, "NAMBlock", [256]]
    assert layers[-1] == [[20, 24, 27, 30], 1, "Detect", ["nc"]]


def test_sph_full_model_yaml_adds_swin_prediction_blocks() -> None:
    model = _load_model_yaml("configs/models/xh25-yolo26s-sph-full.yaml")
    layers = _layers(model)

    swin_layers = [layer for layer in layers if layer[2] == "SwinPredictionBlock"]

    assert len(swin_layers) == 4
    assert layers[21] == [-1, 1, "SwinPredictionBlock", [128, 4, 7, 2.0]]
    assert layers[26] == [-1, 1, "SwinPredictionBlock", [256, 4, 7, 2.0]]
    assert layers[30] == [-1, 1, "SwinPredictionBlock", [512, 8, 7, 2.0]]
    assert layers[34] == [-1, 1, "SwinPredictionBlock", [1024, 8, 7, 2.0]]
    assert layers[-1] == [[21, 26, 30, 34], 1, "Detect", ["nc"]]


def test_sph_p2_nam_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-sph-p2-nam.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-sph-p2-nam/weights/best.pt"
    assert config.image_size == 1024
    assert config.batch_size == 8
    assert set(config.class_thresholds) == set(range(25))


def test_sph_full_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-sph-full.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-sph-full/weights/best.pt"
    assert config.image_size == 1024
    assert config.batch_size == 8
    assert set(config.class_thresholds) == set(range(25))


@pytest.mark.parametrize(
    "path",
    [
        "configs/models/xh25-yolo26s-sph-p2-nam.yaml",
        "configs/models/xh25-yolo26s-sph-full.yaml",
    ],
)
def test_sph_custom_model_variants_smoke_load_with_detection_model(path: str) -> None:
    register_custom_modules()

    model = YOLO(path)

    assert model.model.__class__.__name__ == "DetectionModel"
```

Add `import pytest` to the top of `tests/test_sph_configs.py`:

```python
import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Expected: FAIL because NAM and full YAML files do not exist.

- [ ] **Step 3: Create NAM model YAML**

Create `configs/models/xh25-yolo26s-sph-p2-nam.yaml` with:

```yaml
# XH25 SPH-YOLOv5-inspired detector with P2 head and NAM attention.
nc: 25
end2end: true
reg_max: 1
scale: s
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, false, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, false, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, true]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, true]]
  - [-1, 1, SPPF, [1024, 5, 3, true]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, true]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, true]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, true]]
  - [-1, 1, NAMBlock, [128]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, true]]
  - [-1, 1, NAMBlock, [256]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, true]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, true, 0.5, true]]

  - [[20, 24, 27, 30], 1, Detect, [nc]]
```

- [ ] **Step 4: Create full SPH model YAML**

Create `configs/models/xh25-yolo26s-sph-full.yaml` with:

```yaml
# XH25 SPH-YOLOv5-inspired detector with P2 head, NAM, and Swin prediction blocks.
nc: 25
end2end: true
reg_max: 1
scale: s
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, false, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, false, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, true]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, true]]
  - [-1, 1, SPPF, [1024, 5, 3, true]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, true]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, true]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, true]]
  - [-1, 1, NAMBlock, [128]]
  - [-1, 1, SwinPredictionBlock, [128, 4, 7, 2.0]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, true]]
  - [-1, 1, NAMBlock, [256]]
  - [-1, 1, SwinPredictionBlock, [256, 4, 7, 2.0]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, true]]
  - [-1, 1, SwinPredictionBlock, [512, 8, 7, 2.0]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, true, 0.5, true]]
  - [-1, 1, SwinPredictionBlock, [1024, 8, 7, 2.0]]

  - [[21, 26, 30, 34], 1, Detect, [nc]]
```

- [ ] **Step 5: Create NAM and full runtime configs**

Create `configs/xh25-sph-p2-nam.yaml` with:

```yaml
task: detect
taxonomy: xh25
model_path: runs/train/xh25-sph-p2-nam/weights/best.pt
device: "0"
image_size: 1024
tile_size: 1024
overlap: 0.2
batch_size: 8
merge_iou: 0.3
edge_margin: 16
half: true
class_thresholds:
  0: 0.25
  1: 0.25
  2: 0.25
  3: 0.25
  4: 0.25
  5: 0.25
  6: 0.25
  7: 0.25
  8: 0.25
  9: 0.25
  10: 0.25
  11: 0.25
  12: 0.25
  13: 0.25
  14: 0.25
  15: 0.25
  16: 0.25
  17: 0.25
  18: 0.25
  19: 0.25
  20: 0.25
  21: 0.25
  22: 0.25
  23: 0.25
  24: 0.25
```

Create `configs/xh25-sph-full.yaml` with:

```yaml
task: detect
taxonomy: xh25
model_path: runs/train/xh25-sph-full/weights/best.pt
device: "0"
image_size: 1024
tile_size: 1024
overlap: 0.2
batch_size: 8
merge_iou: 0.3
edge_margin: 16
half: true
class_thresholds:
  0: 0.25
  1: 0.25
  2: 0.25
  3: 0.25
  4: 0.25
  5: 0.25
  6: 0.25
  7: 0.25
  8: 0.25
  9: 0.25
  10: 0.25
  11: 0.25
  12: 0.25
  13: 0.25
  14: 0.25
  15: 0.25
  16: 0.25
  17: 0.25
  18: 0.25
  19: 0.25
  20: 0.25
  21: 0.25
  22: 0.25
  23: 0.25
  24: 0.25
```

- [ ] **Step 6: Run SPH config tests**

Run:

```bash
python -m pytest tests/test_sph_configs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add configs/models/xh25-yolo26s-sph-p2-nam.yaml configs/models/xh25-yolo26s-sph-full.yaml configs/xh25-sph-p2-nam.yaml configs/xh25-sph-full.yaml tests/test_sph_configs.py
git commit -m "config: add sph nam and full variants"
```

---

### Task 4: Add SPH Experiment Runbook

**Files:**
- Create: `docs/experiments/sph-yolov5-small-object.md`

**Interfaces:**
- Consumes:
  - model YAMLs from Tasks 2 and 3
  - runtime configs from Tasks 2 and 3
  - baseline metrics from `docs/experiments/xh25-yolo26s-e80.md`
- Produces:
  - documented training/evaluation commands and result table structure.

- [ ] **Step 1: Create runbook**

Create `docs/experiments/sph-yolov5-small-object.md` with:

```markdown
# SPH-YOLOv5 Small-Object Experiment

This experiment adapts SPH-YOLOv5 ideas to the XH25 YOLO26-style detector. The
first trainable candidate is `sph-p2`, which adds a shallow P2 detection path for
FSC vehicle targets. NAM and Swin variants are follow-up ablations.

## Baseline

| Candidate | Overall Recall | Overall FDR | Ship Recall | Ship FDR | Aircraft Recall | Aircraft FDR | Vehicle Recall | Vehicle FDR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main / xh25-yolo26s-e80 | 0.961562 | 0.037244 | 0.823383 | 0.157761 | 0.989075 | 0.015942 | 0.705128 | 0.202899 |
| sph-p2 raw | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |
| sph-p2 thresholded | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |
| sph-p2-nam raw | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |
| sph-full raw | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |

## Local Smoke Tests

```bash
python -m pytest tests/test_sph_yolo.py tests/test_sph_configs.py -q
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); model=YOLO('configs/models/xh25-yolo26s-sph-p2.yaml'); print(model.model.__class__.__name__)"
```

## Train P2 Candidate

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model configs/models/xh25-yolo26s-sph-p2.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-sph-p2 \
  --no-resume
```

## Evaluate P2 Candidate

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-sph-p2.yaml \
  --output-json outputs/xh25/sph-p2/val-predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/sph-p2/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-path outputs/xh25/sph-p2/report.json \
  --taxonomy xh25

.venv/bin/xh-detect competition-report \
  --report-json outputs/xh25/sph-p2/report.json \
  --output-dir outputs/xh25/sph-p2/competition-proxy \
  --experiment-name xh25-sph-p2
```

## Threshold Calibration

```bash
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/sph-p2/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-dir outputs/xh25/sph-p2/threshold-optimized \
  --taxonomy xh25 \
  --baseline-report outputs/xh25/baseline/report.json \
  --experiment-name xh25-sph-p2-threshold-optimized
```

## Keep Criteria

Prefer the SPH candidate only if:

- Vehicle Recall is greater than 0.705128.
- Vehicle FDR is at most 0.202899, or threshold optimization reaches that value while keeping the recall gain.
- Ship Recall is at least 0.803383.
- Aircraft Recall is at least 0.984075.
- Overall Recall and Overall FDR pass the competition hard gates.
```

- [ ] **Step 2: Commit Task 4**

Run:

```bash
git add docs/experiments/sph-yolov5-small-object.md
git commit -m "docs: add sph yolov5 runbook"
```

---

### Task 5: Full Verification

**Files:**
- Read: all files changed in Tasks 1-4.

**Interfaces:**
- Consumes: completed SPH modules, YAMLs, configs, and docs.
- Produces: verified branch ready for training on the RTX3090 server.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_sph_yolo.py tests/test_sph_configs.py tests/test_mksnet_configs.py -q
```

Expected: PASS.

- [ ] **Step 2: Run formatting check**

Run:

```bash
python -m ruff format --check src tests
```

Expected: PASS. If it fails, run:

```bash
python -m ruff format src tests
```

Then rerun the format check.

- [ ] **Step 3: Run lint check**

Run:

```bash
python -m ruff check src tests
```

Expected: PASS.

- [ ] **Step 4: Smoke-load all SPH YAMLs**

Run:

```bash
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); paths=['configs/models/xh25-yolo26s-sph-p2.yaml','configs/models/xh25-yolo26s-sph-p2-nam.yaml','configs/models/xh25-yolo26s-sph-full.yaml']; [print(path, YOLO(path).model.__class__.__name__) for path in paths]"
```

Expected output includes:

```text
configs/models/xh25-yolo26s-sph-p2.yaml DetectionModel
configs/models/xh25-yolo26s-sph-p2-nam.yaml DetectionModel
configs/models/xh25-yolo26s-sph-full.yaml DetectionModel
```

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
git log --oneline -6
```

Expected: no unstaged or uncommitted files except intentionally ignored local artifacts; recent commits include the four SPH implementation commits.
