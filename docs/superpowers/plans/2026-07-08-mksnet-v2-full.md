# MKSNet-v2 Full Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a MKSNet-style full backbone experiment for XH25 vehicle/FSC small-object detection, then train and evaluate it against the main `xh25-yolo26s-e80` baseline.

**Architecture:** Add paper-facing MKSNet v2 PyTorch modules: channel attention, spatial multi-kernel attention with adaptive branch selection, MKS blocks, MKS stages, and a standalone MKSNet backbone. Use explicit `MKSStage` layers in an Ultralytics YAML so P3/P4/P5 feature maps remain compatible with the existing YOLO detection head and current training, inference, threshold, and competition-report pipeline.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics 8.4.71, PyYAML, Typer, pytest, ruff, existing `xh_detect` CLI.

## Global Constraints

- Train the first full run on `datasets/xh25/dataset.yaml`, not the ship-balanced dataset.
- Keep image size `1024`, first-run epochs `80`, batch `8`, workers `4`, and `--no-amp`.
- Use `P3/P4/P5` detection first; do not add a P2 detection head in this implementation.
- Use vehicle/FSC as the primary target, but keep ship and aircraft metrics as hard regression checks.
- Report raw evaluation before threshold calibration.
- Do not claim official bit-for-bit reproduction without official code.
- Keep the result description as "MKSNet-style full backbone reproduction adapted to the XH25 YOLO detection pipeline."

---

## File Structure

- Create `src/xh_detect/models/mksnet_v2.py`: paper-facing modules `MKSChannelAttention`, `MKSSpatialAttention`, `MKSBlock`, `MKSStage`, and `MKSNetBackbone`.
- Modify `src/xh_detect/models/__init__.py`: export MKSNet v2 modules alongside `MKSNetLiteBlock`.
- Modify `src/xh_detect/models/ultralytics.py`: register MKSNet v2 modules in `ultralytics.nn.tasks` before YAML parsing.
- Create `tests/test_mksnet_v2.py`: unit tests for attention paths, branch selection, MKS order validation, stage shape preservation, standalone backbone outputs, and Ultralytics registration.
- Modify `tests/test_mksnet_configs.py`: add static checks for the full MKSNet YAML and pipeline config.
- Create `configs/models/xh25-yolo-mksnet-v2-full.yaml`: Ultralytics-compatible full-backbone experiment using explicit MKS stages.
- Create `configs/xh25-mksnet-v2-full.yaml`: inference config pointing to `runs/train/xh25-mksnet-v2-full-vehicle/weights/best.pt`.
- Create `docs/experiments/mksnet-v2-full-vehicle.md`: runbook and result table for local smoke, server training, raw evaluation, calibrated evaluation, and comparison to main/MKSNet-Lite.

## Task 1: MKSNet v2 Core Modules

**Files:**
- Create: `src/xh_detect/models/mksnet_v2.py`
- Create: `tests/test_mksnet_v2.py`

**Interfaces:**
- Produces: `MKSChannelAttention(channels: int, reduction: int = 16) -> nn.Module`
- Produces: `MKSSpatialAttention(channels: int, kernel_sizes: Sequence[int] = (3, 5, 7, 9), dilations: Sequence[int] = (1, 1, 2, 2), reduction: int = 16, branch_reduction: int = 4) -> nn.Module`
- Produces: `MKSSpatialAttention.selection_weights(x: torch.Tensor) -> torch.Tensor`
- Produces: `MKSBlock(channels: int, kernel_sizes: Sequence[int] = (3, 5, 7, 9), dilations: Sequence[int] = (1, 1, 2, 2), reduction: int = 16, order: str = "ca_sa") -> nn.Module`

- [ ] **Step 1: Write failing core module tests**

Create `tests/test_mksnet_v2.py` with these initial tests:

```python
from __future__ import annotations

import pytest
import torch

from xh_detect.models.mksnet_v2 import (
    MKSBlock,
    MKSChannelAttention,
    MKSSpatialAttention,
)


def test_channel_attention_preserves_shape_and_uses_avg_and_max_paths() -> None:
    torch.manual_seed(7)
    module = MKSChannelAttention(16, reduction=4)
    x = torch.randn(2, 16, 12, 14, requires_grad=True)

    y = module(x)
    y.mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert module.avg_pool.output_size == 1
    assert module.max_pool.output_size == 1


def test_spatial_attention_selection_weights_sum_to_one() -> None:
    torch.manual_seed(7)
    module = MKSSpatialAttention(
        24,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 1, 2),
        reduction=4,
        branch_reduction=2,
    )
    x = torch.randn(2, 24, 16, 16)

    weights = module.selection_weights(x)
    y = module(x)

    assert weights.shape == (2, 3, 1, 1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1, 1), atol=1e-6)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mks_block_supports_both_attention_orders() -> None:
    x = torch.randn(1, 16, 10, 10)

    ca_sa = MKSBlock(16, kernel_sizes=(3, 5), dilations=(1, 2), reduction=4, order="ca_sa")
    sa_ca = MKSBlock(16, kernel_sizes=(3, 5), dilations=(1, 2), reduction=4, order="sa_ca")

    assert ca_sa(x).shape == x.shape
    assert sa_ca(x).shape == x.shape


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: MKSChannelAttention(0), "channels must be a positive integer"),
        (lambda: MKSChannelAttention(8, reduction=0), "reduction must be a positive integer"),
        (
            lambda: MKSSpatialAttention(8, kernel_sizes=(3,), dilations=(1, 2)),
            "kernel_sizes and dilations must have the same length",
        ),
        (
            lambda: MKSSpatialAttention(8, kernel_sizes=(2,), dilations=(1,)),
            "kernel sizes must be odd positive integers",
        ),
        (
            lambda: MKSSpatialAttention(8, kernel_sizes=(3,), dilations=(0,)),
            "dilations must be positive integers",
        ),
        (lambda: MKSBlock(8, order="bad"), "order must be one of: ca_sa, sa_ca"),
    ],
)
def test_mksnet_v2_validates_arguments(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
```

- [ ] **Step 2: Run core tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_mksnet_v2.py -q
```

Expected: FAIL with `ModuleNotFoundError` or import errors for `xh_detect.models.mksnet_v2`.

- [ ] **Step 3: Implement validation helpers and channel attention**

Create `src/xh_detect/models/mksnet_v2.py` with this first block:

```python
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _kernel_sizes(value: Sequence[int]) -> tuple[int, ...]:
    kernels = tuple(value)
    if not kernels:
        raise ValueError("kernel_sizes must contain at least one kernel")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0 or item % 2 == 0
        for item in kernels
    ):
        raise ValueError("kernel sizes must be odd positive integers")
    return kernels


def _dilations(value: Sequence[int], *, expected: int) -> tuple[int, ...]:
    dilations = tuple(value)
    if len(dilations) != expected:
        raise ValueError("kernel_sizes and dilations must have the same length")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in dilations):
        raise ValueError("dilations must be positive integers")
    return dilations


def _attention_order(value: str) -> str:
    if value not in {"ca_sa", "sa_ca"}:
        raise ValueError("order must be one of: ca_sa, sa_ca")
    return value


class MKSChannelAttention(nn.Module):
    """SENet-style channel attention with average and maximum pooling paths."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        reduction = _positive_int(reduction, "reduction")
        hidden_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.gate = nn.Sigmoid()

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        avg_logits = self.shared_mlp(self.avg_pool(x))
        max_logits = self.shared_mlp(self.max_pool(x))
        return self.gate(0.5 * (avg_logits + max_logits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)
```

- [ ] **Step 4: Implement spatial attention and MKS block**

Append to `src/xh_detect/models/mksnet_v2.py`:

```python
class MKSSpatialAttention(nn.Module):
    """Multi-kernel spatial attention with adaptive branch selection."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        dilations: Sequence[int] = (1, 1, 2, 2),
        reduction: int = 16,
        branch_reduction: int = 4,
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        reduction = _positive_int(reduction, "reduction")
        branch_reduction = _positive_int(branch_reduction, "branch_reduction")
        kernels = _kernel_sizes(kernel_sizes)
        dilations_tuple = _dilations(dilations, expected=len(kernels))
        branch_channels = max(8, channels // branch_reduction)
        hidden_channels = max(1, channels // reduction)

        self.branch_count = len(kernels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        padding=dilation * (kernel // 2),
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.SiLU(inplace=True),
                )
                for kernel, dilation in zip(kernels, dilations_tuple, strict=True)
            ]
        )
        fused_channels = branch_channels * self.branch_count
        self.selection_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.branch_count, kernel_size=1, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def _branch_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [branch(x) for branch in self.branches]

    def selection_weights(self, x: torch.Tensor) -> torch.Tensor:
        features = self._branch_features(x)
        fused = torch.cat(features, dim=1)
        return torch.softmax(self.selection_gate(fused), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._branch_features(x)
        fused = torch.cat(features, dim=1)
        weights = torch.softmax(self.selection_gate(fused), dim=1)
        stacked = torch.stack(features, dim=1)
        selected = (stacked * weights[:, :, None]).sum(dim=1)
        avg_pool = fused.mean(dim=1, keepdim=True)
        max_pool = fused.amax(dim=1, keepdim=True)
        spatial = self.spatial_gate(torch.cat((avg_pool, max_pool), dim=1))
        return self.project(selected * spatial)


class MKSBlock(nn.Module):
    """MKS block combining channel and spatial attention with a residual path."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        dilations: Sequence[int] = (1, 1, 2, 2),
        reduction: int = 16,
        order: str = "ca_sa",
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        self.order = _attention_order(order)
        self.channel_attention = MKSChannelAttention(channels, reduction=reduction)
        self.spatial_attention = MKSSpatialAttention(
            channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            reduction=reduction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == "ca_sa":
            y = self.spatial_attention(self.channel_attention(x))
        else:
            y = self.channel_attention(self.spatial_attention(x))
        return x + y
```

- [ ] **Step 5: Run core tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_mksnet_v2.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit core modules**

Run:

```powershell
git add src/xh_detect/models/mksnet_v2.py tests/test_mksnet_v2.py
git commit -m "feat: add mksnet v2 attention blocks"
```

## Task 2: MKS Stages, Standalone Backbone, And Registration

**Files:**
- Modify: `src/xh_detect/models/mksnet_v2.py`
- Modify: `src/xh_detect/models/__init__.py`
- Modify: `src/xh_detect/models/ultralytics.py`
- Modify: `tests/test_mksnet_v2.py`

**Interfaces:**
- Produces: `MKSStage(channels: int, depth: int = 1, kernel_sizes: Sequence[int] = (3, 5, 7, 9), dilations: Sequence[int] = (1, 1, 2, 2), reduction: int = 16, order: str = "ca_sa") -> nn.Module`
- Produces: `MKSNetBackbone(in_channels: int = 3, channels: Sequence[int] = (64, 128, 256, 512, 768), depths: Sequence[int] = (1, 2, 2, 2), kernel_sizes: Sequence[int] = (3, 5, 7, 9), dilations: Sequence[int] = (1, 1, 2, 2), reduction: int = 16, order: str = "ca_sa") -> nn.Module`
- Produces: `register_custom_modules()` exposes `MKSChannelAttention`, `MKSSpatialAttention`, `MKSBlock`, and `MKSStage` to `ultralytics.nn.tasks`.

- [ ] **Step 1: Add failing stage, backbone, and registration tests**

Modify the import block at the top of `tests/test_mksnet_v2.py` so it contains all v2 modules and the registration helper:

```python
from xh_detect.models.ultralytics import register_custom_modules
from xh_detect.models.mksnet_v2 import (
    MKSBlock,
    MKSChannelAttention,
    MKSNetBackbone,
    MKSStage,
    MKSSpatialAttention,
)
```

Then append these tests to `tests/test_mksnet_v2.py`:

```python


def test_mks_stage_preserves_shape_and_depth() -> None:
    stage = MKSStage(16, depth=2, kernel_sizes=(3, 5), dilations=(1, 2), reduction=4)
    x = torch.randn(1, 16, 12, 12)

    y = stage(x)

    assert y.shape == x.shape
    assert len(stage.blocks) == 2


def test_mksnet_backbone_returns_p3_p4_p5_feature_maps() -> None:
    backbone = MKSNetBackbone(
        channels=(16, 32, 64, 128, 192),
        depths=(1, 1, 1, 1),
        kernel_sizes=(3, 5),
        dilations=(1, 2),
        reduction=4,
    )
    x = torch.randn(1, 3, 128, 128)

    p3, p4, p5 = backbone(x)

    assert p3.shape == (1, 64, 16, 16)
    assert p4.shape == (1, 128, 8, 8)
    assert p5.shape == (1, 192, 4, 4)


def test_mksnet_backbone_validates_channel_and_depth_lengths() -> None:
    with pytest.raises(ValueError, match="channels must contain five stage widths"):
        MKSNetBackbone(channels=(16, 32, 64, 128))
    with pytest.raises(ValueError, match="depths must contain four stage depths"):
        MKSNetBackbone(depths=(1, 1, 1))


def test_register_custom_modules_exposes_mksnet_v2_to_ultralytics() -> None:
    import ultralytics.nn.tasks as tasks

    names = ("MKSChannelAttention", "MKSSpatialAttention", "MKSBlock", "MKSStage")
    originals = {name: getattr(tasks, name, None) for name in names}
    for name in names:
        if hasattr(tasks, name):
            delattr(tasks, name)
    try:
        register_custom_modules()

        assert tasks.MKSChannelAttention is MKSChannelAttention
        assert tasks.MKSSpatialAttention is MKSSpatialAttention
        assert tasks.MKSBlock is MKSBlock
        assert tasks.MKSStage is MKSStage
    finally:
        for name, original in originals.items():
            if original is not None:
                setattr(tasks, name, original)
            elif hasattr(tasks, name):
                delattr(tasks, name)
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_mksnet_v2.py -q
```

Expected: FAIL because `MKSStage`, `MKSNetBackbone`, and registration exports are missing.

- [ ] **Step 3: Add stage and backbone helpers**

Append to `src/xh_detect/models/mksnet_v2.py`:

```python
class _ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MKSStage(nn.Module):
    """A repeated stack of channel-preserving MKS blocks."""

    def __init__(
        self,
        channels: int,
        depth: int = 1,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        dilations: Sequence[int] = (1, 1, 2, 2),
        reduction: int = 16,
        order: str = "ca_sa",
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        depth = _positive_int(depth, "depth")
        self.blocks = nn.Sequential(
            *[
                MKSBlock(
                    channels,
                    kernel_sizes=kernel_sizes,
                    dilations=dilations,
                    reduction=reduction,
                    order=order,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MKSNetBackbone(nn.Module):
    """Standalone MKSNet-style backbone that returns P3, P4, and P5 maps."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 512, 768),
        depths: Sequence[int] = (1, 2, 2, 2),
        kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        dilations: Sequence[int] = (1, 1, 2, 2),
        reduction: int = 16,
        order: str = "ca_sa",
    ) -> None:
        super().__init__()
        in_channels = _positive_int(in_channels, "in_channels")
        widths = tuple(channels)
        if len(widths) != 5:
            raise ValueError("channels must contain five stage widths")
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in widths):
            raise ValueError("channels must contain positive integers")
        stage_depths = tuple(depths)
        if len(stage_depths) != 4:
            raise ValueError("depths must contain four stage depths")
        for index, depth in enumerate(stage_depths):
            _positive_int(depth, f"depths[{index}]")

        self.stem = _ConvBNAct(in_channels, widths[0], stride=2)
        self.stage1 = nn.Sequential(
            _ConvBNAct(widths[0], widths[1], stride=2),
            MKSStage(widths[1], stage_depths[0], kernel_sizes, dilations, reduction, order),
        )
        self.stage2 = nn.Sequential(
            _ConvBNAct(widths[1], widths[2], stride=2),
            MKSStage(widths[2], stage_depths[1], kernel_sizes, dilations, reduction, order),
        )
        self.stage3 = nn.Sequential(
            _ConvBNAct(widths[2], widths[3], stride=2),
            MKSStage(widths[3], stage_depths[2], kernel_sizes, dilations, reduction, order),
        )
        self.stage4 = nn.Sequential(
            _ConvBNAct(widths[3], widths[4], stride=2),
            MKSStage(widths[4], stage_depths[3], kernel_sizes, dilations, reduction, order),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5
```

- [ ] **Step 4: Export modules from package**

Replace `src/xh_detect/models/__init__.py` with:

```python
from xh_detect.models.mksnet_lite import MKSNetLiteBlock
from xh_detect.models.mksnet_v2 import (
    MKSBlock,
    MKSChannelAttention,
    MKSNetBackbone,
    MKSStage,
    MKSSpatialAttention,
)

__all__ = [
    "MKSBlock",
    "MKSChannelAttention",
    "MKSNetBackbone",
    "MKSNetLiteBlock",
    "MKSStage",
    "MKSSpatialAttention",
]
```

- [ ] **Step 5: Register MKSNet v2 modules with Ultralytics**

Replace `src/xh_detect/models/ultralytics.py` with:

```python
from __future__ import annotations

from xh_detect.models.mksnet_lite import MKSNetLiteBlock
from xh_detect.models.mksnet_v2 import (
    MKSBlock,
    MKSChannelAttention,
    MKSStage,
    MKSSpatialAttention,
)


def register_custom_modules() -> None:
    """Expose custom modules to Ultralytics YAML parsing and checkpoint loading."""
    import ultralytics.nn.tasks as tasks

    tasks.MKSNetLiteBlock = MKSNetLiteBlock
    tasks.MKSChannelAttention = MKSChannelAttention
    tasks.MKSSpatialAttention = MKSSpatialAttention
    tasks.MKSBlock = MKSBlock
    tasks.MKSStage = MKSStage
```

- [ ] **Step 6: Run focused tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_mksnet_v2.py tests/test_mksnet_lite.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit stage, backbone, and registration work**

Run:

```powershell
git add src/xh_detect/models/__init__.py src/xh_detect/models/mksnet_v2.py src/xh_detect/models/ultralytics.py tests/test_mksnet_v2.py
git commit -m "feat: add mksnet v2 backbone registration"
```

## Task 3: Full MKSNet Experiment Configs

**Files:**
- Create: `configs/models/xh25-yolo-mksnet-v2-full.yaml`
- Create: `configs/xh25-mksnet-v2-full.yaml`
- Modify: `tests/test_mksnet_configs.py`

**Interfaces:**
- Consumes: `MKSStage` registered in Ultralytics as a shape-preserving module.
- Produces: model YAML with four `MKSStage` layers and final `Detect` from `[15, 18, 21]`.
- Produces: pipeline config with `model_path: runs/train/xh25-mksnet-v2-full-vehicle/weights/best.pt`.

- [ ] **Step 1: Add failing config tests**

Append to `tests/test_mksnet_configs.py`:

```python

def test_mksnet_v2_full_model_yaml_contains_mks_stages() -> None:
    path = Path("configs/models/xh25-yolo-mksnet-v2-full.yaml")
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    layers = model["backbone"] + model["head"]

    custom_layers = [layer for layer in layers if layer[2] == "MKSStage"]

    assert model["nc"] == 25
    assert model["scale"] == "s"
    assert model["end2end"] is True
    assert model["reg_max"] == 1
    assert len(custom_layers) == 4
    assert custom_layers[0] == [-1, 1, "MKSStage", [128, 1, [3, 5, 7, 9], [1, 1, 2, 2], 16, "ca_sa"]]
    assert custom_layers[1] == [-1, 1, "MKSStage", [256, 2, [3, 5, 7, 9], [1, 1, 2, 2], 16, "ca_sa"]]
    assert custom_layers[2] == [-1, 1, "MKSStage", [512, 2, [3, 5, 7, 9], [1, 1, 2, 2], 16, "ca_sa"]]
    assert custom_layers[3] == [-1, 1, "MKSStage", [768, 2, [3, 5, 7, 9], [1, 1, 2, 2], 16, "ca_sa"]]
    assert layers[-1] == [[15, 18, 21], 1, "Detect", ["nc"]]


def test_mksnet_v2_full_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-mksnet-v2-full.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-mksnet-v2-full-vehicle/weights/best.pt"
    assert config.image_size == 1024
    assert config.batch_size == 8
    assert set(config.class_thresholds) == set(range(25))
```

- [ ] **Step 2: Run config tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_mksnet_configs.py -q
```

Expected: FAIL because `configs/models/xh25-yolo-mksnet-v2-full.yaml` and `configs/xh25-mksnet-v2-full.yaml` do not exist.

- [ ] **Step 3: Create full MKSNet model YAML**

Create `configs/models/xh25-yolo-mksnet-v2-full.yaml`:

```yaml
# XH25 HBB detector with a MKSNet-style full backbone and YOLO detection head.
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
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 1, MKSStage, [128, 1, [3, 5, 7, 9], [1, 1, 2, 2], 16, ca_sa]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 1, MKSStage, [256, 2, [3, 5, 7, 9], [1, 1, 2, 2], 16, ca_sa]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 1, MKSStage, [512, 2, [3, 5, 7, 9], [1, 1, 2, 2], 16, ca_sa]]
  - [-1, 1, Conv, [1536, 3, 2]]
  - [-1, 1, MKSStage, [768, 2, [3, 5, 7, 9], [1, 1, 2, 2], 16, ca_sa]]
  - [-1, 1, SPPF, [1536, 5, 3, true]]

head:
  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, true]]

  - [-1, 1, nn.Upsample, [null, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, true]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, true]]

  - [-1, 1, Conv, [1024, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 1, C3k2, [1536, true, 0.5, true]]

  - [[15, 18, 21], 1, Detect, [nc]]
```

- [ ] **Step 4: Create full MKSNet inference config**

Create `configs/xh25-mksnet-v2-full.yaml`:

```yaml
task: detect
taxonomy: xh25
model_path: runs/train/xh25-mksnet-v2-full-vehicle/weights/best.pt
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

- [ ] **Step 5: Run config tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_mksnet_configs.py -q
```

Expected: PASS.

- [ ] **Step 6: Smoke-load the full model YAML**

Run:

```powershell
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); model=YOLO('configs/models/xh25-yolo-mksnet-v2-full.yaml'); print(model.model.__class__.__name__)"
```

Expected: command exits 0 and prints a model class name such as `DetectionModel`.

- [ ] **Step 7: Commit full config work**

Run:

```powershell
git add configs/models/xh25-yolo-mksnet-v2-full.yaml configs/xh25-mksnet-v2-full.yaml tests/test_mksnet_configs.py
git commit -m "config: add mksnet v2 full experiment"
```

## Task 4: Experiment Runbook And Local Verification

**Files:**
- Create: `docs/experiments/mksnet-v2-full-vehicle.md`

**Interfaces:**
- Consumes: full model YAML and inference config from Task 3.
- Produces: a runbook with exact local smoke, server training, raw evaluation, competition proxy, and threshold calibration commands.

- [ ] **Step 1: Create the experiment runbook**

Create `docs/experiments/mksnet-v2-full-vehicle.md`:

````markdown
# MKSNet-v2-full Vehicle Experiment

This experiment implements a MKSNet-style full backbone adapted to the XH25 YOLO HBB detection pipeline. It is compared against `xh25-yolo26s-e80` and the previous `xh25-mksnet-lite` run.

## Local Smoke

```bash
python -m pytest tests/test_mksnet_v2.py tests/test_mksnet_configs.py -q
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); model=YOLO('configs/models/xh25-yolo-mksnet-v2-full.yaml'); print(model.model.__class__.__name__)"
```

## Training

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model configs/models/xh25-yolo-mksnet-v2-full.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-mksnet-v2-full-vehicle \
  --no-resume
```

## Raw Evaluation

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-mksnet-v2-full.yaml \
  --output-json outputs/xh25/mksnet-v2-full-vehicle/val-predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/mksnet-v2-full-vehicle/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-path outputs/xh25/mksnet-v2-full-vehicle/report.json \
  --taxonomy xh25

.venv/bin/xh-detect competition-report \
  --report-json outputs/xh25/mksnet-v2-full-vehicle/report.json \
  --output-dir outputs/xh25/mksnet-v2-full-vehicle/competition-proxy \
  --experiment-name xh25-mksnet-v2-full-vehicle
```

## Threshold Calibration

```bash
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/mksnet-v2-full-vehicle/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-dir outputs/xh25/mksnet-v2-full-vehicle/threshold-optimized \
  --taxonomy xh25 \
  --baseline-report outputs/xh25/baseline/report.json \
  --experiment-name xh25-mksnet-v2-full-vehicle-threshold-optimized
```

## Result Table

| Candidate | Overall Recall | Overall FDR | Ship Recall | Ship FDR | Vehicle Recall | Vehicle FDR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main / xh25-yolo26s-e80 | 0.961562 | 0.037244 | 0.823383 | 0.157761 | 0.705128 | 0.202899 |
| MKSNet-Lite thresholded | 0.958772 | 0.029190 | 0.800995 | 0.154856 | 0.692308 | 0.129032 |

After raw and calibrated evaluation complete, copy the six printed metrics from Task 5 Step 7 into a dated result note under `outputs/xh25/mksnet-v2-full-vehicle/`.

## Keep Criteria

- Overall Recall >= 0.85.
- Overall FDR <= 0.20.
- Vehicle Recall >= 0.735128, which is main vehicle recall plus 0.03.
- Raw Vehicle FDR <= 0.25.
- Ship Recall >= 0.803383, which is main ship recall minus 0.02.
- Aircraft Recall drop <= 0.005 versus main.
````

- [ ] **Step 2: Run local verification**

Run:

```powershell
python -m pytest tests/test_mksnet_v2.py tests/test_mksnet_lite.py tests/test_mksnet_configs.py -q
python -m ruff format --check src/xh_detect/models tests/test_mksnet_v2.py tests/test_mksnet_configs.py
python -m ruff check src/xh_detect/models tests/test_mksnet_v2.py tests/test_mksnet_configs.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Commit runbook**

Run:

```powershell
git add docs/experiments/mksnet-v2-full-vehicle.md
git commit -m "docs: add mksnet v2 full runbook"
```

## Task 5: Server Training And Evaluation

**Files:**
- No source edits expected.
- Produce artifacts under `outputs/xh25/mksnet-v2-full-vehicle/` on the server.
- Download compact reports into the same path locally if training completes during this work session.

**Interfaces:**
- Consumes: committed branch `codex/mksnet-lite`.
- Produces: raw and calibrated evaluation artifacts for `xh25-mksnet-v2-full-vehicle`.

- [ ] **Step 1: Push local branch**

Run locally:

```powershell
git status --short
git push origin codex/mksnet-lite
```

Expected: status is clean before push. Push exits 0.

- [ ] **Step 2: Sync server branch and configure mirror**

Run on the server:

```bash
ssh -p 2222 root@ssh.zw1.paratera.com
cd /root/XH-202625-remote-sensing-detection
git fetch origin
git checkout codex/mksnet-lite
git pull --ff-only origin codex/mksnet-lite
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

Expected: server worktree is on `codex/mksnet-lite` with the latest commits. Pip mirror is set to Tsinghua.

- [ ] **Step 3: Server environment sync and smoke construction**

Run on the server:

```bash
cd /root/XH-202625-remote-sensing-detection
python -m pip install -e ".[dev]"
python -m pytest tests/test_mksnet_v2.py tests/test_mksnet_configs.py -q
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); model=YOLO('configs/models/xh25-yolo-mksnet-v2-full.yaml'); print(model.model.__class__.__name__)"
nvidia-smi
```

Expected: tests pass, model construction exits 0, and `nvidia-smi` shows one RTX 3090.

- [ ] **Step 4: Start the 80-epoch server training run**

Run on the server:

```bash
cd /root/XH-202625-remote-sensing-detection
nohup .venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model configs/models/xh25-yolo-mksnet-v2-full.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-mksnet-v2-full-vehicle \
  --no-resume \
  > logs/xh25-mksnet-v2-full-vehicle.log 2>&1 &
tail -f logs/xh25-mksnet-v2-full-vehicle.log
```

Expected: training starts and prints transferred-weight information, epoch progress, loss values, and validation metrics.

- [ ] **Step 5: Evaluate raw model after training**

Run on the server after `runs/train/xh25-mksnet-v2-full-vehicle/weights/best.pt` exists:

```bash
cd /root/XH-202625-remote-sensing-detection
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-mksnet-v2-full.yaml \
  --output-json outputs/xh25/mksnet-v2-full-vehicle/val-predictions.json
.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/mksnet-v2-full-vehicle/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-path outputs/xh25/mksnet-v2-full-vehicle/report.json \
  --taxonomy xh25
.venv/bin/xh-detect competition-report \
  --report-json outputs/xh25/mksnet-v2-full-vehicle/report.json \
  --output-dir outputs/xh25/mksnet-v2-full-vehicle/competition-proxy \
  --experiment-name xh25-mksnet-v2-full-vehicle
```

Expected: raw `report.json` and `competition-proxy.json` are written.

- [ ] **Step 6: Run threshold calibration as a separate candidate**

Run on the server:

```bash
cd /root/XH-202625-remote-sensing-detection
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/mksnet-v2-full-vehicle/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-dir outputs/xh25/mksnet-v2-full-vehicle/threshold-optimized \
  --taxonomy xh25 \
  --baseline-report outputs/xh25/baseline/report.json \
  --experiment-name xh25-mksnet-v2-full-vehicle-threshold-optimized
```

Expected: calibrated `report.json`, optimized config, and threshold summary are written under `threshold-optimized`.

- [ ] **Step 7: Summarize and download compact artifacts**

Run on the server:

```bash
cd /root/XH-202625-remote-sensing-detection
python - <<'PY'
import json
from pathlib import Path

paths = {
    "raw": Path("outputs/xh25/mksnet-v2-full-vehicle/report.json"),
    "calibrated": Path("outputs/xh25/mksnet-v2-full-vehicle/threshold-optimized/report.json"),
}
for name, path in paths.items():
    report = json.loads(path.read_text())
    overall = report["overall_class_agnostic"]
    vehicle = report["by_coarse_class"]["vehicle"]
    ship = report["by_coarse_class"]["ship"]
    print(name, {
        "overall_recall": overall["recall"],
        "overall_fdr": overall["fdr"],
        "ship_recall": ship["recall"],
        "ship_fdr": ship["fdr"],
        "vehicle_recall": vehicle["recall"],
        "vehicle_fdr": vehicle["fdr"],
    })
PY
```

Run locally:

```powershell
New-Item -ItemType Directory -Force outputs/xh25/mksnet-v2-full-vehicle
scp -P 2222 root@ssh.zw1.paratera.com:/root/XH-202625-remote-sensing-detection/outputs/xh25/mksnet-v2-full-vehicle/report.json outputs/xh25/mksnet-v2-full-vehicle/report.json
scp -P 2222 root@ssh.zw1.paratera.com:/root/XH-202625-remote-sensing-detection/outputs/xh25/mksnet-v2-full-vehicle/competition-proxy/competition-proxy.json outputs/xh25/mksnet-v2-full-vehicle/competition-proxy.json
scp -P 2222 root@ssh.zw1.paratera.com:/root/XH-202625-remote-sensing-detection/outputs/xh25/mksnet-v2-full-vehicle/threshold-optimized/report.json outputs/xh25/mksnet-v2-full-vehicle/threshold-optimized-report.json
```

Expected: console prints raw and calibrated metrics; compact report files exist locally.

## Self-Review

- Spec coverage: Task 1 implements CA, SA, adaptive branch selection, and MKS blocks. Task 2 implements repeated MKS stages, standalone backbone, and Ultralytics registration. Task 3 creates the full experiment YAML and pipeline config. Task 4 documents the workflow. Task 5 covers server training, raw evaluation, competition proxy, threshold calibration, and artifact download.
- Scope check: The plan implements one experiment, `MKSNet-v2-full`, and leaves P2 heads, balanced sampling, and threshold-only optimization as separate follow-up factors.
- Placeholder scan: The plan contains no unfinished markers, future-value table cells, or unspecified test-writing steps.
- Type consistency: Module names are consistently `MKSChannelAttention`, `MKSSpatialAttention`, `MKSBlock`, `MKSStage`, and `MKSNetBackbone`; the config name is consistently `xh25-yolo-mksnet-v2-full.yaml`; the run name is consistently `xh25-mksnet-v2-full-vehicle`.
