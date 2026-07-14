# MKSNet-Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-stage `xh25-mksnet-lite` YOLO experiment with a lightweight multi-kernel dual-attention module, reusable configs, and baseline comparison reporting.

**Architecture:** Keep the existing Ultralytics training, tiled inference, evaluation, and benchmark pipeline. Add a focused PyTorch module, register it with Ultralytics YAML parsing before model load, use a custom model YAML for the experiment, and compare the resulting metrics against the existing XH25 baseline.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics 8.4.x, Typer, PyYAML, pytest, existing `xh_detect` CLI and evaluation modules.

---

## File Structure

- Create `src/xh_detect/models/__init__.py`: package marker and public exports for custom model modules.
- Create `src/xh_detect/models/mksnet_lite.py`: `MKSNetLiteBlock`, a channel-preserving multi-kernel depthwise convolution block with channel and spatial attention.
- Create `src/xh_detect/models/ultralytics.py`: idempotent registration hook that exposes custom modules to `ultralytics.nn.tasks.parse_model`.
- Create `tests/test_mksnet_lite.py`: unit tests for shape preservation, gradients, validation, and Ultralytics registration.
- Modify `src/xh_detect/training.py`: register custom modules before constructing `YOLO`; add optional pretrained warm start.
- Modify `src/xh_detect/detector.py`: register custom modules before inference model load.
- Modify `src/xh_detect/cli.py`: expose `--pretrained` on `xh-detect train`.
- Modify `tests/test_training.py`, `tests/test_detector.py`, and `tests/test_cli.py`: cover registration and pretrained forwarding.
- Create `configs/models/xh25-mksnet-lite.yaml`: Ultralytics-compatible YOLO26s-style detect model with MKSNet-Lite blocks inserted in the neck.
- Create `configs/xh25-mksnet-lite.yaml`: inference config pointing to `runs/train/xh25-mksnet-lite/weights/best.pt`.
- Create `tests/test_mksnet_configs.py`: static validation for the model YAML and pipeline config.
- Create `src/xh_detect/compare.py`: compare baseline and experiment evaluation reports plus optional benchmark summaries.
- Modify `src/xh_detect/cli.py`: add `compare-experiments`.
- Create `tests/test_compare.py`: unit tests for comparison output.
- Modify `README.md`: document the MKSNet-Lite experiment commands and comparison workflow.

## Task 1: MKSNet-Lite Module And Ultralytics Registration

**Files:**
- Create: `src/xh_detect/models/__init__.py`
- Create: `src/xh_detect/models/mksnet_lite.py`
- Create: `src/xh_detect/models/ultralytics.py`
- Create: `tests/test_mksnet_lite.py`

- [ ] **Step 1: Write failing module tests**

Create `tests/test_mksnet_lite.py`:

```python
from __future__ import annotations

import pytest
import torch

from xh_detect.models.mksnet_lite import MKSNetLiteBlock
from xh_detect.models.ultralytics import register_custom_modules


def test_mksnet_lite_block_preserves_shape_and_allows_gradients() -> None:
    torch.manual_seed(7)
    block = MKSNetLiteBlock(16, kernel_sizes=(3, 5, 7), reduction=4)
    x = torch.randn(2, 16, 24, 32, requires_grad=True)

    y = block(x)
    loss = y.mean()
    loss.backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mksnet_lite_block_supports_single_kernel() -> None:
    block = MKSNetLiteBlock(8, kernel_sizes=(3,), reduction=4)
    x = torch.randn(1, 8, 10, 10)

    assert block(x).shape == x.shape


@pytest.mark.parametrize(
    ("channels", "kernel_sizes", "reduction", "message"),
    [
        (0, (3, 5), 16, "channels must be a positive integer"),
        (8, (), 16, "kernel_sizes must contain at least one kernel"),
        (8, (2, 3), 16, "kernel sizes must be odd positive integers"),
        (8, (3,), 0, "reduction must be a positive integer"),
    ],
)
def test_mksnet_lite_block_validates_arguments(
    channels: int,
    kernel_sizes: tuple[int, ...],
    reduction: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MKSNetLiteBlock(channels, kernel_sizes=kernel_sizes, reduction=reduction)


def test_register_custom_modules_exposes_block_to_ultralytics() -> None:
    import ultralytics.nn.tasks as tasks

    original = getattr(tasks, "MKSNetLiteBlock", None)
    if hasattr(tasks, "MKSNetLiteBlock"):
        delattr(tasks, "MKSNetLiteBlock")
    try:
        register_custom_modules()

        assert tasks.MKSNetLiteBlock is MKSNetLiteBlock
    finally:
        if original is not None:
            tasks.MKSNetLiteBlock = original
        elif hasattr(tasks, "MKSNetLiteBlock"):
            delattr(tasks, "MKSNetLiteBlock")
```

- [ ] **Step 2: Run module tests and confirm red**

Run:

```powershell
python -m pytest tests/test_mksnet_lite.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'xh_detect.models'`.

- [ ] **Step 3: Add the models package**

Create `src/xh_detect/models/__init__.py`:

```python
from xh_detect.models.mksnet_lite import MKSNetLiteBlock

__all__ = ["MKSNetLiteBlock"]
```

- [ ] **Step 4: Implement MKSNet-Lite block**

Create `src/xh_detect/models/mksnet_lite.py`:

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
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 or item % 2 == 0 for item in kernels):
        raise ValueError("kernel sizes must be odd positive integers")
    return kernels


class MKSNetLiteBlock(nn.Module):
    """Channel-preserving multi-kernel spatial/channel attention block."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        reduction: int = 16,
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        reduction = _positive_int(reduction, "reduction")
        kernels = _kernel_sizes(kernel_sizes)
        hidden_channels = max(1, channels // reduction)

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=channels,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = torch.stack([branch(x) for branch in self.branches], dim=0).mean(dim=0)
        fused = self.activation(self.norm(self.project(fused)))
        fused = fused * self.channel_attention(fused)
        avg_pool = fused.mean(dim=1, keepdim=True)
        max_pool = fused.amax(dim=1, keepdim=True)
        spatial_gate = self.spatial_attention(torch.cat((avg_pool, max_pool), dim=1))
        return x + fused * spatial_gate
```

- [ ] **Step 5: Implement Ultralytics registration**

Create `src/xh_detect/models/ultralytics.py`:

```python
from __future__ import annotations

from xh_detect.models.mksnet_lite import MKSNetLiteBlock


def register_custom_modules() -> None:
    """Expose custom modules to Ultralytics YAML parsing and checkpoint loading."""
    import ultralytics.nn.tasks as tasks

    tasks.MKSNetLiteBlock = MKSNetLiteBlock
```

- [ ] **Step 6: Run module tests and confirm green**

Run:

```powershell
python -m pytest tests/test_mksnet_lite.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit module work**

Run:

```powershell
git add src/xh_detect/models tests/test_mksnet_lite.py
git commit -m "feat: add mksnet lite module"
```

## Task 2: Register Custom Modules In Training And Inference

**Files:**
- Modify: `src/xh_detect/training.py`
- Modify: `src/xh_detect/detector.py`
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_training.py`
- Modify: `tests/test_detector.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing training tests**

Append to `tests/test_training.py`:

```python
@patch("xh_detect.training.register_custom_modules")
@patch("xh_detect.training.YOLO")
def test_train_model_registers_custom_modules_before_model_load(
    yolo_class: Mock,
    register_custom_modules: Mock,
) -> None:
    events: list[str] = []
    register_custom_modules.side_effect = lambda: events.append("register")
    yolo_class.side_effect = lambda model_path: events.append(f"yolo:{model_path}") or Mock()

    train_model("dataset.yaml", "configs/models/xh25-mksnet-lite.yaml", 1, 640, "cpu")

    assert events[:2] == ["register", "yolo:configs/models/xh25-mksnet-lite.yaml"]


@patch("xh_detect.training.register_custom_modules")
@patch("xh_detect.training.YOLO")
def test_train_model_loads_optional_pretrained_weights(
    yolo_class: Mock,
    register_custom_modules: Mock,
) -> None:
    model = yolo_class.return_value
    model.load.return_value = model

    train_model(
        "dataset.yaml",
        "configs/models/xh25-mksnet-lite.yaml",
        1,
        640,
        "cpu",
        pretrained="yolo26s.pt",
    )

    register_custom_modules.assert_called_once_with()
    yolo_class.assert_called_once_with("configs/models/xh25-mksnet-lite.yaml")
    model.load.assert_called_once_with("yolo26s.pt")
    model.train.assert_called_once()
```

- [ ] **Step 2: Add failing detector registration test**

Append to `tests/test_detector.py`:

```python
def test_ultralytics_detector_registers_custom_modules_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect import detector as detector_module

    events: list[str] = []
    monkeypatch.setattr(
        detector_module,
        "register_custom_modules",
        lambda: events.append("register"),
    )
    monkeypatch.setattr(
        detector_module,
        "YOLO",
        lambda model_path: events.append(f"yolo:{model_path}") or FakeModel([]),
    )

    detector_module.UltralyticsDetector("weights.pt", "cpu", 640, False, task="detect")

    assert events == ["register", "yolo:weights.pt"]
```

- [ ] **Step 3: Add failing CLI pretrained test**

Append to `tests/test_cli.py` after `test_train_command_forwards_reproducible_options`:

```python
@patch("xh_detect.cli.train_model")
def test_train_command_forwards_pretrained_option(
    train_model: Mock,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--dataset-yaml",
            str(dataset),
            "--model",
            "configs/models/xh25-mksnet-lite.yaml",
            "--pretrained",
            "yolo26s.pt",
            "--epochs",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    train_model.assert_called_once_with(
        str(dataset),
        "configs/models/xh25-mksnet-lite.yaml",
        2,
        1024,
        "0",
        batch=8,
        workers=4,
        amp=False,
        project="runs/train",
        name="xh25-baseline",
        resume=False,
        pretrained="yolo26s.pt",
    )
```

- [ ] **Step 4: Run focused tests and confirm red**

Run:

```powershell
python -m pytest tests/test_training.py::test_train_model_registers_custom_modules_before_model_load tests/test_training.py::test_train_model_loads_optional_pretrained_weights tests/test_detector.py::test_ultralytics_detector_registers_custom_modules_before_model_load tests/test_cli.py::test_train_command_forwards_pretrained_option -q
```

Expected: FAIL because `register_custom_modules` and `pretrained` are not wired yet.

- [ ] **Step 5: Update training wrapper**

In `src/xh_detect/training.py`, add this import:

```python
from xh_detect.models.ultralytics import register_custom_modules
```

Change the `train_model` signature to include the keyword argument:

```python
    pretrained: str | None = None,
) -> None:
```

Add validation and load logic immediately before `model.train(...)`:

```python
    pretrained_model = None if pretrained is None else _non_empty(pretrained, "pretrained")

    register_custom_modules()
    model = YOLO(model_source)
    if pretrained_model is not None:
        model = model.load(pretrained_model)
```

Remove the old line:

```python
    model = YOLO(model_source)
```

- [ ] **Step 6: Update detector model loading**

In `src/xh_detect/detector.py`, add this import:

```python
from xh_detect.models.ultralytics import register_custom_modules
```

In `UltralyticsDetector.__init__`, replace:

```python
        self.model = YOLO(validated_model_path)
```

with:

```python
        register_custom_modules()
        self.model = YOLO(validated_model_path)
```

- [ ] **Step 7: Update CLI train command**

In `src/xh_detect/cli.py`, add this parameter to `train(...)` after `model`:

```python
    pretrained: Annotated[str | None, typer.Option()] = None,
```

Pass it through to `train_model(...)`:

```python
        pretrained=pretrained,
```

- [ ] **Step 8: Update existing training and CLI test expectations**

In `tests/test_training.py`, each `model.train.assert_called_once_with(...)` should remain unchanged. The new `pretrained` argument is only passed to `train_model`, not to `model.train`.

In `tests/test_cli.py`, update the existing `test_train_command_calls_wrapper` expected call to include:

```python
        pretrained=None,
```

Update `test_train_command_forwards_reproducible_options` expected call to include:

```python
        pretrained=None,
```

- [ ] **Step 9: Run focused tests and confirm green**

Run:

```powershell
python -m pytest tests/test_training.py tests/test_detector.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit training/inference registration work**

Run:

```powershell
git add src/xh_detect/training.py src/xh_detect/detector.py src/xh_detect/cli.py tests/test_training.py tests/test_detector.py tests/test_cli.py
git commit -m "feat: register custom yolo modules"
```

## Task 3: Experiment Model YAML And Inference Config

**Files:**
- Create: `configs/models/xh25-mksnet-lite.yaml`
- Create: `configs/xh25-mksnet-lite.yaml`
- Create: `tests/test_mksnet_configs.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_mksnet_configs.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from xh_detect.config import PipelineConfig


def test_mksnet_lite_model_yaml_contains_custom_blocks() -> None:
    path = Path("configs/models/xh25-mksnet-lite.yaml")
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    layers = model["backbone"] + model["head"]

    custom_layers = [layer for layer in layers if layer[2] == "MKSNetLiteBlock"]

    assert model["nc"] == 25
    assert model["scale"] == "s"
    assert len(custom_layers) == 2
    assert custom_layers[0] == [-1, 1, "MKSNetLiteBlock", [128]]
    assert custom_layers[1] == [-1, 1, "MKSNetLiteBlock", [256]]
    assert layers[-1] == [[17, 21, 24], 1, "Detect", ["nc"]]


def test_mksnet_lite_pipeline_config_loads() -> None:
    config = PipelineConfig.from_yaml("configs/xh25-mksnet-lite.yaml")

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-mksnet-lite/weights/best.pt"
    assert config.image_size == 1024
    assert config.batch_size == 8
    assert set(config.class_thresholds) == set(range(25))
```

- [ ] **Step 2: Run config tests and confirm red**

Run:

```powershell
python -m pytest tests/test_mksnet_configs.py -q
```

Expected: FAIL because the new config files do not exist.

- [ ] **Step 3: Create MKSNet-Lite model YAML**

Create `configs/models/xh25-mksnet-lite.yaml`:

```yaml
# XH25 YOLO26s-style HBB detector with lightweight MKSNet-inspired neck blocks.
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
  - [-1, 1, MKSNetLiteBlock, [128]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, true]]
  - [-1, 1, MKSNetLiteBlock, [256]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, true, 0.5, true]]

  - [[17, 21, 24], 1, Detect, [nc]]
```

- [ ] **Step 4: Create MKSNet-Lite inference config**

Create `configs/xh25-mksnet-lite.yaml`:

```yaml
task: detect
taxonomy: xh25
model_path: runs/train/xh25-mksnet-lite/weights/best.pt
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

- [ ] **Step 5: Run config tests and confirm green**

Run:

```powershell
python -m pytest tests/test_mksnet_configs.py -q
```

Expected: PASS.

- [ ] **Step 6: Smoke-load the model YAML without training**

Run:

```powershell
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); model=YOLO('configs/models/xh25-mksnet-lite.yaml'); print(model.model.__class__.__name__)"
```

Expected: command exits 0 and prints a model class name such as `DetectionModel`.

- [ ] **Step 7: Commit configs**

Run:

```powershell
git add configs/models/xh25-mksnet-lite.yaml configs/xh25-mksnet-lite.yaml tests/test_mksnet_configs.py
git commit -m "config: add mksnet lite experiment"
```

## Task 4: Baseline-Vs-Experiment Comparison Report

**Files:**
- Create: `src/xh_detect/compare.py`
- Modify: `src/xh_detect/cli.py`
- Create: `tests/test_compare.py`

- [ ] **Step 1: Write failing comparison tests**

Create `tests/test_compare.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from xh_detect.compare import compare_experiments


def _report(tp: int, fp: int, fn: int) -> dict[str, object]:
    return {
        "overall_class_agnostic": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": tp / (tp + fn),
            "fdr": fp / (fp + tp),
        },
        "by_coarse_class": {
            "ship": {"tp": 2, "fp": 1, "fn": 3, "recall": 0.4, "fdr": 1 / 3},
            "aircraft": {"tp": 5, "fp": 2, "fn": 1, "recall": 5 / 6, "fdr": 2 / 7},
            "vehicle": {"tp": 1, "fp": 3, "fn": 4, "recall": 0.2, "fdr": 0.75},
        },
        "by_fine_class": {
            "0": {"tp": 1, "fp": 0, "fn": 1, "recall": 0.5, "fdr": 0.0},
            "1": {"tp": 0, "fp": 1, "fn": 2, "recall": 0.0, "fdr": 1.0},
            "24": {"tp": 1, "fp": 3, "fn": 4, "recall": 0.2, "fdr": 0.75},
        },
        "by_image": {},
    }


def test_compare_experiments_writes_json_and_markdown(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline-report.json"
    experiment_report = tmp_path / "experiment-report.json"
    baseline_benchmark = tmp_path / "baseline-benchmark.json"
    experiment_benchmark = tmp_path / "experiment-benchmark.json"
    output_dir = tmp_path / "comparison"
    baseline_report.write_text(json.dumps(_report(10, 5, 10)), encoding="utf-8")
    experiment_report.write_text(json.dumps(_report(12, 6, 8)), encoding="utf-8")
    baseline_benchmark.write_text(json.dumps({"median_s": 10.0, "p95_s": 12.0}), encoding="utf-8")
    experiment_benchmark.write_text(json.dumps({"median_s": 11.0, "p95_s": 13.5}), encoding="utf-8")

    comparison = compare_experiments(
        baseline_report=baseline_report,
        experiment_report=experiment_report,
        output_dir=output_dir,
        baseline_name="xh25-yolo26s-e80",
        experiment_name="xh25-mksnet-lite",
        baseline_benchmark=baseline_benchmark,
        experiment_benchmark=experiment_benchmark,
    )

    saved = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "comparison.md").read_text(encoding="utf-8")

    assert comparison["overall"]["recall_delta"] == 0.1
    assert saved["overall"]["experiment_recall"] == 0.6
    assert saved["benchmark"]["median_s_delta"] == 1.0
    assert "xh25-mksnet-lite" in markdown
    assert "| vehicle |" in markdown
```

- [ ] **Step 2: Run comparison tests and confirm red**

Run:

```powershell
python -m pytest tests/test_compare.py -q
```

Expected: FAIL because `xh_detect.compare` does not exist.

- [ ] **Step 3: Implement comparison module**

Create `src/xh_detect/compare.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _metric_block(
    baseline: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, float | int]:
    baseline_recall = float(baseline["recall"])
    experiment_recall = float(experiment["recall"])
    baseline_fdr = float(baseline["fdr"])
    experiment_fdr = float(experiment["fdr"])
    return {
        "baseline_tp": int(baseline["tp"]),
        "baseline_fp": int(baseline["fp"]),
        "baseline_fn": int(baseline["fn"]),
        "baseline_recall": baseline_recall,
        "baseline_fdr": baseline_fdr,
        "experiment_tp": int(experiment["tp"]),
        "experiment_fp": int(experiment["fp"]),
        "experiment_fn": int(experiment["fn"]),
        "experiment_recall": experiment_recall,
        "experiment_fdr": experiment_fdr,
        "recall_delta": experiment_recall - baseline_recall,
        "fdr_delta": experiment_fdr - baseline_fdr,
    }


def _benchmark_block(
    baseline_benchmark: Path | str | None,
    experiment_benchmark: Path | str | None,
) -> dict[str, float] | None:
    if baseline_benchmark is None or experiment_benchmark is None:
        return None
    baseline = _load_json(baseline_benchmark)
    experiment = _load_json(experiment_benchmark)
    baseline_median = float(baseline["median_s"])
    experiment_median = float(experiment["median_s"])
    baseline_p95 = float(baseline["p95_s"])
    experiment_p95 = float(experiment["p95_s"])
    return {
        "baseline_median_s": baseline_median,
        "experiment_median_s": experiment_median,
        "median_s_delta": experiment_median - baseline_median,
        "baseline_p95_s": baseline_p95,
        "experiment_p95_s": experiment_p95,
        "p95_s_delta": experiment_p95 - baseline_p95,
    }


def _write_markdown(
    path: Path,
    comparison: dict[str, Any],
    baseline_name: str,
    experiment_name: str,
) -> None:
    lines = [
        "# MKSNet-Lite Comparison",
        "",
        f"Baseline: `{baseline_name}`",
        f"Experiment: `{experiment_name}`",
        "",
        "## Overall",
        "",
        "| run | TP | FP | FN | Recall | FDR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    overall = comparison["overall"]
    lines.extend(
        [
            f"| {baseline_name} | {overall['baseline_tp']} | {overall['baseline_fp']} | {overall['baseline_fn']} | {overall['baseline_recall']:.4f} | {overall['baseline_fdr']:.4f} |",
            f"| {experiment_name} | {overall['experiment_tp']} | {overall['experiment_fp']} | {overall['experiment_fn']} | {overall['experiment_recall']:.4f} | {overall['experiment_fdr']:.4f} |",
            "",
            "## Coarse Groups",
            "",
            "| group | recall delta | FDR delta |",
            "| --- | ---: | ---: |",
        ]
    )
    for group, metrics in comparison["coarse"].items():
        lines.append(f"| {group} | {metrics['recall_delta']:.4f} | {metrics['fdr_delta']:.4f} |")
    if comparison.get("benchmark") is not None:
        benchmark = comparison["benchmark"]
        lines.extend(
            [
                "",
                "## Benchmark",
                "",
                f"Median delta: {benchmark['median_s_delta']:.4f}s",
                f"P95 delta: {benchmark['p95_s_delta']:.4f}s",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_experiments(
    baseline_report: Path | str,
    experiment_report: Path | str,
    output_dir: Path | str,
    baseline_name: str,
    experiment_name: str,
    baseline_benchmark: Path | str | None = None,
    experiment_benchmark: Path | str | None = None,
) -> dict[str, Any]:
    baseline = _load_json(baseline_report)
    experiment = _load_json(experiment_report)
    comparison: dict[str, Any] = {
        "baseline_name": baseline_name,
        "experiment_name": experiment_name,
        "overall": _metric_block(
            baseline["overall_class_agnostic"],
            experiment["overall_class_agnostic"],
        ),
        "coarse": {
            group: _metric_block(
                baseline["by_coarse_class"][group],
                experiment["by_coarse_class"][group],
            )
            for group in sorted(set(baseline["by_coarse_class"]) | set(experiment["by_coarse_class"]))
        },
        "fine_watchlist": {
            class_id: _metric_block(
                baseline["by_fine_class"][class_id],
                experiment["by_fine_class"][class_id],
            )
            for class_id in ("0", "1", "24")
            if class_id in baseline["by_fine_class"] and class_id in experiment["by_fine_class"]
        },
        "benchmark": _benchmark_block(baseline_benchmark, experiment_benchmark),
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(target / "comparison.md", comparison, baseline_name, experiment_name)
    return comparison
```

- [ ] **Step 4: Add comparison CLI command**

In `src/xh_detect/cli.py`, add this import:

```python
from xh_detect.compare import compare_experiments
```

Add this command before `serve(...)`:

```python
@app.command("compare-experiments")
def compare_experiments_command(
    baseline_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    experiment_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("outputs/xh25/mksnet-lite"),
    baseline_name: Annotated[str, typer.Option()] = "xh25-yolo26s-e80",
    experiment_name: Annotated[str, typer.Option()] = "xh25-mksnet-lite",
    baseline_benchmark: Annotated[Path | None, typer.Option()] = None,
    experiment_benchmark: Annotated[Path | None, typer.Option()] = None,
) -> None:
    comparison = compare_experiments(
        baseline_report=baseline_report,
        experiment_report=experiment_report,
        output_dir=output_dir,
        baseline_name=baseline_name,
        experiment_name=experiment_name,
        baseline_benchmark=baseline_benchmark,
        experiment_benchmark=experiment_benchmark,
    )
    typer.echo(json.dumps(comparison["overall"], ensure_ascii=False, allow_nan=False))
```

- [ ] **Step 5: Run comparison tests and CLI smoke**

Run:

```powershell
python -m pytest tests/test_compare.py tests/test_cli.py -q
python -m xh_detect.cli compare-experiments --help
```

Expected: tests PASS; help command exits 0 and includes `compare-experiments` options.

- [ ] **Step 6: Commit comparison work**

Run:

```powershell
git add src/xh_detect/compare.py src/xh_detect/cli.py tests/test_compare.py tests/test_cli.py
git commit -m "feat: compare mksnet lite experiment"
```

## Task 5: README Workflow And Verification Commands

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with MKSNet-Lite workflow**

Add this section after the official XH25 workflow section:

````markdown
## MKSNet-Lite 实验

`xh25-mksnet-lite` 是一个 MKSNet-inspired 中等改动实验。它保留现有 YOLO26s/HBB、
滑窗推理和比赛评估流程，只在 YOLO neck 中加入轻量多核空间/通道注意力模块。

训练前先准备官方数据：

```bash
.venv/bin/xh-detect prepare-xh25 \
  --source-root data \
  --output-root datasets/xh25 \
  --val-ratio 0.15 \
  --seed 42
```

训练 MKSNet-Lite：

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model configs/models/xh25-mksnet-lite.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --batch 8 \
  --workers 4 \
  --name xh25-mksnet-lite \
  --device 0
```

导出验证集预测：

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-mksnet-lite.yaml \
  --output-json outputs/xh25/mksnet-lite/val-predictions.json
```

评估与对比：

```bash
.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/baseline/val-predictions.json \
  --ground-truth-json datasets/xh25/annotations/val-coco.json \
  --output-path outputs/xh25/baseline/report.json \
  --taxonomy xh25

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/mksnet-lite/val-predictions.json \
  --ground-truth-json datasets/xh25/annotations/val-coco.json \
  --output-path outputs/xh25/mksnet-lite/report.json \
  --taxonomy xh25

.venv/bin/xh-detect compare-experiments \
  --baseline-report outputs/xh25/baseline/report.json \
  --experiment-report outputs/xh25/mksnet-lite/report.json \
  --output-dir outputs/xh25/mksnet-lite
```

保留 `comparison.json` 和 `comparison.md` 作为是否继续完整复刻 MKSNet 的依据。
````

- [ ] **Step 2: Run README-adjacent tests and full unit tests**

Run:

```powershell
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run lint/format checks if dev dependencies are installed**

Run:

```powershell
python -m ruff format --check .
python -m ruff check .
```

Expected: PASS. If `ruff` is not installed, run `.\\scripts\\uv.ps1 sync --extra dev` first, then rerun the commands through `.\\.venv\\Scripts\\python.exe`.

- [ ] **Step 4: Commit README update**

Run:

```powershell
git add README.md
git commit -m "docs: document mksnet lite workflow"
```

## Task 6: Local Smoke And GPU Handoff

**Files:**
- No source edits expected.

- [ ] **Step 1: Verify package environment**

Run:

```powershell
.\\scripts\\uv.ps1 sync --extra dev
.\\.venv\\Scripts\\xh-detect.exe env
```

Expected: `env` prints JSON with Python, PyTorch, Ultralytics, CUDA availability, and GPU name when a GPU is available.

- [ ] **Step 2: Prepare XH25 data**

Run:

```powershell
.\\.venv\\Scripts\\xh-detect.exe prepare-xh25 --source-root data --output-root datasets/xh25 --val-ratio 0.15 --seed 42
```

Expected: JSON includes `train_images` around 3807 and `val_images` around 674.

- [ ] **Step 3: Smoke train MKSNet-Lite for one epoch**

Run:

```powershell
.\\.venv\\Scripts\\xh-detect.exe train `
  --dataset-yaml datasets/xh25/dataset.yaml `
  --model configs/models/xh25-mksnet-lite.yaml `
  --pretrained yolo26s.pt `
  --epochs 1 `
  --image-size 640 `
  --batch 2 `
  --workers 0 `
  --name xh25-mksnet-lite-smoke `
  --device 0
```

Expected: command exits 0 and writes `runs/train/xh25-mksnet-lite-smoke/weights/best.pt`.

- [ ] **Step 4: Smoke infer one validation image**

Run:

```powershell
$first = Get-ChildItem datasets\\xh25\\images\\val -Filter *.jpg | Select-Object -First 1
.\\.venv\\Scripts\\xh-detect.exe infer `
  --image-path $first.FullName `
  --config-path configs/xh25-mksnet-lite.yaml `
  --output-dir outputs/xh25/mksnet-lite-smoke
```

Expected: command exits 0 and writes an annotated JPG plus JSON under `outputs/xh25/mksnet-lite-smoke`.

- [ ] **Step 5: Run final verification**

Run:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m ruff format --check .
.\\.venv\\Scripts\\python.exe -m ruff check .
git status --short --branch
```

Expected: tests and ruff pass. `git status` shows no tracked changes. Ignored `datasets/`, `runs/`, and `outputs/` may remain on disk.

## Self-Review

- Spec coverage: Tasks 1-3 implement the MKSNet-Lite module, Ultralytics registration, custom model YAML, and dedicated inference config. Task 4 implements the comparison report. Task 5 documents the workflow. Task 6 verifies local and GPU smoke execution.
- Red-flag scan: No unfinished markers or unspecified "write tests" steps remain.
- Type consistency: The custom block name is consistently `MKSNetLiteBlock`; the experiment name is consistently `xh25-mksnet-lite`; the optional training warm start is consistently `pretrained`.
