# Shared Types and Validated Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the shared domain dataclasses and a validated pipeline configuration loader backed by the baseline YAML.

**Architecture:** Keep shared records in a small immutable types module, and keep config parsing/validation in a separate config module that only depends on those types and `yaml`. Drive the whole change from tests: start with baseline-loading and error-path tests, then add validation-focused tests, then implement the minimal parser and dataclasses.

**Tech Stack:** Python 3.11, `dataclasses`, `pathlib`, `typing`, `PyYAML`, `pytest`, `ruff`.

---

### Task 1: Baseline config loading and error handling

**Files:**
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from xh_detect.config import PipelineConfig


def test_baseline_yaml_loads_expected_values():
    config = PipelineConfig.from_yaml(Path("configs/baseline.yaml"))

    assert config.model_path == "yolo26s-obb.pt"
    assert config.tile_size == 1024
    assert config.thresholds == {0: 0.25, 1: 0.25, 2: 0.25}


def test_overlap_1_0_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        PipelineConfig(overlap=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'xh_detect.config'`

- [ ] **Step 3: Write minimal implementation**

No production code yet.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS after config module exists

- [ ] **Step 5: Commit**

```bash
git add tests/test_config.py
git commit -m "feat: add shared types and validated pipeline config"
```

### Task 2: Immutable shared types and config validation

**Files:**
- Create: `src/xh_detect/types.py`
- Create: `src/xh_detect/config.py`
- Modify: `tests/test_config.py`
- Create: `configs/baseline.yaml`

- [ ] **Step 1: Add focused tests for validation and YAML parsing**

```python
from pathlib import Path

import pytest

from xh_detect.config import PipelineConfig


def test_validation_rejects_non_positive_sizes():
    with pytest.raises(ValueError, match="tile_size"):
        PipelineConfig(tile_size=0)


def test_validation_rejects_merge_iou_out_of_range():
    with pytest.raises(ValueError, match="merge_iou"):
        PipelineConfig(merge_iou=1.1)


def test_validation_rejects_missing_class_thresholds_mapping():
    with pytest.raises(ValueError, match="class_thresholds"):
        PipelineConfig.from_yaml(Path("tests/fixtures/no_thresholds.yaml"))
```

- [ ] **Step 2: Run focused tests to verify red**

Run: `uv run pytest tests/test_config.py -v`
Expected: failures for missing validation and parsing behavior

- [ ] **Step 3: Implement minimal config loader and dataclasses**

Use frozen dataclasses for the shared records, and validate values in `PipelineConfig.__post_init__` plus `from_yaml`.

- [ ] **Step 4: Run targeted and full tests**

Run: `uv run pytest tests/test_config.py -v`
Run: `uv run pytest -q`

- [ ] **Step 5: Commit**

```bash
git add src/xh_detect/types.py src/xh_detect/config.py tests/test_config.py configs/baseline.yaml
git commit -m "feat: add shared types and validated pipeline config"
```

### Task 3: Quality checks and handoff

**Files:**
- Modify: only if formatting or linting requires it

- [ ] **Step 1: Run Ruff format and check**

Run: `uv run ruff format --check .`
Run: `uv run ruff check .`

- [ ] **Step 2: Self-review behavior against spec**

Confirm the baseline values, validation ranges, immutable records, and YAML error messages all match the requirements.

- [ ] **Step 3: Final commit only if needed**

If lint fixes were required, commit those as part of the same feature commit only when the repository is green.
