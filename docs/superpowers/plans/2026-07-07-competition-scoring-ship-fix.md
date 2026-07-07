# Competition Scoring Ship Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add official-scoring proxy reports, freeze the current optimized MKSNet-Lite inference config, build a deterministic ship-balanced training dataset variant, and run one 3090 experiment focused on improving ship ranking signals.

**Architecture:** Keep evaluator matching unchanged. Add focused modules for competition report generation and ship-balanced dataset materialization, expose them through thin Typer commands, and use ignored server artifacts for the long training run. The current thresholded MKSNet-Lite config remains the comparison baseline for the new ship-balanced run.

**Tech Stack:** Python 3.11/3.12, Typer, PyYAML, pytest, existing `xh_detect.evaluator`, `xh_detect.config`, `xh_detect.data.xh25`, `xh_detect.thresholds`, and Ultralytics training wrapper.

## Global Constraints

- Source scoring reference: `C:\Users\feng\project\fight\比赛评分方案-V1.5.pdf`.
- Overall detection Recall must be at least `0.85`.
- Overall FDR must be at most `0.20`.
- Inference time for one `10000x10000` image must be at most `20s` on one RTX3090 or equivalent accelerator.
- Ranking proxy signals are ship Recall, ship FDR, aircraft Recall, aircraft FDR, vehicle Recall, vehicle FDR, and overall timeliness.
- Do not change evaluator matching logic; it already uses IoU `0.50` for ship/aircraft and `0.35` for vehicle.
- Do not implement the full MKSNet paper in this iteration.
- Do not commit model weights, raw datasets, generated predictions, or large run artifacts.

---

## File Structure

- Create `src/xh_detect/competition.py`: load evaluation reports, compute hard-gate status, render competition proxy JSON and Markdown, and write artifacts.
- Create `tests/test_competition.py`: unit tests for hard gates, timing gate, coarse ranking signals, report loading, and artifact rendering.
- Create `configs/xh25-mksnet-lite-thresholded.yaml`: current MKSNet-Lite config using optimized validation thresholds.
- Modify `tests/test_config.py`: assert the thresholded config loads and contains the optimized class thresholds.
- Create `src/xh_detect/data/ship_balance.py`: build `datasets/xh25-ship-balanced` from an existing prepared XH25 dataset with deterministic QHS/MS train duplication.
- Create `tests/test_ship_balance.py`: unit tests for duplication policy, validation preservation, deterministic manifests, reports, and safety validation.
- Modify `src/xh_detect/cli.py`: add `competition-report` and `build-ship-balanced-xh25` commands.
- Modify `tests/test_cli.py`: cover both new commands.
- Modify `README.md`: document the scoring-first workflow, thresholded config, ship-balanced dataset command, training command, and report commands.
- Server-only ignored artifacts: `datasets/xh25-ship-balanced/`, `outputs/xh25/mksnet-lite-thresholded/`, `outputs/xh25/mksnet-lite-ship-balanced/`.

## Task 1: Competition Proxy Report

**Files:**
- Create: `src/xh_detect/competition.py`
- Create: `tests/test_competition.py`

**Interfaces:**
- Consumes: `xh_detect.evaluator.EvaluationReport`, `xh_detect.evaluator.Metrics`, `xh_detect.evaluator.report_to_dict`.
- Produces:
  - `RECALL_GATE: float = 0.85`
  - `FDR_GATE: float = 0.20`
  - `LATENCY_GATE_SECONDS: float = 20.0`
  - `load_evaluation_report(path: Path | str) -> EvaluationReport`
  - `build_competition_proxy(report: EvaluationReport, *, experiment_name: str, latency_seconds: float | None = None) -> dict[str, object]`
  - `render_competition_proxy_markdown(proxy: Mapping[str, object]) -> str`
  - `write_competition_proxy_artifacts(report: EvaluationReport, *, output_dir: Path, experiment_name: str, latency_seconds: float | None = None) -> dict[str, Path]`

- [ ] **Step 1: Write failing competition report tests**

Create `tests/test_competition.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from xh_detect.competition import (
    build_competition_proxy,
    load_evaluation_report,
    render_competition_proxy_markdown,
    write_competition_proxy_artifacts,
)
from xh_detect.evaluator import EvaluationReport, Metrics, report_to_dict


def _report(
    overall: Metrics = Metrics(90, 5, 10),
    ship: Metrics = Metrics(20, 4, 5),
    aircraft: Metrics = Metrics(60, 1, 3),
    vehicle: Metrics = Metrics(10, 0, 2),
) -> EvaluationReport:
    return EvaluationReport(
        overall_class_agnostic=overall,
        by_coarse_class={
            "ship": ship,
            "aircraft": aircraft,
            "vehicle": vehicle,
        },
        by_fine_class={0: ship, 4: aircraft, 24: vehicle},
        by_image={},
    )


def test_build_competition_proxy_marks_pass_candidate_without_timing() -> None:
    proxy = build_competition_proxy(_report(), experiment_name="unit")

    assert proxy["experiment_name"] == "unit"
    assert proxy["recommendation"] == "pass_candidate"
    assert proxy["hard_gates"] == {
        "overall_recall": {"value": 0.9, "threshold": 0.85, "passed": True},
        "overall_fdr": {"value": pytest.approx(5 / 95), "threshold": 0.2, "passed": True},
        "latency_seconds": {"value": None, "threshold": 20.0, "passed": None},
    }
    assert proxy["ranking_proxy"]["ship_recall"] == 0.8
    assert proxy["ranking_proxy"]["ship_fdr"] == pytest.approx(4 / 24)
    assert proxy["ranking_proxy"]["overall_timeliness_seconds"] is None


def test_build_competition_proxy_fails_accuracy_gate_before_timing_gate() -> None:
    proxy = build_competition_proxy(
        _report(overall=Metrics(80, 30, 30)),
        experiment_name="bad-accuracy",
        latency_seconds=25.0,
    )

    assert proxy["recommendation"] == "accuracy_gate_failed"
    assert proxy["hard_gates"]["overall_recall"]["passed"] is False
    assert proxy["hard_gates"]["overall_fdr"]["passed"] is False
    assert proxy["hard_gates"]["latency_seconds"]["passed"] is False


def test_build_competition_proxy_fails_timing_when_accuracy_passes() -> None:
    proxy = build_competition_proxy(_report(), experiment_name="slow", latency_seconds=20.1)

    assert proxy["recommendation"] == "timing_gate_failed"
    assert proxy["hard_gates"]["latency_seconds"] == {
        "value": 20.1,
        "threshold": 20.0,
        "passed": False,
    }


def test_build_competition_proxy_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="latency_seconds"):
        build_competition_proxy(_report(), experiment_name="bad", latency_seconds=-0.1)


def test_load_evaluation_report_round_trips_report_to_dict(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report_to_dict(_report())), encoding="utf-8")

    loaded = load_evaluation_report(path)

    assert loaded.overall_class_agnostic == Metrics(90, 5, 10)
    assert loaded.by_coarse_class["ship"] == Metrics(20, 4, 5)
    assert loaded.by_fine_class[24] == Metrics(10, 0, 2)


def test_write_competition_proxy_artifacts_writes_json_and_markdown(tmp_path: Path) -> None:
    artifacts = write_competition_proxy_artifacts(
        _report(),
        output_dir=tmp_path,
        experiment_name="unit",
        latency_seconds=12.5,
    )

    assert set(artifacts) == {"json", "markdown"}
    payload = json.loads((tmp_path / "competition-proxy.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "competition-proxy.md").read_text(encoding="utf-8")
    assert payload["ranking_proxy"]["overall_timeliness_seconds"] == 12.5
    assert "| Overall Recall |" in markdown
    assert "pass_candidate" in markdown


def test_render_competition_proxy_markdown_includes_all_ranking_signals() -> None:
    proxy = build_competition_proxy(_report(), experiment_name="unit", latency_seconds=10.0)

    markdown = render_competition_proxy_markdown(proxy)

    for label in (
        "Ship Recall",
        "Ship FDR",
        "Aircraft Recall",
        "Aircraft FDR",
        "Vehicle Recall",
        "Vehicle FDR",
        "Overall Timeliness",
    ):
        assert label in markdown
```

- [ ] **Step 2: Run the new tests and confirm red**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_competition.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'xh_detect.competition'`.

- [ ] **Step 3: Implement `src/xh_detect/competition.py`**

Create `src/xh_detect/competition.py` with these functions and constants:

```python
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from xh_detect.evaluator import EvaluationReport, Metrics, report_to_dict

RECALL_GATE = 0.85
FDR_GATE = 0.20
LATENCY_GATE_SECONDS = 20.0
COARSE_GROUPS = ("ship", "aircraft", "vehicle")


def _metric_from_mapping(payload: Mapping[str, object], label: str) -> Metrics:
    values: dict[str, int] = {}
    for key in ("tp", "fp", "fn"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} metric {key!r} must be a non-negative integer")
        values[key] = value
    return Metrics(values["tp"], values["fp"], values["fn"])


def _mapping_section(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    section = payload.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"evaluation report missing mapping section {key!r}")
    return section


def load_evaluation_report(path: Path | str) -> EvaluationReport:
    report_path = Path(path)
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation report JSON root must be an object")

    overall = _metric_from_mapping(
        _mapping_section(raw, "overall_class_agnostic"),
        "overall_class_agnostic",
    )
    coarse_raw = _mapping_section(raw, "by_coarse_class")
    fine_raw = _mapping_section(raw, "by_fine_class")
    image_raw = _mapping_section(raw, "by_image")
    by_coarse = {
        str(name): _metric_from_mapping(_mapping_section(coarse_raw, str(name)), f"coarse {name}")
        for name in sorted(coarse_raw)
    }
    by_fine = {
        int(class_id): _metric_from_mapping(
            _mapping_section(fine_raw, str(class_id)),
            f"fine {class_id}",
        )
        for class_id in sorted(fine_raw, key=lambda item: int(item))
    }
    by_image = {
        str(image_id): _metric_from_mapping(
            _mapping_section(image_raw, str(image_id)),
            f"image {image_id}",
        )
        for image_id in sorted(image_raw)
    }
    return EvaluationReport(
        overall_class_agnostic=overall,
        by_coarse_class=by_coarse,
        by_fine_class=by_fine,
        by_image=by_image,
    )


def _metric_payload(metrics: Metrics) -> dict[str, float | int]:
    return {
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "recall": metrics.recall,
        "fdr": metrics.fdr,
    }


def _gate(value: float | None, threshold: float, passed: bool | None) -> dict[str, float | bool | None]:
    return {"value": value, "threshold": threshold, "passed": passed}


def _validate_latency(latency_seconds: float | None) -> float | None:
    if latency_seconds is None:
        return None
    if (
        isinstance(latency_seconds, bool)
        or not isinstance(latency_seconds, int | float)
        or not math.isfinite(float(latency_seconds))
        or latency_seconds < 0.0
    ):
        raise ValueError("latency_seconds must be a non-negative finite number")
    return float(latency_seconds)


def build_competition_proxy(
    report: EvaluationReport,
    *,
    experiment_name: str,
    latency_seconds: float | None = None,
) -> dict[str, object]:
    if not experiment_name.strip():
        raise ValueError("experiment_name must be non-empty")
    latency = _validate_latency(latency_seconds)
    overall = report.overall_class_agnostic
    missing_groups = [group for group in COARSE_GROUPS if group not in report.by_coarse_class]
    if missing_groups:
        raise ValueError("missing coarse groups: " + ", ".join(missing_groups))

    recall_passed = overall.recall >= RECALL_GATE
    fdr_passed = overall.fdr <= FDR_GATE
    timing_passed = None if latency is None else latency <= LATENCY_GATE_SECONDS
    if not recall_passed or not fdr_passed:
        recommendation = "accuracy_gate_failed"
    elif timing_passed is False:
        recommendation = "timing_gate_failed"
    else:
        recommendation = "pass_candidate"

    ranking_proxy = {
        "ship_recall": report.by_coarse_class["ship"].recall,
        "ship_fdr": report.by_coarse_class["ship"].fdr,
        "aircraft_recall": report.by_coarse_class["aircraft"].recall,
        "aircraft_fdr": report.by_coarse_class["aircraft"].fdr,
        "vehicle_recall": report.by_coarse_class["vehicle"].recall,
        "vehicle_fdr": report.by_coarse_class["vehicle"].fdr,
        "overall_timeliness_seconds": latency,
    }
    return {
        "experiment_name": experiment_name,
        "recommendation": recommendation,
        "hard_gates": {
            "overall_recall": _gate(overall.recall, RECALL_GATE, recall_passed),
            "overall_fdr": _gate(overall.fdr, FDR_GATE, fdr_passed),
            "latency_seconds": _gate(latency, LATENCY_GATE_SECONDS, timing_passed),
        },
        "overall": _metric_payload(overall),
        "coarse": {
            group: _metric_payload(report.by_coarse_class[group])
            for group in COARSE_GROUPS
        },
        "ranking_proxy": ranking_proxy,
    }
```

Append Markdown rendering and artifact writing:

```python
def _fmt(value: object) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_competition_proxy_markdown(proxy: Mapping[str, object]) -> str:
    hard_gates = proxy["hard_gates"]
    ranking = proxy["ranking_proxy"]
    if not isinstance(hard_gates, Mapping) or not isinstance(ranking, Mapping):
        raise ValueError("competition proxy is malformed")

    gate_rows = (
        ("Overall Recall", hard_gates["overall_recall"]),
        ("Overall FDR", hard_gates["overall_fdr"]),
        ("Latency Seconds", hard_gates["latency_seconds"]),
    )
    ranking_rows = (
        ("Ship Recall", ranking["ship_recall"]),
        ("Ship FDR", ranking["ship_fdr"]),
        ("Aircraft Recall", ranking["aircraft_recall"]),
        ("Aircraft FDR", ranking["aircraft_fdr"]),
        ("Vehicle Recall", ranking["vehicle_recall"]),
        ("Vehicle FDR", ranking["vehicle_fdr"]),
        ("Overall Timeliness", ranking["overall_timeliness_seconds"]),
    )
    lines = [
        f"# {proxy['experiment_name']} Competition Proxy",
        "",
        f"- Recommendation: `{proxy['recommendation']}`",
        "",
        "## Hard Gates",
        "",
        "| Gate | Value | Threshold | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for label, gate in gate_rows:
        if not isinstance(gate, Mapping):
            raise ValueError("competition proxy gate is malformed")
        lines.append(
            f"| {label} | {_fmt(gate['value'])} | {_fmt(gate['threshold'])} | {gate['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Ranking Proxy Signals",
            "",
            "| Signal | Value |",
            "| --- | ---: |",
        ]
    )
    for label, value in ranking_rows:
        lines.append(f"| {label} | {_fmt(value)} |")
    lines.append("")
    return "\n".join(lines)


def write_competition_proxy_artifacts(
    report: EvaluationReport,
    *,
    output_dir: Path,
    experiment_name: str,
    latency_seconds: float | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy = build_competition_proxy(
        report,
        experiment_name=experiment_name,
        latency_seconds=latency_seconds,
    )
    json_path = output_dir / "competition-proxy.json"
    markdown_path = output_dir / "competition-proxy.md"
    json_path.write_text(
        json.dumps(proxy, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_competition_proxy_markdown(proxy), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
```

- [ ] **Step 4: Run Task 1 tests and confirm green**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_competition.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/xh_detect/competition.py tests/test_competition.py
git commit -m "feat: add competition proxy reports"
```

## Task 2: Thresholded MKSNet-Lite Config

**Files:**
- Create: `configs/xh25-mksnet-lite-thresholded.yaml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `PipelineConfig.from_yaml`.
- Produces: a committed config for the current best MKSNet-Lite thresholded candidate.

- [ ] **Step 1: Add failing config test**

Append this test to `tests/test_config.py`:

```python
def test_xh25_mksnet_lite_thresholded_yaml_uses_optimized_thresholds() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "xh25-mksnet-lite-thresholded.yaml"
    )

    config = PipelineConfig.from_yaml(config_path)

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-mksnet-lite/weights/best.pt"
    assert config.class_thresholds[2] == 0.40
    assert config.class_thresholds[4] == 0.55
    assert config.class_thresholds[5] == 0.50
    for class_id in set(range(25)) - {2, 4, 5}:
        assert config.class_thresholds[class_id] == 0.30
```

- [ ] **Step 2: Run config test and confirm red**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_config.py::test_xh25_mksnet_lite_thresholded_yaml_uses_optimized_thresholds -q
```

Expected: FAIL because `configs/xh25-mksnet-lite-thresholded.yaml` does not exist.

- [ ] **Step 3: Add thresholded config**

Create `configs/xh25-mksnet-lite-thresholded.yaml`:

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
  0: 0.30
  1: 0.30
  2: 0.40
  3: 0.30
  4: 0.55
  5: 0.50
  6: 0.30
  7: 0.30
  8: 0.30
  9: 0.30
  10: 0.30
  11: 0.30
  12: 0.30
  13: 0.30
  14: 0.30
  15: 0.30
  16: 0.30
  17: 0.30
  18: 0.30
  19: 0.30
  20: 0.30
  21: 0.30
  22: 0.30
  23: 0.30
  24: 0.30
```

- [ ] **Step 4: Run config tests and confirm green**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add configs/xh25-mksnet-lite-thresholded.yaml tests/test_config.py
git commit -m "config: add thresholded mksnet lite candidate"
```

## Task 3: Ship-Balanced Dataset Builder

**Files:**
- Create: `src/xh_detect/data/ship_balance.py`
- Create: `tests/test_ship_balance.py`

**Interfaces:**
- Consumes: prepared XH25 directory layout with `dataset.yaml`, `images/train`, `images/val`, `labels/train`, `labels/val`.
- Produces:
  - `ShipBalanceResult` dataclass with `output_root`, `original_train_images`, `balanced_train_images`, `duplicated_train_images`, `original_train_targets`, `balanced_train_targets`, and `duplicated_by_class`.
  - `build_ship_balanced_dataset(source_root: Path, output_root: Path, *, qhs_factor: int = 2, ms_factor: int = 2) -> ShipBalanceResult`.

- [ ] **Step 1: Write failing ship-balance tests**

Create `tests/test_ship_balance.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xh_detect.data.ship_balance import build_ship_balanced_dataset


def _write_sample(root: Path, split: str, stem: str, label: str) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{stem}.jpg").write_bytes(b"fake-jpeg")
    (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")


def _write_dataset(root: Path) -> None:
    names = {index: f"class-{index}" for index in range(25)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_sample(root, "train", "aircraft", "4 0.5 0.5 0.2 0.2\n")
    _write_sample(root, "train", "qhs", "2 0.5 0.5 0.2 0.2\n")
    _write_sample(root, "train", "ms", "3 0.5 0.5 0.2 0.2\n")
    _write_sample(
        root,
        "train",
        "qhs_ms",
        "2 0.4 0.4 0.2 0.2\n3 0.6 0.6 0.2 0.2\n",
    )
    _write_sample(root, "val", "val_ship", "3 0.5 0.5 0.2 0.2\n")


def test_build_ship_balanced_dataset_duplicates_qhs_and_ms_train_images(tmp_path: Path) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)

    result = build_ship_balanced_dataset(source, output)

    train_images = sorted(path.name for path in (output / "images" / "train").glob("*.jpg"))
    train_labels = sorted(path.name for path in (output / "labels" / "train").glob("*.txt"))
    assert train_images == [
        "aircraft.jpg",
        "ms.jpg",
        "ms__shipbal01.jpg",
        "qhs.jpg",
        "qhs__shipbal01.jpg",
        "qhs_ms.jpg",
        "qhs_ms__shipbal01.jpg",
    ]
    assert [Path(name).stem for name in train_labels] == [Path(name).stem for name in train_images]
    assert result.original_train_images == 4
    assert result.balanced_train_images == 7
    assert result.duplicated_train_images == 3
    assert result.duplicated_by_class == {2: 2, 3: 2}


def test_build_ship_balanced_dataset_keeps_validation_once_and_writes_reports(tmp_path: Path) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)

    build_ship_balanced_dataset(source, output)

    assert sorted(path.name for path in (output / "images" / "val").glob("*.jpg")) == [
        "val_ship.jpg"
    ]
    assert sorted(path.name for path in (output / "labels" / "val").glob("*.txt")) == [
        "val_ship.txt"
    ]
    dataset_yaml = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml["path"] == str(output.resolve())
    assert dataset_yaml["train"] == "images/train"
    assert dataset_yaml["val"] == "images/val"
    report = json.loads((output / "reports" / "ship-balance.json").read_text(encoding="utf-8"))
    assert report["policy"] == {"qhs_factor": 2, "ms_factor": 2}
    assert report["original_train_images"] == 4
    assert report["balanced_train_images"] == 7
    markdown = (output / "reports" / "ship-balance.md").read_text(encoding="utf-8")
    assert "| Original Train Images | 4 |" in markdown


def test_build_ship_balanced_dataset_rejects_overlapping_output(tmp_path: Path) -> None:
    source = tmp_path / "xh25"
    _write_dataset(source)

    with pytest.raises(ValueError, match="overlap"):
        build_ship_balanced_dataset(source, source / "nested")


def test_build_ship_balanced_dataset_rejects_existing_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)
    output.mkdir()
    (output / "existing.txt").write_text("busy", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        build_ship_balanced_dataset(source, output)


@pytest.mark.parametrize("kwargs", [{"qhs_factor": 0}, {"ms_factor": -1}, {"qhs_factor": True}])
def test_build_ship_balanced_dataset_rejects_bad_factors(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    _write_dataset(source)

    with pytest.raises(ValueError, match="factor"):
        build_ship_balanced_dataset(source, output, **kwargs)
```

- [ ] **Step 2: Run ship-balance tests and confirm red**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_ship_balance.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'xh_detect.data.ship_balance'`.

- [ ] **Step 3: Implement ship-balanced dataset builder**

Create `src/xh_detect/data/ship_balance.py` with:

```python
from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

QHS_CLASS_ID = 2
MS_CLASS_ID = 3


@dataclass(frozen=True)
class ShipBalanceResult:
    output_root: Path
    original_train_images: int
    balanced_train_images: int
    duplicated_train_images: int
    original_train_targets: dict[int, int]
    balanced_train_targets: dict[int, int]
    duplicated_by_class: dict[int, int]


def _positive_factor(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} factor must be a positive integer")
    return value


def _class_ids(label_path: Path) -> tuple[int, ...]:
    class_ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path} contains an invalid YOLO label line")
        class_ids.append(int(fields[0]))
    return tuple(class_ids)


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
```

Then add the validation, materialization, and report functions:

```python
def _validate_source_root(source_root: Path) -> dict[object, object]:
    dataset_yaml = source_root / "dataset.yaml"
    required = [
        dataset_yaml,
        source_root / "images" / "train",
        source_root / "images" / "val",
        source_root / "labels" / "train",
        source_root / "labels" / "val",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("source dataset is incomplete: " + ", ".join(missing))
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("names"), dict):
        raise ValueError("source dataset.yaml must contain names mapping")
    return payload


def _validate_output_root(source_root: Path, output_root: Path) -> None:
    resolved_source = source_root.resolve()
    resolved_output = output_root.resolve()
    if (
        resolved_source == resolved_output
        or resolved_source.is_relative_to(resolved_output)
        or resolved_output.is_relative_to(resolved_source)
    ):
        raise ValueError(
            f"source_root and output_root overlap: {resolved_source} / {resolved_output}"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output_root already exists and is not empty: {output_root}")


def _target_counts(label_paths: list[Path]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for path in label_paths:
        counts.update(_class_ids(path))
    return {class_id: counts.get(class_id, 0) for class_id in range(25)}


def _frequency(class_ids: tuple[int, ...], *, qhs_factor: int, ms_factor: int) -> int:
    frequency = 1
    if QHS_CLASS_ID in class_ids:
        frequency = max(frequency, qhs_factor)
    if MS_CLASS_ID in class_ids:
        frequency = max(frequency, ms_factor)
    return frequency


def _materialize_split_once(source_root: Path, output_root: Path, split: str) -> list[str]:
    stems: list[str] = []
    for image_path in sorted((source_root / "images" / split).glob("*.jpg")):
        label_path = source_root / "labels" / split / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"missing label for image {image_path.name}")
        _copy_or_link(image_path, output_root / "images" / split / image_path.name)
        _copy_or_link(label_path, output_root / "labels" / split / label_path.name)
        stems.append(image_path.stem)
    return stems


def _write_manifest(output_root: Path, split: str, stems: list[str]) -> None:
    manifest = output_root / "manifests" / f"{split}.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(f"images/{split}/{stem}.jpg\n" for stem in sorted(stems)),
        encoding="utf-8",
    )
```

Finish with:

```python
def _write_reports(
    result: ShipBalanceResult,
    *,
    output_root: Path,
    qhs_factor: int,
    ms_factor: int,
) -> None:
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_root": str(output_root.resolve()),
        "policy": {"qhs_factor": qhs_factor, "ms_factor": ms_factor},
        "original_train_images": result.original_train_images,
        "balanced_train_images": result.balanced_train_images,
        "duplicated_train_images": result.duplicated_train_images,
        "original_train_targets": result.original_train_targets,
        "balanced_train_targets": result.balanced_train_targets,
        "duplicated_by_class": result.duplicated_by_class,
    }
    (reports_dir / "ship-balance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown = "\n".join(
        [
            "# Ship-Balanced Dataset",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Original Train Images | {result.original_train_images} |",
            f"| Balanced Train Images | {result.balanced_train_images} |",
            f"| Duplicated Train Images | {result.duplicated_train_images} |",
            "",
        ]
    )
    (reports_dir / "ship-balance.md").write_text(markdown, encoding="utf-8")


def build_ship_balanced_dataset(
    source_root: Path,
    output_root: Path,
    *,
    qhs_factor: int = 2,
    ms_factor: int = 2,
) -> ShipBalanceResult:
    qhs_factor = _positive_factor(qhs_factor, "qhs")
    ms_factor = _positive_factor(ms_factor, "ms")
    source_root = Path(source_root)
    output_root = Path(output_root)
    dataset_payload = _validate_source_root(source_root)
    _validate_output_root(source_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_stems: list[str] = []
    balanced_label_paths: list[Path] = []
    original_label_paths = sorted((source_root / "labels" / "train").glob("*.txt"))
    duplicated_by_class: Counter[int] = Counter()

    for label_path in original_label_paths:
        image_path = source_root / "images" / "train" / f"{label_path.stem}.jpg"
        if not image_path.is_file():
            raise ValueError(f"missing image for label {label_path.name}")
        class_ids = _class_ids(label_path)
        frequency = _frequency(class_ids, qhs_factor=qhs_factor, ms_factor=ms_factor)
        for copy_index in range(frequency):
            suffix = "" if copy_index == 0 else f"__shipbal{copy_index:02d}"
            stem = f"{label_path.stem}{suffix}"
            if copy_index > 0:
                duplicated_by_class.update(set(class_ids) & {QHS_CLASS_ID, MS_CLASS_ID})
            _copy_or_link(image_path, output_root / "images" / "train" / f"{stem}.jpg")
            _copy_or_link(label_path, output_root / "labels" / "train" / f"{stem}.txt")
            train_stems.append(stem)
            balanced_label_paths.append(output_root / "labels" / "train" / f"{stem}.txt")

    val_stems = _materialize_split_once(source_root, output_root, "val")
    _write_manifest(output_root, "train", train_stems)
    _write_manifest(output_root, "val", val_stems)
    dataset_yaml = dict(dataset_payload)
    dataset_yaml["path"] = str(output_root.resolve())
    dataset_yaml["train"] = "images/train"
    dataset_yaml["val"] = "images/val"
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    result = ShipBalanceResult(
        output_root=output_root,
        original_train_images=len(original_label_paths),
        balanced_train_images=len(train_stems),
        duplicated_train_images=len(train_stems) - len(original_label_paths),
        original_train_targets=_target_counts(original_label_paths),
        balanced_train_targets=_target_counts(balanced_label_paths),
        duplicated_by_class=dict(sorted(duplicated_by_class.items())),
    )
    _write_reports(result, output_root=output_root, qhs_factor=qhs_factor, ms_factor=ms_factor)
    return result
```

- [ ] **Step 4: Run ship-balance tests and confirm green**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_ship_balance.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src/xh_detect/data/ship_balance.py tests/test_ship_balance.py
git commit -m "feat: build ship-balanced xh25 dataset"
```

## Task 4: CLI Commands

**Files:**
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes:
  - `build_ship_balanced_dataset(source_root, output_root, qhs_factor=2, ms_factor=2)`.
  - `load_evaluation_report(path)`.
  - `write_competition_proxy_artifacts(report, output_dir=..., experiment_name=..., latency_seconds=...)`.
- Produces:
  - `xh-detect build-ship-balanced-xh25`
  - `xh-detect competition-report`

- [ ] **Step 1: Add failing CLI tests**

Append these tests to `tests/test_cli.py`:

```python
@patch("xh_detect.cli.build_ship_balanced_dataset")
def test_build_ship_balanced_xh25_command_forwards_options(
    build_ship_balanced_dataset: Mock,
    tmp_path: Path,
) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    source.mkdir()
    build_ship_balanced_dataset.return_value = SimpleNamespace(
        output_root=output,
        original_train_images=4,
        balanced_train_images=7,
        duplicated_train_images=3,
    )

    result = CliRunner().invoke(
        app,
        [
            "build-ship-balanced-xh25",
            "--source-root",
            str(source),
            "--output-root",
            str(output),
            "--qhs-factor",
            "3",
            "--ms-factor",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["output_root"] == str(output)
    assert payload["balanced_train_images"] == 7
    build_ship_balanced_dataset.assert_called_once_with(
        source,
        output,
        qhs_factor=3,
        ms_factor=2,
    )


@patch("xh_detect.cli.write_competition_proxy_artifacts")
@patch("xh_detect.cli.load_evaluation_report")
def test_competition_report_command_writes_artifacts(
    load_evaluation_report: Mock,
    write_competition_proxy_artifacts: Mock,
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    output = tmp_path / "competition"
    report.write_text("{}", encoding="utf-8")
    loaded = object()
    load_evaluation_report.return_value = loaded

    result = CliRunner().invoke(
        app,
        [
            "competition-report",
            "--report-json",
            str(report),
            "--output-dir",
            str(output),
            "--experiment-name",
            "unit",
            "--latency-seconds",
            "12.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(output / "competition-proxy.json")
    load_evaluation_report.assert_called_once_with(report)
    write_competition_proxy_artifacts.assert_called_once_with(
        loaded,
        output_dir=output,
        experiment_name="unit",
        latency_seconds=12.5,
    )
```

Add `SimpleNamespace` to the existing imports if it is not already imported:

```python
from types import SimpleNamespace
```

- [ ] **Step 2: Run CLI tests and confirm red**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_cli.py::test_build_ship_balanced_xh25_command_forwards_options tests\test_cli.py::test_competition_report_command_writes_artifacts -q
```

Expected: FAIL because the imported CLI helper names and commands do not exist.

- [ ] **Step 3: Add CLI imports**

In `src/xh_detect/cli.py`, add:

```python
from xh_detect.competition import (
    load_evaluation_report,
    write_competition_proxy_artifacts,
)
from xh_detect.data.ship_balance import build_ship_balanced_dataset
```

- [ ] **Step 4: Add CLI commands**

Add these commands after `prepare_xh25` and before `train`:

```python
@app.command("build-ship-balanced-xh25")
def build_ship_balanced_xh25_command(
    source_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = Path("datasets/xh25"),
    output_root: Annotated[Path, typer.Option()] = Path("datasets/xh25-ship-balanced"),
    qhs_factor: Annotated[int, typer.Option(min=1)] = 2,
    ms_factor: Annotated[int, typer.Option(min=1)] = 2,
) -> None:
    try:
        result = build_ship_balanced_dataset(
            source_root,
            output_root,
            qhs_factor=qhs_factor,
            ms_factor=ms_factor,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "original_train_images": result.original_train_images,
                "balanced_train_images": result.balanced_train_images,
                "duplicated_train_images": result.duplicated_train_images,
            },
            ensure_ascii=False,
        )
    )
```

Add this command after `evaluate`:

```python
@app.command("competition-report")
def competition_report_command(
    report_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    output_dir: Annotated[Path, typer.Option()] = Path("outputs/xh25/competition-proxy"),
    experiment_name: Annotated[str, typer.Option()] = "xh25-experiment",
    latency_seconds: Annotated[float | None, typer.Option(min=0.0)] = None,
) -> None:
    try:
        report = load_evaluation_report(report_json)
        write_competition_proxy_artifacts(
            report,
            output_dir=output_dir,
            experiment_name=experiment_name,
            latency_seconds=latency_seconds,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(output_dir / "competition-proxy.json"))
```

- [ ] **Step 5: Run CLI focused tests and confirm green**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_cli.py::test_build_ship_balanced_xh25_command_forwards_options tests\test_cli.py::test_competition_report_command_writes_artifacts -q
```

Expected: PASS.

- [ ] **Step 6: Run related tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_competition.py tests\test_ship_balance.py tests\test_config.py tests\test_cli.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

Run:

```powershell
git add src/xh_detect/cli.py tests/test_cli.py
git commit -m "feat: expose competition and ship balance cli"
```

## Task 5: Documentation And Local Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes all commands from Tasks 1-4.
- Produces documented scoring-first workflow commands.

- [ ] **Step 1: Update README scoring-first workflow**

Add this subsection after the existing threshold optimization section:

````markdown
### 比赛评分优先实验

评分方案 `比赛评分方案-V1.5.pdf` 的初赛硬门槛是整体 Recall `>=0.85`、整体 FDR
`<=0.20`、单幅 `10000x10000` 图像推理时间 `<=20s`。通过硬门槛后，专家评分还会
参考 ship、aircraft、vehicle 各自 Recall/FDR 和总时效性 7 个排序信号。

当前 MKSNet-Lite 的阈值优化版配置为：

```bash
configs/xh25-mksnet-lite-thresholded.yaml
```

生成比赛评分代理报告：

```bash
.venv/bin/xh-detect competition-report \
  --report-json outputs/xh25/mksnet-lite/threshold-optimized/report.json \
  --output-dir outputs/xh25/mksnet-lite/threshold-optimized \
  --experiment-name xh25-mksnet-lite-thresholded
```

构建 QHS/MS 轻度重采样训练集：

```bash
.venv/bin/xh-detect build-ship-balanced-xh25 \
  --source-root datasets/xh25 \
  --output-root datasets/xh25-ship-balanced \
  --qhs-factor 2 \
  --ms-factor 2
```

训练 ship-balanced MKSNet-Lite：

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25-ship-balanced/dataset.yaml \
  --model configs/models/xh25-yolo26s-mksnet-lite.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-mksnet-lite-ship-balanced \
  --no-resume
```
````

- [ ] **Step 2: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_competition.py tests\test_ship_balance.py tests\test_config.py tests\test_cli.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full local verification**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m ruff check .
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Commit Task 5**

Run:

```powershell
git add README.md
git commit -m "docs: document scoring-first ship experiment"
```

## Task 6: Server Run On RTX3090

**Files:**
- No source edits expected.
- Ignored server/local artifacts under `datasets/` and `outputs/`.

**Interfaces:**
- Consumes committed branch from Tasks 1-5.
- Produces server artifacts for team comparison.

- [ ] **Step 1: Sync latest branch to the server**

From local worktree:

```powershell
git bundle create C:\Users\feng\project\fight\mksnet-lite-threshold.bundle codex/mksnet-lite
```

Upload the bundle to `/root/mksnet-lite-threshold.bundle`, then run on the server:

```bash
cd /root/XH-202625-remote-sensing-detection
git fetch /root/mksnet-lite-threshold.bundle codex/mksnet-lite:refs/remotes/bundle/codex-mksnet-lite
git checkout codex/mksnet-lite
git merge --ff-only refs/remotes/bundle/codex-mksnet-lite
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
.venv/bin/python -m pip install -e '.[dev]'
```

Expected: server branch equals local HEAD and editable install exits 0.

- [ ] **Step 2: Build the ship-balanced dataset on the server**

Run:

```bash
cd /root/XH-202625-remote-sensing-detection
python - <<'PY'
from pathlib import Path
import shutil

root = Path.cwd().resolve()
target = (root / "datasets/xh25-ship-balanced").resolve()
allowed = (root / "datasets").resolve()
if target == allowed or not target.is_relative_to(allowed):
    raise SystemExit(f"refusing unsafe removal target: {target}")
if target.exists():
    shutil.rmtree(target)
PY
.venv/bin/xh-detect build-ship-balanced-xh25 \
  --source-root datasets/xh25 \
  --output-root datasets/xh25-ship-balanced \
  --qhs-factor 2 \
  --ms-factor 2
```

Expected: command prints JSON with `balanced_train_images` greater than `original_train_images`, and `datasets/xh25-ship-balanced/reports/ship-balance.md` exists.

- [ ] **Step 3: Generate current thresholded competition proxy**

Run:

```bash
cd /root/XH-202625-remote-sensing-detection
.venv/bin/xh-detect competition-report \
  --report-json outputs/xh25/mksnet-lite/threshold-optimized/report.json \
  --output-dir outputs/xh25/mksnet-lite/threshold-optimized \
  --experiment-name xh25-mksnet-lite-thresholded
```

Expected: writes `competition-proxy.json` and `competition-proxy.md` with recommendation `pass_candidate`.

- [ ] **Step 4: Start ship-balanced training**

Run:

```bash
cd /root/XH-202625-remote-sensing-detection
mkdir -p outputs/xh25/mksnet-lite-ship-balanced
nohup .venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25-ship-balanced/dataset.yaml \
  --model configs/models/xh25-yolo26s-mksnet-lite.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-mksnet-lite-ship-balanced \
  --no-resume \
  > outputs/xh25/mksnet-lite-ship-balanced/train.log 2>&1 &
echo $! > outputs/xh25/mksnet-lite-ship-balanced/train.pid
```

Expected: background PID is written. Monitor with:

```bash
tail -n 80 outputs/xh25/mksnet-lite-ship-balanced/train.log
nvidia-smi
```

- [ ] **Step 5: After training completes, create validation predictions**

If `runs/train/xh25-mksnet-lite-ship-balanced/weights/best.pt` exists, create a temporary inference config on the server:

```bash
cat > outputs/xh25/mksnet-lite-ship-balanced/infer-search.yaml <<'YAML'
task: detect
taxonomy: xh25
model_path: runs/train/xh25-mksnet-lite-ship-balanced/weights/best.pt
device: "0"
image_size: 1024
tile_size: 1024
overlap: 0.2
batch_size: 8
merge_iou: 0.3
edge_margin: 16
half: true
class_thresholds:
  0: 0.05
  1: 0.05
  2: 0.05
  3: 0.05
  4: 0.05
  5: 0.05
  6: 0.05
  7: 0.05
  8: 0.05
  9: 0.05
  10: 0.05
  11: 0.05
  12: 0.05
  13: 0.05
  14: 0.05
  15: 0.05
  16: 0.05
  17: 0.05
  18: 0.05
  19: 0.05
  20: 0.05
  21: 0.05
  22: 0.05
  23: 0.05
  24: 0.05
YAML
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path outputs/xh25/mksnet-lite-ship-balanced/infer-search.yaml \
  --output-json outputs/xh25/mksnet-lite-ship-balanced/val-predictions.json
```

Expected: validation predictions JSON exists.

- [ ] **Step 6: Evaluate and optimize thresholds**

Run:

```bash
cd /root/XH-202625-remote-sensing-detection
.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/mksnet-lite-ship-balanced/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/mksnet-lite-ship-balanced/report.json
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/mksnet-lite-ship-balanced/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --baseline-report outputs/xh25/mksnet-lite/threshold-optimized/report.json \
  --taxonomy xh25 \
  --output-dir outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized
.venv/bin/xh-detect competition-report \
  --report-json outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized/report.json \
  --output-dir outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized \
  --experiment-name xh25-mksnet-lite-ship-balanced-thresholded
```

Expected: all three commands exit 0 and threshold-optimized artifacts exist.

- [ ] **Step 7: Inspect final metrics**

Run:

```bash
cd /root/XH-202625-remote-sensing-detection
.venv/bin/python - <<'PY'
import json
from pathlib import Path

current = json.loads(Path("outputs/xh25/mksnet-lite/threshold-optimized/competition-proxy.json").read_text())
new = json.loads(Path("outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized/competition-proxy.json").read_text())
summary = {
    "current": {
        "overall_recall": current["overall"]["recall"],
        "overall_fdr": current["overall"]["fdr"],
        "ship_recall": current["ranking_proxy"]["ship_recall"],
        "ship_fdr": current["ranking_proxy"]["ship_fdr"],
        "recommendation": current["recommendation"],
    },
    "ship_balanced": {
        "overall_recall": new["overall"]["recall"],
        "overall_fdr": new["overall"]["fdr"],
        "ship_recall": new["ranking_proxy"]["ship_recall"],
        "ship_fdr": new["ranking_proxy"]["ship_fdr"],
        "recommendation": new["recommendation"],
    },
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
```

Expected: output clearly shows whether ship-balanced improves ship Recall or FDR while preserving the overall gates.

- [ ] **Step 8: Download compact artifacts locally**

Copy these directories or files to `C:\Users\feng\project\fight\.worktrees\mksnet-lite\outputs\server\mksnet-lite-ship-balanced\`:

```text
datasets/xh25-ship-balanced/reports/ship-balance.json
datasets/xh25-ship-balanced/reports/ship-balance.md
outputs/xh25/mksnet-lite/threshold-optimized/competition-proxy.json
outputs/xh25/mksnet-lite/threshold-optimized/competition-proxy.md
outputs/xh25/mksnet-lite-ship-balanced/report.json
outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized/optimized-thresholds.yaml
outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized/report.json
outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized/search-summary.json
outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized/search-summary.md
outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized/competition-proxy.json
outputs/xh25/mksnet-lite-ship-balanced/threshold-optimized/competition-proxy.md
```

Expected: local ignored `outputs/server/mksnet-lite-ship-balanced/` contains compact artifacts for team discussion. Do not commit them unless the team explicitly asks to version results.

## Self-Review

- Spec coverage: Task 1 implements official hard-gate and seven-signal proxy reporting. Task 2 freezes the current optimized threshold config. Task 3 implements the QHS/MS ship-balanced dataset variant. Task 4 exposes the workflow through CLI. Task 5 documents and verifies locally. Task 6 runs the 3090 experiment and retrieves compact artifacts.
- Scope: The plan does not change evaluator matching, does not implement the full MKSNet paper, and does not commit weights or generated datasets.
- Type consistency: `build_competition_proxy`, `write_competition_proxy_artifacts`, `build_ship_balanced_dataset`, and CLI command names are consistent across tasks.
