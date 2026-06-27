# Threshold Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible `xh-detect optimize-thresholds` workflow that selects per-class confidence thresholds from existing validation predictions and writes reports ready for comparison with the main-line result.

**Architecture:** Keep evaluator matching, inference, model code, and dataset conversion unchanged. Add a focused `xh_detect.thresholds` module for threshold validation, objective scoring, deterministic greedy search, and artifact writing, then expose it through a thin Typer command.

**Tech Stack:** Python 3.11, Typer, PyYAML, pytest, existing `xh_detect.evaluator`, `xh_detect.compare`, and `xh_detect.taxonomy` modules.

---

## File Structure

- Create `src/xh_detect/thresholds.py`: threshold grid parsing, threshold-map validation, objective scoring, per-class prediction filtering, deterministic greedy threshold optimization, and report artifact writing.
- Create `tests/test_thresholds.py`: unit tests for filtering, validation, objective ordering, optimizer behavior, recall floor behavior, and output artifacts.
- Modify `src/xh_detect/cli.py`: add `optimize-thresholds`, parse CLI options, load predictions and ground truth through existing evaluator helpers, and convert validation failures to Typer errors.
- Modify `tests/test_cli.py`: add CLI forwarding and invalid-grid tests for the new command.
- Modify `README.md`: document the post-training threshold optimization step and how to apply the generated `class_thresholds`.
- Produce server artifacts under `outputs/xh25/mksnet-lite/threshold-optimized/` after implementation, using the already completed 3090 run.

## Task 1: Objective Scoring And Threshold Filtering

**Files:**
- Create: `src/xh_detect/thresholds.py`
- Create: `tests/test_thresholds.py`

- [ ] **Step 1: Write failing tests for parsing, filtering, and objective ordering**

Create `tests/test_thresholds.py` with this initial content:

```python
from __future__ import annotations

import math

import pytest

from xh_detect.evaluator import EvaluationReport, Metrics
from xh_detect.taxonomy import get_taxonomy
from xh_detect.thresholds import (
    ObjectiveScore,
    filter_predictions_by_class_threshold,
    f1_score,
    is_better_objective,
    objective_from_report,
    parse_threshold_grid,
    validate_threshold_map,
)
from xh_detect.types import Detection

BOX = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def _report(tp: int, fp: int, fn: int) -> EvaluationReport:
    metrics = Metrics(tp, fp, fn)
    return EvaluationReport(
        overall_class_agnostic=metrics,
        by_coarse_class={"ship": metrics},
        by_fine_class={0: metrics},
        by_image={},
    )


def test_parse_threshold_grid_returns_sorted_unique_values() -> None:
    assert parse_threshold_grid("0.30, 0.10,0.30,0.20") == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("grid", ["", "0.2,bad", "-0.1,0.2", "0.2,1.1", "nan", "inf"])
def test_parse_threshold_grid_rejects_invalid_values(grid: str) -> None:
    with pytest.raises(ValueError):
        parse_threshold_grid(grid)


def test_validate_threshold_map_rejects_invalid_class_ids_and_values() -> None:
    taxonomy = get_taxonomy("legacy3")

    assert validate_threshold_map({0: 0.25, "1": 0.3}, taxonomy) == {0: 0.25, 1: 0.3}

    with pytest.raises(ValueError, match="class ID"):
        validate_threshold_map({9: 0.25}, taxonomy)
    with pytest.raises(TypeError, match="threshold"):
        validate_threshold_map({0: True}, taxonomy)
    with pytest.raises(ValueError, match="threshold"):
        validate_threshold_map({0: math.nan}, taxonomy)


def test_filter_predictions_uses_class_specific_thresholds_inclusively() -> None:
    predictions = [
        Detection("img", 0, 0.50, BOX),
        Detection("img", 0, 0.49, BOX),
        Detection("img", 1, 0.20, BOX),
        Detection("img", 1, 0.19, BOX),
    ]

    filtered = filter_predictions_by_class_threshold(
        predictions,
        {0: 0.50, 1: 0.20},
        taxonomy=get_taxonomy("legacy3"),
    )

    assert filtered == [predictions[0], predictions[2]]


def test_objective_from_report_computes_precision_recall_fdr_and_f1() -> None:
    objective = objective_from_report(_report(tp=8, fp=2, fn=2))

    assert objective == ObjectiveScore(
        f1=pytest.approx(0.8),
        precision=0.8,
        recall=0.8,
        fdr=0.2,
        tp=8,
        fp=2,
        fn=2,
    )
    assert f1_score(recall=0.0, fdr=1.0) == 0.0


def test_is_better_objective_prefers_f1_then_lower_fdr_then_higher_recall() -> None:
    incumbent = ObjectiveScore(f1=0.90, precision=0.90, recall=0.90, fdr=0.10, tp=9, fp=1, fn=1)

    assert is_better_objective(
        ObjectiveScore(f1=0.91, precision=0.91, recall=0.90, fdr=0.09, tp=9, fp=1, fn=1),
        incumbent,
    )
    assert is_better_objective(
        ObjectiveScore(f1=0.9001, precision=0.92, recall=0.88, fdr=0.08, tp=9, fp=1, fn=1),
        incumbent,
        tie_epsilon=0.0005,
    )
    assert is_better_objective(
        ObjectiveScore(f1=0.9001, precision=0.90, recall=0.91, fdr=0.1001, tp=9, fp=1, fn=1),
        incumbent,
        tie_epsilon=0.0005,
    )
    assert not is_better_objective(
        ObjectiveScore(f1=0.8990, precision=0.99, recall=0.99, fdr=0.01, tp=9, fp=1, fn=1),
        incumbent,
        tie_epsilon=0.0005,
    )
```

- [ ] **Step 2: Run the new tests and confirm red**

Run:

```powershell
python -m pytest tests/test_thresholds.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'xh_detect.thresholds'`.

- [ ] **Step 3: Implement threshold parsing, validation, filtering, and objectives**

Create `src/xh_detect/thresholds.py`:

```python
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import yaml

from xh_detect.compare import compare_experiments
from xh_detect.evaluator import EvaluationReport, evaluate, report_to_dict
from xh_detect.taxonomy import Taxonomy
from xh_detect.types import Detection, ObjectAnnotation

DEFAULT_THRESHOLD_GRID = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
)
DEFAULT_THRESHOLD_GRID_TEXT = ",".join(f"{value:.2f}" for value in DEFAULT_THRESHOLD_GRID)


@dataclass(frozen=True)
class ObjectiveScore:
    f1: float
    precision: float
    recall: float
    fdr: float
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    thresholds: dict[int, float]
    report: EvaluationReport
    objective: ObjectiveScore
    baseline_objective: ObjectiveScore | None
    global_threshold: float
    grid: Sequence[float]
    recall_floor: float | None
    candidates: Sequence[dict[str, object]]


def _threshold_value(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} threshold must be a finite real number")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{context} threshold must be in [0, 1]")
    return threshold


def parse_threshold_grid(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(part == "" for part in parts):
            raise ValueError("threshold grid must contain comma-separated numbers")
        raw_values: list[object] = []
        for part in parts:
            try:
                raw_values.append(float(part))
            except ValueError as exc:
                raise ValueError(f"invalid threshold value {part!r}") from exc
    else:
        raw_values = list(value)
        if not raw_values:
            raise ValueError("threshold grid must contain at least one threshold")

    thresholds = {_threshold_value(item, f"threshold grid item {index}") for index, item in enumerate(raw_values)}
    if not thresholds:
        raise ValueError("threshold grid must contain at least one threshold")
    return sorted(thresholds)


def _class_id(value: object, taxonomy: Taxonomy) -> int:
    if isinstance(value, bool):
        raise TypeError("class ID must be an integer")
    if isinstance(value, str):
        if not value.isdecimal():
            raise TypeError("class ID must be an integer")
        class_id = int(value)
    elif isinstance(value, Integral):
        class_id = int(value)
    else:
        raise TypeError("class ID must be an integer")
    if class_id not in taxonomy.valid_ids:
        valid = ", ".join(str(item) for item in sorted(taxonomy.valid_ids))
        raise ValueError(f"class ID must be one of {valid}")
    return class_id


def validate_threshold_map(
    thresholds: Mapping[object, object],
    taxonomy: Taxonomy,
) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for key, value in thresholds.items():
        normalized[_class_id(key, taxonomy)] = _threshold_value(value, f"class {key}")
    return normalized


def filter_predictions_by_class_threshold(
    predictions: Iterable[Detection],
    thresholds: Mapping[object, object],
    taxonomy: Taxonomy,
) -> list[Detection]:
    threshold_map = validate_threshold_map(thresholds, taxonomy)
    return [
        item
        for item in predictions
        if item.score >= threshold_map.get(item.class_id, 0.0)
    ]


def f1_score(*, recall: float, fdr: float) -> float:
    precision = 1.0 - fdr
    denominator = precision + recall
    return (2.0 * precision * recall / denominator) if denominator else 0.0


def objective_from_report(report: EvaluationReport) -> ObjectiveScore:
    metrics = report.overall_class_agnostic
    precision = 1.0 - metrics.fdr
    return ObjectiveScore(
        f1=f1_score(recall=metrics.recall, fdr=metrics.fdr),
        precision=precision,
        recall=metrics.recall,
        fdr=metrics.fdr,
        tp=metrics.tp,
        fp=metrics.fp,
        fn=metrics.fn,
    )


def is_better_objective(
    candidate: ObjectiveScore,
    incumbent: ObjectiveScore,
    *,
    tie_epsilon: float = 0.0005,
) -> bool:
    if candidate.f1 > incumbent.f1 + tie_epsilon:
        return True
    if incumbent.f1 > candidate.f1 + tie_epsilon:
        return False
    if candidate.fdr < incumbent.fdr - tie_epsilon:
        return True
    if incumbent.fdr < candidate.fdr - tie_epsilon:
        return False
    return candidate.recall > incumbent.recall + tie_epsilon
```

- [ ] **Step 4: Run Task 1 tests and confirm green**

Run:

```powershell
python -m pytest tests/test_thresholds.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/xh_detect/thresholds.py tests/test_thresholds.py
git commit -m "feat: add threshold objective utilities"
```

## Task 2: Deterministic Greedy Threshold Optimizer

**Files:**
- Modify: `src/xh_detect/thresholds.py`
- Modify: `tests/test_thresholds.py`

- [ ] **Step 1: Add failing optimizer tests**

Append these tests to `tests/test_thresholds.py`:

```python
from xh_detect.thresholds import optimize_thresholds
from xh_detect.types import ObjectAnnotation


def test_optimize_thresholds_selects_different_thresholds_per_class() -> None:
    taxonomy = get_taxonomy("legacy3")
    truth = [
        ObjectAnnotation("ship-ok", 0, BOX),
        ObjectAnnotation("aircraft-low", 1, BOX),
    ]
    predictions = [
        Detection("ship-ok", 0, 0.90, BOX),
        Detection("ship-fp", 0, 0.20, BOX),
        Detection("aircraft-low", 1, 0.20, BOX),
    ]

    result = optimize_thresholds(
        predictions,
        truth,
        taxonomy=taxonomy,
        thresholds=(0.20, 0.50),
        passes=2,
    )

    assert result.global_threshold == 0.20
    assert result.thresholds[0] == 0.50
    assert result.thresholds[1] == 0.20
    assert result.objective.f1 == pytest.approx(1.0)
    assert result.report.overall_class_agnostic == Metrics(tp=2, fp=0, fn=0)


def test_optimize_thresholds_recall_floor_rejects_high_f1_low_recall_candidate() -> None:
    taxonomy = get_taxonomy("legacy3")
    truth = [
        ObjectAnnotation("keep-high", 0, BOX),
        ObjectAnnotation("keep-low", 0, BOX),
    ]
    predictions = [
        Detection("keep-high", 0, 0.90, BOX),
        Detection("keep-low", 0, 0.20, BOX),
        Detection("fp-1", 0, 0.20, BOX),
        Detection("fp-2", 0, 0.20, BOX),
        Detection("fp-3", 0, 0.20, BOX),
        Detection("fp-4", 0, 0.20, BOX),
    ]
    baseline = ObjectiveScore(f1=0.50, precision=0.33, recall=1.0, fdr=0.67, tp=2, fp=4, fn=0)

    result = optimize_thresholds(
        predictions,
        truth,
        taxonomy=taxonomy,
        thresholds=(0.20, 0.50),
        baseline_objective=baseline,
        recall_floor_delta=0.0,
        passes=2,
    )

    assert result.recall_floor == 1.0
    assert result.thresholds[0] == 0.20
    assert result.report.overall_class_agnostic.recall == 1.0
```

- [ ] **Step 2: Run optimizer tests and confirm red**

Run:

```powershell
python -m pytest tests/test_thresholds.py::test_optimize_thresholds_selects_different_thresholds_per_class tests/test_thresholds.py::test_optimize_thresholds_recall_floor_rejects_high_f1_low_recall_candidate -q
```

Expected: FAIL with `ImportError` or `AttributeError` for `optimize_thresholds`.

- [ ] **Step 3: Add optimizer helper functions**

Append this code to `src/xh_detect/thresholds.py`:

```python
def _objective_dict(objective: ObjectiveScore) -> dict[str, float | int]:
    return {
        "f1": objective.f1,
        "precision": objective.precision,
        "recall": objective.recall,
        "fdr": objective.fdr,
        "tp": objective.tp,
        "fp": objective.fp,
        "fn": objective.fn,
    }


def _passes_recall_floor(objective: ObjectiveScore, recall_floor: float | None) -> bool:
    return recall_floor is None or objective.recall >= recall_floor


def _candidate_is_preferred(
    candidate: tuple[float, EvaluationReport, ObjectiveScore],
    incumbent: tuple[float, EvaluationReport, ObjectiveScore],
    *,
    tie_epsilon: float,
    reference_threshold: float,
) -> bool:
    candidate_threshold, _, candidate_objective = candidate
    incumbent_threshold, _, incumbent_objective = incumbent
    if is_better_objective(candidate_objective, incumbent_objective, tie_epsilon=tie_epsilon):
        return True
    if is_better_objective(incumbent_objective, candidate_objective, tie_epsilon=tie_epsilon):
        return False
    return abs(candidate_threshold - reference_threshold) < abs(incumbent_threshold - reference_threshold)


def _select_best_candidate(
    candidates: list[tuple[float, EvaluationReport, ObjectiveScore]],
    *,
    tie_epsilon: float,
    reference_threshold: float,
    recall_floor: float | None,
) -> tuple[float, EvaluationReport, ObjectiveScore]:
    eligible = [
        candidate
        for candidate in candidates
        if _passes_recall_floor(candidate[2], recall_floor)
    ]
    pool = eligible if eligible else candidates
    best = pool[0]
    for candidate in pool[1:]:
        if _candidate_is_preferred(
            candidate,
            best,
            tie_epsilon=tie_epsilon,
            reference_threshold=reference_threshold,
        ):
            best = candidate
    return best


def _evaluate_threshold_map(
    predictions: list[Detection],
    ground_truth: list[ObjectAnnotation],
    taxonomy: Taxonomy,
    threshold_map: Mapping[int, float],
) -> EvaluationReport:
    return evaluate(
        filter_predictions_by_class_threshold(predictions, threshold_map, taxonomy),
        ground_truth,
        taxonomy=taxonomy,
    )
```

- [ ] **Step 4: Add the optimizer implementation**

Append this function to `src/xh_detect/thresholds.py`:

```python
def optimize_thresholds(
    predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
    *,
    taxonomy: Taxonomy,
    thresholds: str | Sequence[float] = DEFAULT_THRESHOLD_GRID,
    baseline_objective: ObjectiveScore | None = None,
    recall_floor_delta: float = 0.003,
    tie_epsilon: float = 0.0005,
    passes: int = 2,
) -> ThresholdOptimizationResult:
    if isinstance(recall_floor_delta, bool) or recall_floor_delta < 0.0:
        raise ValueError("recall_floor_delta must be non-negative")
    if isinstance(tie_epsilon, bool) or tie_epsilon < 0.0:
        raise ValueError("tie_epsilon must be non-negative")
    if isinstance(passes, bool) or passes < 1:
        raise ValueError("passes must be a positive integer")

    prediction_items = list(predictions)
    truth_items = list(ground_truth)
    grid = parse_threshold_grid(thresholds)
    recall_floor = (
        None
        if baseline_objective is None
        else max(0.0, baseline_objective.recall - recall_floor_delta)
    )

    global_candidates: list[tuple[float, EvaluationReport, ObjectiveScore]] = []
    for threshold in grid:
        threshold_map = {class_id: threshold for class_id in sorted(taxonomy.valid_ids)}
        report = _evaluate_threshold_map(prediction_items, truth_items, taxonomy, threshold_map)
        global_candidates.append((threshold, report, objective_from_report(report)))

    global_threshold, best_report, best_objective = _select_best_candidate(
        global_candidates,
        tie_epsilon=tie_epsilon,
        reference_threshold=0.25,
        recall_floor=recall_floor,
    )
    threshold_map = {class_id: global_threshold for class_id in sorted(taxonomy.valid_ids)}
    records: list[dict[str, object]] = [
        {
            "stage": "global",
            "threshold": global_threshold,
            "objective": _objective_dict(best_objective),
        }
    ]

    for pass_index in range(1, passes + 1):
        for class_id in sorted(taxonomy.valid_ids):
            class_candidates: list[tuple[float, EvaluationReport, ObjectiveScore]] = []
            for threshold in grid:
                candidate_thresholds = dict(threshold_map)
                candidate_thresholds[class_id] = threshold
                report = _evaluate_threshold_map(
                    prediction_items,
                    truth_items,
                    taxonomy,
                    candidate_thresholds,
                )
                class_candidates.append((threshold, report, objective_from_report(report)))

            selected_threshold, selected_report, selected_objective = _select_best_candidate(
                class_candidates,
                tie_epsilon=tie_epsilon,
                reference_threshold=global_threshold,
                recall_floor=recall_floor,
            )
            accepted = selected_threshold != threshold_map[class_id] and is_better_objective(
                selected_objective,
                best_objective,
                tie_epsilon=tie_epsilon,
            )
            if accepted:
                threshold_map[class_id] = selected_threshold
                best_report = selected_report
                best_objective = selected_objective
            records.append(
                {
                    "stage": "class-pass",
                    "pass": pass_index,
                    "class_id": class_id,
                    "class_name": taxonomy.names[class_id],
                    "selected_threshold": selected_threshold,
                    "accepted": accepted,
                    "objective": _objective_dict(selected_objective),
                }
            )

    return ThresholdOptimizationResult(
        thresholds=threshold_map,
        report=best_report,
        objective=best_objective,
        baseline_objective=baseline_objective,
        global_threshold=global_threshold,
        grid=grid,
        recall_floor=recall_floor,
        candidates=tuple(records),
    )
```

- [ ] **Step 5: Run optimizer tests and confirm green**

Run:

```powershell
python -m pytest tests/test_thresholds.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add src/xh_detect/thresholds.py tests/test_thresholds.py
git commit -m "feat: optimize per-class thresholds"
```

## Task 3: Artifact Writing And Baseline Objective Loading

**Files:**
- Modify: `src/xh_detect/thresholds.py`
- Modify: `tests/test_thresholds.py`

- [ ] **Step 1: Add failing artifact tests**

Append these tests to `tests/test_thresholds.py`:

```python
import json
import yaml

from xh_detect.thresholds import load_report_objective, write_threshold_artifacts


def test_load_report_objective_reads_existing_evaluation_report(tmp_path: Path) -> None:
    report_path = tmp_path / "baseline.json"
    report_path.write_text(
        json.dumps(
            {
                "overall_class_agnostic": {
                    "tp": 8,
                    "fp": 2,
                    "fn": 2,
                    "recall": 0.8,
                    "fdr": 0.2,
                }
            }
        ),
        encoding="utf-8",
    )

    objective = load_report_objective(report_path)

    assert objective.f1 == pytest.approx(0.8)
    assert objective.tp == 8


def test_write_threshold_artifacts_writes_yaml_json_and_markdown(tmp_path: Path) -> None:
    taxonomy = get_taxonomy("legacy3")
    truth = [ObjectAnnotation("img", 0, BOX)]
    predictions = [
        Detection("img", 0, 0.90, BOX),
        Detection("fp", 0, 0.20, BOX),
    ]
    result = optimize_thresholds(
        predictions,
        truth,
        taxonomy=taxonomy,
        thresholds=(0.20, 0.50),
    )

    artifacts = write_threshold_artifacts(
        result,
        output_dir=tmp_path,
        taxonomy=taxonomy,
        experiment_name="unit-thresholds",
    )

    thresholds_yaml = yaml.safe_load((tmp_path / "optimized-thresholds.yaml").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "search-summary.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "search-summary.md").read_text(encoding="utf-8")

    assert artifacts["report"] == tmp_path / "report.json"
    assert thresholds_yaml["class_thresholds"][0] == 0.5
    assert report["overall_class_agnostic"]["tp"] == 1
    assert summary["experiment_name"] == "unit-thresholds"
    assert summary["thresholds"]["0"] == 0.5
    assert "| ship |" in markdown
    assert "Recommendation" in markdown
```

Also add this import near the top of `tests/test_thresholds.py`:

```python
from pathlib import Path
```

- [ ] **Step 2: Run artifact tests and confirm red**

Run:

```powershell
python -m pytest tests/test_thresholds.py::test_load_report_objective_reads_existing_evaluation_report tests/test_thresholds.py::test_write_threshold_artifacts_writes_yaml_json_and_markdown -q
```

Expected: FAIL with missing `load_report_objective` and `write_threshold_artifacts`.

- [ ] **Step 3: Add report objective loading**

Append this code to `src/xh_detect/thresholds.py`:

```python
def _metrics_from_report_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    metrics = payload.get("overall_class_agnostic")
    if not isinstance(metrics, Mapping):
        raise ValueError("report must contain object field 'overall_class_agnostic'")
    return metrics


def _metric_number(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"report metric {key!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"report metric {key!r} must be finite")
    return number


def _metric_count(metrics: Mapping[str, object], key: str) -> int:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"report metric {key!r} must be a non-negative integer")
    return int(value)


def load_report_objective(path: Path | str) -> ObjectiveScore:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("report JSON root must be an object")
    metrics = _metrics_from_report_payload(payload)
    recall = _metric_number(metrics, "recall")
    fdr = _metric_number(metrics, "fdr")
    return ObjectiveScore(
        f1=f1_score(recall=recall, fdr=fdr),
        precision=1.0 - fdr,
        recall=recall,
        fdr=fdr,
        tp=_metric_count(metrics, "tp"),
        fp=_metric_count(metrics, "fp"),
        fn=_metric_count(metrics, "fn"),
    )
```

- [ ] **Step 4: Add artifact writers**

Append this code to `src/xh_detect/thresholds.py`:

```python
def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _thresholds_by_coarse_group(
    thresholds: Mapping[int, float],
    taxonomy: Taxonomy,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for class_id in sorted(taxonomy.valid_ids):
        group = taxonomy.coarse_name(class_id)
        grouped.setdefault(group, {})[f"{class_id}:{taxonomy.names[class_id]}"] = thresholds[class_id]
    return grouped


def _recommendation(result: ThresholdOptimizationResult) -> str:
    if result.baseline_objective is None:
        return "Use this threshold set if its comparison report beats the current global-threshold run."
    if result.objective.f1 > result.baseline_objective.f1 and _passes_recall_floor(
        result.objective,
        result.recall_floor,
    ):
        return "Keep the threshold-only configuration and use it for the next MKSNet-Lite comparison."
    if result.objective.f1 > result.baseline_objective.f1:
        return "Review Recall before adopting this threshold set because it misses the configured floor."
    return "Do not adopt this threshold set as the main result; move to ship-focused data or training work."


def _render_search_markdown(
    result: ThresholdOptimizationResult,
    taxonomy: Taxonomy,
    experiment_name: str,
) -> str:
    lines = [
        f"# {experiment_name} Threshold Optimization",
        "",
        "## Overall",
        "",
        "| F1 | Precision | Recall | FDR | TP | FP | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {result.objective.f1:.6f} | {result.objective.precision:.6f} | "
            f"{result.objective.recall:.6f} | {result.objective.fdr:.6f} | "
            f"{result.objective.tp} | {result.objective.fp} | {result.objective.fn} |"
        ),
        "",
        "## Thresholds By Coarse Group",
        "",
        "| Group | Class | Threshold |",
        "| --- | --- | ---: |",
    ]
    for group, classes in _thresholds_by_coarse_group(result.thresholds, taxonomy).items():
        for label, threshold in classes.items():
            lines.append(f"| {group} | {label} | {threshold:.2f} |")

    ship = result.report.by_coarse_class.get("ship")
    if ship is not None:
        lines.extend(
            [
                "",
                "## Ship Check",
                "",
                f"- Recall: {ship.recall:.6f}",
                f"- FDR: {ship.fdr:.6f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            _recommendation(result),
            "",
        ]
    )
    return "\n".join(lines)


def write_threshold_artifacts(
    result: ThresholdOptimizationResult,
    *,
    output_dir: Path,
    taxonomy: Taxonomy,
    experiment_name: str,
    baseline_report: Path | None = None,
    baseline_name: str = "xh25-yolo26s-e80",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = output_dir / "optimized-thresholds.yaml"
    report_path = output_dir / "report.json"
    summary_json_path = output_dir / "search-summary.json"
    summary_md_path = output_dir / "search-summary.md"

    thresholds_path.write_text(
        yaml.safe_dump(
            {
                "class_thresholds": {
                    class_id: result.thresholds[class_id]
                    for class_id in sorted(result.thresholds)
                }
            },
            sort_keys=True,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    _write_json(report_path, report_to_dict(result.report))
    _write_json(
        summary_json_path,
        {
            "experiment_name": experiment_name,
            "global_threshold": result.global_threshold,
            "grid": list(result.grid),
            "recall_floor": result.recall_floor,
            "objective": _objective_dict(result.objective),
            "baseline_objective": (
                None
                if result.baseline_objective is None
                else _objective_dict(result.baseline_objective)
            ),
            "thresholds": {
                str(class_id): result.thresholds[class_id]
                for class_id in sorted(result.thresholds)
            },
            "candidates": list(result.candidates),
        },
    )
    summary_md_path.write_text(
        _render_search_markdown(result, taxonomy, experiment_name),
        encoding="utf-8",
    )

    artifacts = {
        "thresholds": thresholds_path,
        "report": report_path,
        "search_summary_json": summary_json_path,
        "search_summary_md": summary_md_path,
    }
    if baseline_report is not None:
        compare_experiments(
            baseline_report=baseline_report,
            experiment_report=report_path,
            output_dir=output_dir,
            baseline_name=baseline_name,
            experiment_name=experiment_name,
        )
        artifacts["comparison_json"] = output_dir / "comparison.json"
        artifacts["comparison_md"] = output_dir / "comparison.md"
    return artifacts
```

- [ ] **Step 5: Run artifact tests and confirm green**

Run:

```powershell
python -m pytest tests/test_thresholds.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add src/xh_detect/thresholds.py tests/test_thresholds.py
git commit -m "feat: write threshold optimization artifacts"
```

## Task 4: CLI Command

**Files:**
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI forwarding test**

Append this test to `tests/test_cli.py` near the evaluation and sweep command tests:

```python
@patch("xh_detect.cli.write_threshold_artifacts")
@patch("xh_detect.cli.optimize_thresholds_search")
@patch("xh_detect.cli.load_report_objective")
@patch("xh_detect.cli.load_coco_ground_truth", return_value=["truth"])
@patch("xh_detect.cli.load_coco_predictions", return_value=["prediction"])
def test_optimize_thresholds_command_forwards_options(
    load_predictions: Mock,
    load_truth: Mock,
    load_report_objective: Mock,
    optimize_thresholds_search: Mock,
    write_threshold_artifacts: Mock,
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "optimized"
    predictions.write_text("[]", encoding="utf-8")
    truth.write_text('{"annotations":[]}', encoding="utf-8")
    baseline.write_text('{"overall_class_agnostic":{"tp":1,"fp":0,"fn":0,"recall":1.0,"fdr":0.0}}', encoding="utf-8")
    baseline_objective = SimpleNamespace(recall=1.0)
    optimized = SimpleNamespace()
    load_report_objective.return_value = baseline_objective
    optimize_thresholds_search.return_value = optimized

    result = CliRunner().invoke(
        app,
        [
            "optimize-thresholds",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--baseline-report",
            str(baseline),
            "--taxonomy",
            "xh25",
            "--output-dir",
            str(output),
            "--experiment-name",
            "custom-thresholds",
            "--thresholds",
            "0.20,0.50",
            "--recall-floor-delta",
            "0.01",
            "--tie-epsilon",
            "0.001",
        ],
    )

    taxonomy = get_taxonomy("xh25")
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(output / "report.json")
    load_predictions.assert_called_once_with(predictions, taxonomy=taxonomy)
    load_truth.assert_called_once_with(truth, taxonomy=taxonomy)
    load_report_objective.assert_called_once_with(baseline)
    optimize_thresholds_search.assert_called_once_with(
        ["prediction"],
        ["truth"],
        taxonomy=taxonomy,
        thresholds=[0.2, 0.5],
        baseline_objective=baseline_objective,
        recall_floor_delta=0.01,
        tie_epsilon=0.001,
    )
    write_threshold_artifacts.assert_called_once_with(
        optimized,
        output_dir=output,
        taxonomy=taxonomy,
        experiment_name="custom-thresholds",
        baseline_report=baseline,
    )
```

- [ ] **Step 2: Add failing CLI validation test**

Append this test to `tests/test_cli.py`:

```python
def test_optimize_thresholds_command_reports_invalid_threshold_grid(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    output = tmp_path / "optimized"
    predictions.write_text("[]", encoding="utf-8")
    truth.write_text('{"annotations":[]}', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "optimize-thresholds",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--output-dir",
            str(output),
            "--thresholds",
            "0.25,bad",
        ],
    )

    assert result.exit_code != 0
    assert "invalid threshold value" in result.output
    assert "Traceback" not in result.output
```

- [ ] **Step 3: Run CLI tests and confirm red**

Run:

```powershell
python -m pytest tests/test_cli.py::test_optimize_thresholds_command_forwards_options tests/test_cli.py::test_optimize_thresholds_command_reports_invalid_threshold_grid -q
```

Expected: FAIL because the command and imported symbols do not exist yet.

- [ ] **Step 4: Add threshold imports to CLI**

In `src/xh_detect/cli.py`, add this import block after the existing evaluator imports:

```python
from xh_detect.thresholds import (
    DEFAULT_THRESHOLD_GRID_TEXT,
    load_report_objective,
    optimize_thresholds as optimize_thresholds_search,
    parse_threshold_grid,
    write_threshold_artifacts,
)
```

- [ ] **Step 5: Add `optimize-thresholds` command**

Add this command after `sweep_thresholds_command` and before `compare_experiments_command`:

```python
@app.command("optimize-thresholds")
def optimize_thresholds_command(
    predictions_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    ground_truth_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    output_dir: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/mksnet-lite/threshold-optimized"
    ),
    taxonomy: Annotated[str, typer.Option()] = "xh25",
    baseline_report: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    experiment_name: Annotated[str, typer.Option()] = "xh25-mksnet-lite-threshold-optimized",
    thresholds: Annotated[str, typer.Option()] = DEFAULT_THRESHOLD_GRID_TEXT,
    recall_floor_delta: Annotated[float, typer.Option(min=0.0)] = 0.003,
    tie_epsilon: Annotated[float, typer.Option(min=0.0)] = 0.0005,
) -> None:
    default_baseline = Path("outputs/xh25/baseline/report.json")
    resolved_baseline = baseline_report
    if resolved_baseline is None and default_baseline.is_file():
        resolved_baseline = default_baseline
    if resolved_baseline is not None and not resolved_baseline.is_file():
        raise typer.BadParameter(f"baseline report does not exist: {resolved_baseline}")

    try:
        threshold_grid = parse_threshold_grid(thresholds)
        taxonomy_object = get_taxonomy(taxonomy)
        predictions = load_coco_predictions(predictions_json, taxonomy=taxonomy_object)
        truth = load_coco_ground_truth(ground_truth_json, taxonomy=taxonomy_object)
        baseline_objective = (
            None
            if resolved_baseline is None
            else load_report_objective(resolved_baseline)
        )
        result = optimize_thresholds_search(
            predictions,
            truth,
            taxonomy=taxonomy_object,
            thresholds=threshold_grid,
            baseline_objective=baseline_objective,
            recall_floor_delta=recall_floor_delta,
            tie_epsilon=tie_epsilon,
        )
        write_threshold_artifacts(
            result,
            output_dir=output_dir,
            taxonomy=taxonomy_object,
            experiment_name=experiment_name,
            baseline_report=resolved_baseline,
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(str(output_dir / "report.json"))
```

- [ ] **Step 6: Run CLI tests and confirm green**

Run:

```powershell
python -m pytest tests/test_cli.py::test_optimize_thresholds_command_forwards_options tests/test_cli.py::test_optimize_thresholds_command_reports_invalid_threshold_grid -q
```

Expected: PASS.

- [ ] **Step 7: Run threshold and CLI focused suite**

Run:

```powershell
python -m pytest tests/test_thresholds.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

Run:

```powershell
git add src/xh_detect/cli.py tests/test_cli.py
git commit -m "feat: add threshold optimization cli"
```

## Task 5: Documentation And Local Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README workflow**

Add this subsection after the existing MKSNet-Lite comparison commands in `README.md`:

````markdown
### 阈值优化

MKSNet-Lite 的 80 epoch 结果显示全局阈值 `0.30` 比 `0.25` 更稳，但 ship 类仍是主要短板。
可以在不重新训练的情况下，用验证集预测搜索逐类别置信度阈值：

```bash
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/mksnet-lite/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --baseline-report outputs/xh25/baseline/report.json \
  --taxonomy xh25 \
  --output-dir outputs/xh25/mksnet-lite/threshold-optimized
```

输出目录包含：

- `optimized-thresholds.yaml`：可复制到 `configs/xh25-mksnet-lite.yaml` 的 `class_thresholds`；
- `report.json`：优化阈值后的验证集评估；
- `comparison.json` 和 `comparison.md`：和 main 线 baseline 的对比；
- `search-summary.json` 和 `search-summary.md`：搜索网格、选择原因和 ship 类检查。
````

- [ ] **Step 2: Run focused tests**

Run:

```powershell
python -m pytest tests/test_thresholds.py tests/test_cli.py tests/test_compare.py tests/test_evaluator.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full verification**

Run:

```powershell
python -m pytest -q
python -m ruff format --check .
python -m ruff check .
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Commit Task 5**

Run:

```powershell
git add README.md
git commit -m "docs: document threshold optimization workflow"
```

## Task 6: Run The Optimizer On The 3090 Server

**Files:**
- No source edits expected.

- [ ] **Step 1: Sync the branch to the server**

From the local worktree, create and upload a fresh bundle:

```powershell
git bundle create C:\Users\feng\project\fight\mksnet-lite-threshold.bundle codex/mksnet-lite
```

On the server, fetch the bundle into `/root/XH-202625-remote-sensing-detection`:

```bash
cd /root/XH-202625-remote-sensing-detection
git fetch /root/mksnet-lite-threshold.bundle codex/mksnet-lite:codex/mksnet-lite
git checkout codex/mksnet-lite
```

Expected: server branch points at the local `codex/mksnet-lite` head.

- [ ] **Step 2: Reinstall the editable package with mirror settings**

Run on the server:

```bash
cd /root/XH-202625-remote-sensing-detection
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
.venv/bin/python -m pip install -e ".[dev]"
```

Expected: install exits 0 and keeps the existing PyTorch 2.7 NVIDIA image packages intact.

- [ ] **Step 3: Run threshold optimization**

Run on the server:

```bash
cd /root/XH-202625-remote-sensing-detection
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/mksnet-lite/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --baseline-report outputs/xh25/baseline/report.json \
  --taxonomy xh25 \
  --output-dir outputs/xh25/mksnet-lite/threshold-optimized
```

Expected: command prints `outputs/xh25/mksnet-lite/threshold-optimized/report.json` and writes all six artifacts listed in the README.

- [ ] **Step 4: Inspect the optimized metrics**

Run on the server:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/xh25/mksnet-lite/threshold-optimized")
summary = json.loads((root / "search-summary.json").read_text())
comparison = json.loads((root / "comparison.json").read_text())
print(json.dumps({
    "f1": summary["objective"]["f1"],
    "recall": summary["objective"]["recall"],
    "fdr": summary["objective"]["fdr"],
    "baseline_recall_delta": comparison["overall"]["recall_delta"],
    "baseline_fdr_delta": comparison["overall"]["fdr_delta"],
}, indent=2, ensure_ascii=False))
PY
```

Expected: output shows whether optimized F1 beats `0.963897`, and whether Recall stays within `0.003` of the baseline.

- [ ] **Step 5: Download artifacts locally**

Copy these files into `C:\Users\feng\project\fight\.worktrees\mksnet-lite\outputs\server\mksnet-lite\threshold-optimized\`:

```text
optimized-thresholds.yaml
report.json
comparison.json
comparison.md
search-summary.json
search-summary.md
```

Expected: local `outputs/server/mksnet-lite/threshold-optimized/` contains the server run artifacts for discussion with the team.

- [ ] **Step 6: Commit downloaded report artifacts if the team wants them versioned**

Check whether `outputs/` is ignored. If report artifacts are intentionally not tracked, skip this commit and summarize the server result in the final response. If the team wants the report committed, run:

```powershell
git add outputs/server/mksnet-lite/threshold-optimized
git commit -m "results: add threshold optimization report"
```

Expected: either no commit is made because outputs are ignored, or a small results commit records only report artifacts.

## Self-Review

- Spec coverage: Task 1 covers threshold filtering and objective ordering. Task 2 covers deterministic per-class greedy search, global-threshold initialization, two class passes, tie handling, and Recall floor behavior. Task 3 covers YAML, JSON, Markdown, baseline comparison, and ship visibility. Task 4 exposes the requested CLI with defaults and Typer-friendly validation. Task 5 documents the workflow. Task 6 runs the final experiment on the 3090 server using mirror settings.
- Red-flag scan: Every task has exact paths, code, commands, and expected outcomes.
- Type consistency: The plan consistently uses `ObjectiveScore`, `ThresholdOptimizationResult`, `optimize_thresholds_search` in CLI, `class_thresholds` in YAML, and the existing `EvaluationReport` plus `Metrics` evaluator types.
