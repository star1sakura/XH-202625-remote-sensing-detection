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

DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (
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
DEFAULT_THRESHOLD_GRID_TEXT = ",".join(f"{threshold:.2f}" for threshold in DEFAULT_THRESHOLD_GRID)
_REPORT_METRIC_TOLERANCE = 1e-9


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
    grid: list[float]
    recall_floor: float | None
    candidates: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _ThresholdCandidate:
    threshold: float
    report: EvaluationReport
    objective: ObjectiveScore


def _validate_threshold(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} threshold must be a finite real number")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{context} threshold must be in [0, 1]")
    return threshold


def parse_threshold_grid(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        thresholds = []
        for index, part in enumerate(value.split(",")):
            text = part.strip()
            if not text:
                raise ValueError("threshold grid must not contain empty values")
            try:
                threshold = float(text)
            except ValueError:
                raise ValueError(f"threshold grid value {index} must be numeric") from None
            thresholds.append(_validate_threshold(threshold, f"threshold grid value {index}"))
    elif isinstance(value, Sequence):
        thresholds = [
            _validate_threshold(threshold, f"threshold grid value {index}")
            for index, threshold in enumerate(value)
        ]
    else:
        raise TypeError("threshold grid must be a comma-separated string or sequence")

    if not thresholds:
        raise ValueError("threshold grid must contain at least one value")
    return sorted(set(thresholds))


def _class_id_from_key(value: object, taxonomy: Taxonomy) -> int:
    if isinstance(value, str):
        if not value.isdecimal():
            raise TypeError("class ID must be an integer or decimal string")
        class_id = int(value)
    else:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("class ID must be an integer or decimal string")
        class_id = int(value)

    if class_id not in taxonomy.valid_ids:
        valid_ids = ", ".join(str(item) for item in sorted(taxonomy.valid_ids))
        raise ValueError(f"class ID must be one of {valid_ids}")
    return class_id


def validate_threshold_map(
    thresholds: Mapping[object, object],
    taxonomy: Taxonomy,
) -> dict[int, float]:
    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be a mapping")

    validated_thresholds: dict[int, float] = {}
    for raw_class_id, threshold in thresholds.items():
        class_id = _class_id_from_key(raw_class_id, taxonomy)
        if class_id in validated_thresholds:
            raise ValueError(f"duplicate class ID: {class_id}")
        validated_thresholds[class_id] = _validate_threshold(
            threshold,
            f"class {raw_class_id!r}",
        )
    return validated_thresholds


def filter_predictions_by_class_threshold(
    predictions: Sequence[Detection],
    thresholds: Mapping[object, object],
    taxonomy: Taxonomy,
) -> list[Detection]:
    threshold_by_class = validate_threshold_map(thresholds, taxonomy)
    filtered: list[Detection] = []
    for prediction in predictions:
        class_id = _class_id_from_key(prediction.class_id, taxonomy)
        if prediction.score >= threshold_by_class.get(class_id, 0.0):
            filtered.append(prediction)
    return filtered


def f1_score(recall: float, fdr: float) -> float:
    precision = 1.0 - fdr
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


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


def _load_report_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return payload


def _metric_section(report: Mapping[str, object], key: str, label: str) -> Mapping[str, object]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain object field {key!r}")
    return value


def _required_metric(metrics: Mapping[str, object], key: str, label: str) -> object:
    if key not in metrics:
        raise ValueError(f"{label} missing required metric {key!r}")
    return metrics[key]


def _report_count(metrics: Mapping[str, object], key: str, label: str) -> int:
    value = _required_metric(metrics, key, label)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} metric {key!r} must be a non-negative integer")
    count = int(value)
    if count < 0:
        raise ValueError(f"{label} metric {key!r} must be a non-negative integer")
    return count


def _report_metric(metrics: Mapping[str, object], key: str, label: str) -> float:
    value = _required_metric(metrics, key, label)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} metric {key!r} must be finite numeric")
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"{label} metric {key!r} must be finite numeric")
    return metric


def _report_probability_metric(metrics: Mapping[str, object], key: str, label: str) -> float:
    metric = _report_metric(metrics, key, label)
    if not 0.0 <= metric <= 1.0:
        raise ValueError(f"{label} metric {key!r} must be in [0, 1]")
    return metric


def _validate_report_metric_consistency(
    metric_name: str,
    loaded_value: float,
    expected_value: float,
    label: str,
) -> None:
    if not math.isclose(
        loaded_value,
        expected_value,
        rel_tol=0.0,
        abs_tol=_REPORT_METRIC_TOLERANCE,
    ):
        raise ValueError(f"{label} metric {metric_name!r} is inconsistent with counts")


def load_report_objective(path: Path | str) -> ObjectiveScore:
    report_path = Path(path)
    report = _load_report_json_object(report_path, "report")
    metrics = _metric_section(report, "overall_class_agnostic", "report")
    tp = _report_count(metrics, "tp", "overall_class_agnostic")
    fp = _report_count(metrics, "fp", "overall_class_agnostic")
    fn = _report_count(metrics, "fn", "overall_class_agnostic")
    recall = _report_probability_metric(metrics, "recall", "overall_class_agnostic")
    fdr = _report_probability_metric(metrics, "fdr", "overall_class_agnostic")
    expected_recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    expected_fdr = fp / (fp + tp) if fp + tp > 0 else 0.0
    _validate_report_metric_consistency(
        "recall",
        loaded_value=recall,
        expected_value=expected_recall,
        label="overall_class_agnostic",
    )
    _validate_report_metric_consistency(
        "fdr",
        loaded_value=fdr,
        expected_value=expected_fdr,
        label="overall_class_agnostic",
    )
    precision = 1.0 - fdr
    return ObjectiveScore(
        f1=f1_score(recall=recall, fdr=fdr),
        precision=precision,
        recall=recall,
        fdr=fdr,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def _json_primitive(value: object) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON artifact values must be finite")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_primitive(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_primitive(item) for item in value]
    raise TypeError(f"JSON artifact value is not serializable: {type(value).__name__}")


def _write_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(_json_primitive(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _format_number(value: object) -> str:
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _metrics_markdown_row(label: str, objective: ObjectiveScore) -> str:
    return (
        f"| {label} | {_format_number(objective.f1)} | "
        f"{_format_number(objective.precision)} | {_format_number(objective.recall)} | "
        f"{_format_number(objective.fdr)} | {objective.tp} | {objective.fp} | {objective.fn} |"
    )


def _metric_mapping_row(label: str, metrics: Mapping[str, object]) -> str:
    return (
        f"| {label} | {_format_number(metrics['recall'])} | "
        f"{_format_number(metrics['fdr'])} | {metrics['tp']} | {metrics['fp']} | "
        f"{metrics['fn']} |"
    )


def _recommendation(result: ThresholdOptimizationResult, baseline: ObjectiveScore | None) -> str:
    if result.recall_floor is not None and result.objective.recall < result.recall_floor:
        return "Review before adoption: no candidate satisfied the configured recall floor."
    if baseline is None:
        return (
            "Use these thresholds as the optimized candidate and compare against a "
            "baseline report."
        )
    if is_better_objective(result.objective, baseline):
        return "Adopt the optimized thresholds for validation against the next benchmark run."
    return "Review before adoption: the optimized objective did not improve over the baseline."


def _render_search_summary_markdown(
    result: ThresholdOptimizationResult,
    *,
    taxonomy: Taxonomy,
    experiment_name: str,
    baseline_objective: ObjectiveScore | None,
) -> str:
    report = report_to_dict(result.report)
    lines = [
        "# Threshold Optimization Summary",
        "",
        f"- Experiment: {experiment_name}",
        f"- Global threshold: {_format_number(result.global_threshold)}",
        f"- Grid: {', '.join(_format_number(threshold) for threshold in result.grid)}",
        (
            f"- Recall floor: {_format_number(result.recall_floor)}"
            if result.recall_floor is not None
            else "- Recall floor: none"
        ),
        "",
        "## Overall Metrics",
        "",
        "| Run | F1 | Precision | Recall | FDR | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if baseline_objective is not None:
        lines.append(_metrics_markdown_row("Baseline", baseline_objective))
    lines.extend(
        [
            _metrics_markdown_row("Optimized", result.objective),
            "",
            "## Thresholds By Coarse Class",
            "",
            "| Coarse class | Class ID | Class name | Threshold |",
            "| --- | ---: | --- | ---: |",
        ]
    )

    for class_id, threshold in sorted(
        result.thresholds.items(),
        key=lambda item: (taxonomy.coarse_name(item[0]), item[0]),
    ):
        lines.append(
            f"| {taxonomy.coarse_name(class_id)} | {class_id} | {taxonomy.names[class_id]} | "
            f"{_format_number(threshold)} |"
        )

    coarse_metrics = report.get("by_coarse_class")
    if isinstance(coarse_metrics, Mapping) and isinstance(coarse_metrics.get("ship"), Mapping):
        lines.extend(
            [
                "",
                "## Ship Check",
                "",
                "| Coarse class | Recall | FDR | TP | FP | FN |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                _metric_mapping_row("ship", coarse_metrics["ship"]),
            ]
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            _recommendation(result, baseline_objective),
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
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    thresholds_path = target_dir / "optimized-thresholds.yaml"
    report_path = target_dir / "report.json"
    summary_json_path = target_dir / "search-summary.json"
    summary_md_path = target_dir / "search-summary.md"

    thresholds_payload = {
        "class_thresholds": {
            class_id: result.thresholds[class_id] for class_id in sorted(result.thresholds)
        }
    }
    thresholds_path.write_text(
        yaml.safe_dump(thresholds_payload, sort_keys=False),
        encoding="utf-8",
    )

    report_payload = report_to_dict(result.report)
    _write_json_artifact(report_path, report_payload)

    baseline_objective = (
        load_report_objective(baseline_report)
        if baseline_report is not None
        else result.baseline_objective
    )

    summary_payload: dict[str, object] = {
        "experiment_name": experiment_name,
        "global_threshold": result.global_threshold,
        "grid": result.grid,
        "recall_floor": result.recall_floor,
        "objective": _objective_dict(result.objective),
        "baseline_objective": (
            _objective_dict(baseline_objective) if baseline_objective is not None else None
        ),
        "thresholds": {
            str(class_id): result.thresholds[class_id] for class_id in sorted(result.thresholds)
        },
        "candidates": list(result.candidates),
    }
    _write_json_artifact(summary_json_path, summary_payload)
    summary_md_path.write_text(
        _render_search_summary_markdown(
            result,
            taxonomy=taxonomy,
            experiment_name=experiment_name,
            baseline_objective=baseline_objective,
        ),
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
            baseline_report=Path(baseline_report),
            experiment_report=report_path,
            output_dir=target_dir,
            baseline_name=baseline_name,
            experiment_name=experiment_name,
        )
        artifacts["comparison_json"] = target_dir / "comparison.json"
        artifacts["comparison_md"] = target_dir / "comparison.md"

    return artifacts


def is_better_objective(
    candidate: ObjectiveScore,
    incumbent: ObjectiveScore,
    tie_epsilon: float = 0.0005,
) -> bool:
    if candidate.f1 > incumbent.f1 + tie_epsilon:
        return True
    if candidate.f1 < incumbent.f1 - tie_epsilon:
        return False

    if candidate.fdr < incumbent.fdr - tie_epsilon:
        return True
    if candidate.fdr > incumbent.fdr + tie_epsilon:
        return False

    return candidate.recall > incumbent.recall + tie_epsilon


def _validate_non_negative_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return number


def _validate_passes(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("passes must be an integer greater than or equal to 1")
    passes = int(value)
    if passes < 1:
        raise ValueError("passes must be an integer greater than or equal to 1")
    return passes


def _evaluate_threshold_map(
    predictions: Sequence[Detection],
    ground_truth: Sequence[ObjectAnnotation],
    thresholds: Mapping[int, float],
    taxonomy: Taxonomy,
) -> tuple[EvaluationReport, ObjectiveScore]:
    filtered = filter_predictions_by_class_threshold(predictions, thresholds, taxonomy)
    report = evaluate(filtered, ground_truth, taxonomy=taxonomy)
    return report, objective_from_report(report)


def _candidate_meets_recall_floor(
    candidate: _ThresholdCandidate,
    recall_floor: float | None,
) -> bool:
    return recall_floor is None or candidate.objective.recall >= recall_floor


def _select_best_candidate(
    candidates: Sequence[_ThresholdCandidate],
    recall_floor: float | None,
    tie_epsilon: float,
) -> _ThresholdCandidate:
    eligible = [
        candidate
        for candidate in candidates
        if _candidate_meets_recall_floor(candidate, recall_floor)
    ]
    pool = eligible if eligible else candidates

    best = pool[0]
    for candidate in pool[1:]:
        if is_better_objective(candidate.objective, best.objective, tie_epsilon):
            best = candidate
    return best


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
    recall_delta = _validate_non_negative_real(recall_floor_delta, "recall_floor_delta")
    validated_tie_epsilon = _validate_non_negative_real(tie_epsilon, "tie_epsilon")
    validated_passes = _validate_passes(passes)
    grid = parse_threshold_grid(thresholds)
    prediction_items = list(predictions)
    truth_items = list(ground_truth)
    recall_floor = (
        max(0.0, baseline_objective.recall - recall_delta)
        if baseline_objective is not None
        else None
    )

    global_candidates: list[_ThresholdCandidate] = []
    valid_class_ids = sorted(taxonomy.valid_ids)
    for threshold in grid:
        threshold_map = dict.fromkeys(valid_class_ids, threshold)
        report, objective = _evaluate_threshold_map(
            prediction_items,
            truth_items,
            threshold_map,
            taxonomy,
        )
        global_candidates.append(_ThresholdCandidate(threshold, report, objective))

    best_global = _select_best_candidate(
        global_candidates,
        recall_floor,
        validated_tie_epsilon,
    )
    current_thresholds = dict.fromkeys(valid_class_ids, best_global.threshold)
    current_report = best_global.report
    current_objective = best_global.objective
    candidate_records: list[dict[str, object]] = [
        {
            "stage": "global",
            "selected_threshold": best_global.threshold,
            "accepted": True,
            "objective": _objective_dict(best_global.objective),
            "recall_floor_satisfied": _candidate_meets_recall_floor(
                best_global,
                recall_floor,
            ),
        }
    ]

    for pass_number in range(1, validated_passes + 1):
        for class_id in valid_class_ids:
            class_candidates: list[_ThresholdCandidate] = []
            for threshold in grid:
                candidate_thresholds = dict(current_thresholds)
                candidate_thresholds[class_id] = threshold
                report, objective = _evaluate_threshold_map(
                    prediction_items,
                    truth_items,
                    candidate_thresholds,
                    taxonomy,
                )
                class_candidates.append(_ThresholdCandidate(threshold, report, objective))

            selected = _select_best_candidate(
                class_candidates,
                recall_floor,
                validated_tie_epsilon,
            )
            accepted = is_better_objective(
                selected.objective,
                current_objective,
                validated_tie_epsilon,
            )
            if accepted:
                current_thresholds[class_id] = selected.threshold
                current_report = selected.report
                current_objective = selected.objective

            candidate_records.append(
                {
                    "stage": "class",
                    "pass": pass_number,
                    "class_id": class_id,
                    "class_name": taxonomy.names[class_id],
                    "selected_threshold": selected.threshold,
                    "accepted": accepted,
                    "objective": _objective_dict(selected.objective),
                    "recall_floor_satisfied": _candidate_meets_recall_floor(
                        selected,
                        recall_floor,
                    ),
                }
            )

    return ThresholdOptimizationResult(
        thresholds=current_thresholds,
        report=current_report,
        objective=current_objective,
        baseline_objective=baseline_objective,
        global_threshold=best_global.threshold,
        grid=grid,
        recall_floor=recall_floor,
        candidates=tuple(candidate_records),
    )
