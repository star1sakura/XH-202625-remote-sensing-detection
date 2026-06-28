from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from xh_detect.evaluator import EvaluationReport, evaluate
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
            "objective": best_global.objective,
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
                    "objective": selected.objective,
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
