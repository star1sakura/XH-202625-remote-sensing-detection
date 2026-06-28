from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from xh_detect.evaluator import EvaluationReport
from xh_detect.taxonomy import Taxonomy
from xh_detect.types import Detection

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

    return {
        _class_id_from_key(class_id, taxonomy): _validate_threshold(
            threshold,
            f"class {class_id!r}",
        )
        for class_id, threshold in thresholds.items()
    }


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
