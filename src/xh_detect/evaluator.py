from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TypeVar

from xh_detect.geometry import hbb_iou, obb_to_hbb
from xh_detect.taxonomy import Taxonomy, get_taxonomy
from xh_detect.types import Detection, ObjectAnnotation, Polygon4

KeyT = TypeVar("KeyT", bound=Hashable)


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        for name, value in (("tp", self.tp), ("fp", self.fp), ("fn", self.fn)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def fdr(self) -> float:
        denominator = self.fp + self.tp
        return self.fp / denominator if denominator else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    overall_class_agnostic: Metrics
    by_coarse_class: dict[str, Metrics]
    by_fine_class: dict[int, Metrics]
    by_image: dict[str, Metrics]

    @property
    def overall(self) -> Metrics:
        return self.overall_class_agnostic

    @property
    def by_class(self) -> dict[int, Metrics]:
        return self.by_fine_class


@dataclass(frozen=True)
class FalsePositiveSources:
    overlap: int
    background: int

    def __post_init__(self) -> None:
        for name, value in (("overlap", self.overlap), ("background", self.background)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def total(self) -> int:
        return self.overlap + self.background


@dataclass(frozen=True)
class FalsePositiveAudit:
    by_coarse_class: dict[str, FalsePositiveSources]


def _validate_class_id(value: object, context: str, taxonomy: Taxonomy) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{context} category_id must be an integer")
    class_id = int(value)
    if class_id not in taxonomy.valid_ids:
        valid_ids = ", ".join(str(item) for item in sorted(taxonomy.valid_ids))
        raise ValueError(f"{context} category_id must be one of {valid_ids}")
    return class_id


def _validate_score(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} score must be a finite real number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{context} score must be in [0, 1]")
    return score


def _normalize_image_id(value: object, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise TypeError(f"{context} image_id must be a string or integer")
    normalized = str(value)
    if not normalized:
        raise ValueError(f"{context} image_id must not be empty")
    return normalized


def _bbox_to_polygon(value: object, context: str) -> Polygon4:
    if type(value) is not list or len(value) != 4:
        raise ValueError(f"{context} bbox must be a four-item list")
    coordinates: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{context} bbox[{index}] must be a finite real number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{context} bbox[{index}] must be finite")
        coordinates.append(number)
    x, y, width, height = coordinates
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{context} bbox width and height must be positive")
    return (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )


def _validate_polygon(polygon: Polygon4, context: str) -> None:
    coordinates = [coordinate for point in polygon for coordinate in point]
    if len(coordinates) != 8 or not all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in coordinates
    ):
        raise ValueError(f"{context} polygon must contain four finite points")


def _validate_detection(item: Detection, index: int, taxonomy: Taxonomy) -> None:
    if not isinstance(item.image_id, str) or not item.image_id:
        raise ValueError(f"prediction {index} image_id must be a non-empty string")
    _validate_class_id(item.class_id, f"prediction {index}", taxonomy)
    _validate_score(item.score, f"prediction {index}")
    _validate_polygon(item.polygon, f"prediction {index}")


def _validate_truth(item: ObjectAnnotation, index: int, taxonomy: Taxonomy) -> None:
    if not isinstance(item.image_id, str) or not item.image_id:
        raise ValueError(f"ground truth {index} image_id must be a non-empty string")
    _validate_class_id(item.class_id, f"ground truth {index}", taxonomy)
    _validate_polygon(item.polygon, f"ground truth {index}")


def _iou_threshold(truth_class_id: int, taxonomy: Taxonomy) -> float:
    return 0.35 if taxonomy.coarse_name(truth_class_id) == "vehicle" else 0.50


def _match(
    predictions: list[Detection],
    truth: list[ObjectAnnotation],
    taxonomy: Taxonomy,
    key: Callable[[int], KeyT],
) -> tuple[Metrics, dict[KeyT, Metrics], dict[str, Metrics]]:
    truth_by_key: dict[tuple[str, KeyT], list[ObjectAnnotation]] = defaultdict(list)
    for item in truth:
        if not item.difficult:
            truth_by_key[(item.image_id, key(item.class_id))].append(item)

    matched: dict[tuple[str, KeyT], set[int]] = defaultdict(set)
    key_counts: dict[KeyT, list[int]] = {
        key(class_id): [0, 0, 0] for class_id in sorted(taxonomy.valid_ids)
    }
    image_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    indexed_predictions = list(enumerate(predictions))
    indexed_predictions.sort(key=lambda pair: (-pair[1].score, pair[0]))
    for _, prediction in indexed_predictions:
        prediction_key = key(prediction.class_id)
        group_key = (prediction.image_id, prediction_key)
        key_counts.setdefault(prediction_key, [0, 0, 0])
        candidates = truth_by_key.get(group_key, [])
        prediction_hbb = obb_to_hbb(prediction.polygon)
        best_index = -1
        best_iou = -1.0
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in matched[group_key]:
                continue
            iou = hbb_iou(prediction_hbb, obb_to_hbb(candidate.polygon))
            if iou >= _iou_threshold(candidate.class_id, taxonomy) and iou > best_iou:
                best_iou = iou
                best_index = candidate_index

        if best_index >= 0:
            matched[group_key].add(best_index)
            key_counts[prediction_key][0] += 1
            image_counts[prediction.image_id][0] += 1
        else:
            key_counts[prediction_key][1] += 1
            image_counts[prediction.image_id][1] += 1

    for (image_id, item_key), items in truth_by_key.items():
        missed = len(items) - len(matched[(image_id, item_key)])
        key_counts.setdefault(item_key, [0, 0, 0])
        key_counts[item_key][2] += missed
        image_counts[image_id][2] += missed

    by_key = {
        item_key: Metrics(tp=values[0], fp=values[1], fn=values[2])
        for item_key, values in key_counts.items()
    }
    by_image = {
        image_id: Metrics(tp=values[0], fp=values[1], fn=values[2])
        for image_id, values in sorted(image_counts.items())
    }
    overall = Metrics(
        tp=sum(item.tp for item in by_key.values()),
        fp=sum(item.fp for item in by_key.values()),
        fn=sum(item.fn for item in by_key.values()),
    )
    return overall, by_key, by_image


def evaluate(
    predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
    taxonomy: Taxonomy = get_taxonomy("legacy3"),  # noqa: B008
) -> EvaluationReport:
    prediction_items = list(predictions)
    truth_items = list(ground_truth)
    for index, item in enumerate(prediction_items):
        _validate_detection(item, index, taxonomy)
    for index, item in enumerate(truth_items):
        _validate_truth(item, index, taxonomy)

    overall_class_agnostic, _, by_image = _match(
        prediction_items,
        truth_items,
        taxonomy,
        lambda _: "all",
    )
    _, by_coarse_class, _ = _match(
        prediction_items,
        truth_items,
        taxonomy,
        taxonomy.coarse_name,
    )
    _, by_fine_class, _ = _match(
        prediction_items,
        truth_items,
        taxonomy,
        lambda class_id: class_id,
    )
    return EvaluationReport(
        overall_class_agnostic=overall_class_agnostic,
        by_coarse_class=by_coarse_class,
        by_fine_class=by_fine_class,
        by_image=by_image,
    )


def audit_false_positives(
    predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
    taxonomy: Taxonomy = get_taxonomy("legacy3"),  # noqa: B008
) -> FalsePositiveAudit:
    prediction_items = list(predictions)
    truth_items = list(ground_truth)
    for index, item in enumerate(prediction_items):
        _validate_detection(item, index, taxonomy)
    for index, item in enumerate(truth_items):
        _validate_truth(item, index, taxonomy)

    truth_by_group: dict[tuple[str, str], list[ObjectAnnotation]] = defaultdict(list)
    for item in truth_items:
        if not item.difficult:
            truth_by_group[(item.image_id, taxonomy.coarse_name(item.class_id))].append(item)

    matched: dict[tuple[str, str], set[int]] = defaultdict(set)
    counts: dict[str, list[int]] = {
        name: [0, 0] for name in sorted(set(taxonomy.coarse_by_id.values()))
    }
    indexed_predictions = sorted(
        enumerate(prediction_items),
        key=lambda pair: (-pair[1].score, pair[0]),
    )
    for _, prediction in indexed_predictions:
        coarse_name = taxonomy.coarse_name(prediction.class_id)
        group_key = (prediction.image_id, coarse_name)
        candidates = truth_by_group.get(group_key, [])
        prediction_hbb = obb_to_hbb(prediction.polygon)
        best_index = -1
        best_iou = -1.0
        overlaps: list[float] = []
        for candidate_index, candidate in enumerate(candidates):
            iou = hbb_iou(prediction_hbb, obb_to_hbb(candidate.polygon))
            overlaps.append(iou)
            if candidate_index in matched[group_key]:
                continue
            if iou >= _iou_threshold(candidate.class_id, taxonomy) and iou > best_iou:
                best_iou = iou
                best_index = candidate_index
        if best_index >= 0:
            matched[group_key].add(best_index)
        elif max(overlaps, default=0.0) > 0.0:
            counts[coarse_name][0] += 1
        else:
            counts[coarse_name][1] += 1

    return FalsePositiveAudit(
        by_coarse_class={
            name: FalsePositiveSources(overlap=values[0], background=values[1])
            for name, values in sorted(counts.items())
        }
    )


def false_positive_audit_to_dict(audit: FalsePositiveAudit) -> dict[str, object]:
    return {
        "by_coarse_class": {
            name: {
                "overlap": sources.overlap,
                "background": sources.background,
                "total": sources.total,
            }
            for name, sources in sorted(audit.by_coarse_class.items())
        }
    }


def threshold_sweep(
    predictions: list[Detection],
    ground_truth: list[ObjectAnnotation],
    thresholds: list[float],
    taxonomy: Taxonomy = get_taxonomy("legacy3"),  # noqa: B008
) -> list[tuple[float, EvaluationReport]]:
    normalized_thresholds: list[float] = []
    for index, threshold in enumerate(thresholds):
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise TypeError(f"threshold {index} must be a finite real number")
        normalized = float(threshold)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError(f"threshold {index} must be in [0, 1]")
        normalized_thresholds.append(normalized)
    return [
        (
            threshold,
            evaluate(
                [item for item in predictions if item.score >= threshold],
                ground_truth,
                taxonomy=taxonomy,
            ),
        )
        for threshold in normalized_thresholds
    ]


def _load_json(path: Path | str) -> object:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def load_coco_predictions(
    path: Path | str,
    taxonomy: Taxonomy = get_taxonomy("legacy3"),  # noqa: B008
) -> list[Detection]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise TypeError("COCO predictions must be a list")

    predictions: list[Detection] = []
    required = {"image_id", "category_id", "bbox", "score"}
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError(f"prediction record {index} has invalid fields")
        predictions.append(
            Detection(
                image_id=_normalize_image_id(item["image_id"], f"prediction {index}"),
                class_id=_validate_class_id(item["category_id"], f"prediction {index}", taxonomy),
                score=_validate_score(item["score"], f"prediction {index}"),
                polygon=_bbox_to_polygon(item["bbox"], f"prediction {index}"),
            )
        )
    return predictions


def load_coco_ground_truth(
    path: Path | str,
    taxonomy: Taxonomy = get_taxonomy("legacy3"),  # noqa: B008
) -> list[ObjectAnnotation]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError("COCO ground truth must be a mapping")
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("COCO ground truth annotations must be a list")

    truth: list[ObjectAnnotation] = []
    required = {"image_id", "category_id", "bbox"}
    for index, item in enumerate(annotations):
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise ValueError(f"ground-truth annotation {index} has invalid fields")
        iscrowd = item.get("iscrowd", 0)
        if isinstance(iscrowd, bool):
            difficult = iscrowd
        elif isinstance(iscrowd, Integral) and int(iscrowd) in {0, 1}:
            difficult = bool(iscrowd)
        else:
            raise ValueError(f"ground-truth annotation {index} iscrowd must be 0 or 1")
        truth.append(
            ObjectAnnotation(
                image_id=_normalize_image_id(item["image_id"], f"ground truth {index}"),
                class_id=_validate_class_id(
                    item["category_id"],
                    f"ground truth {index}",
                    taxonomy,
                ),
                polygon=_bbox_to_polygon(item["bbox"], f"ground truth {index}"),
                difficult=difficult,
            )
        )
    return truth


def report_to_dict(report: EvaluationReport) -> dict[str, object]:
    def metrics_dict(metrics: Metrics) -> dict[str, float | int]:
        return {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "recall": metrics.recall,
            "fdr": metrics.fdr,
        }

    return {
        "overall_class_agnostic": metrics_dict(report.overall_class_agnostic),
        "by_coarse_class": {
            name: metrics_dict(metrics) for name, metrics in sorted(report.by_coarse_class.items())
        },
        "by_fine_class": {
            str(class_id): metrics_dict(metrics)
            for class_id, metrics in sorted(report.by_fine_class.items())
        },
        "by_image": {
            image_id: metrics_dict(metrics) for image_id, metrics in sorted(report.by_image.items())
        },
    }
