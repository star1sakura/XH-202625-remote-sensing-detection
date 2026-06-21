from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

from xh_detect.geometry import hbb_iou, obb_to_hbb
from xh_detect.types import Detection, ObjectAnnotation, Polygon4

IOU_THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.35}
_CLASS_IDS = frozenset(IOU_THRESHOLDS)


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
    overall: Metrics
    by_class: dict[int, Metrics]
    by_image: dict[str, Metrics]


def _validate_class_id(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{context} category_id must be an integer")
    class_id = int(value)
    if class_id not in _CLASS_IDS:
        raise ValueError(f"{context} category_id must be one of 0, 1, or 2")
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


def _validate_detection(item: Detection, index: int) -> None:
    if not isinstance(item.image_id, str) or not item.image_id:
        raise ValueError(f"prediction {index} image_id must be a non-empty string")
    _validate_class_id(item.class_id, f"prediction {index}")
    _validate_score(item.score, f"prediction {index}")
    coordinates = [coordinate for point in item.polygon for coordinate in point]
    if len(coordinates) != 8 or not all(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in coordinates
    ):
        raise ValueError(f"prediction {index} polygon must contain four finite points")


def _validate_truth(item: ObjectAnnotation, index: int) -> None:
    if not isinstance(item.image_id, str) or not item.image_id:
        raise ValueError(f"ground truth {index} image_id must be a non-empty string")
    _validate_class_id(item.class_id, f"ground truth {index}")
    coordinates = [coordinate for point in item.polygon for coordinate in point]
    if len(coordinates) != 8 or not all(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in coordinates
    ):
        raise ValueError(f"ground truth {index} polygon must contain four finite points")


def evaluate(
    predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
) -> EvaluationReport:
    prediction_items = list(predictions)
    truth_items = list(ground_truth)
    for index, item in enumerate(prediction_items):
        _validate_detection(item, index)
    for index, item in enumerate(truth_items):
        _validate_truth(item, index)

    truth_by_key: dict[tuple[str, int], list[ObjectAnnotation]] = defaultdict(list)
    for item in truth_items:
        if not item.difficult:
            truth_by_key[(item.image_id, item.class_id)].append(item)

    matched: dict[tuple[str, int], set[int]] = defaultdict(set)
    class_counts = {class_id: [0, 0, 0] for class_id in sorted(_CLASS_IDS)}
    image_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    indexed_predictions = list(enumerate(prediction_items))
    indexed_predictions.sort(key=lambda pair: (-pair[1].score, pair[0]))
    for _, prediction in indexed_predictions:
        key = (prediction.image_id, prediction.class_id)
        candidates = truth_by_key.get(key, [])
        prediction_hbb = obb_to_hbb(prediction.polygon)
        best_index = -1
        best_iou = -1.0
        for candidate_index, truth in enumerate(candidates):
            if candidate_index in matched[key]:
                continue
            iou = hbb_iou(prediction_hbb, obb_to_hbb(truth.polygon))
            if iou > best_iou:
                best_iou = iou
                best_index = candidate_index

        if best_index >= 0 and best_iou >= IOU_THRESHOLDS[prediction.class_id]:
            matched[key].add(best_index)
            class_counts[prediction.class_id][0] += 1
            image_counts[prediction.image_id][0] += 1
        else:
            class_counts[prediction.class_id][1] += 1
            image_counts[prediction.image_id][1] += 1

    for (image_id, class_id), truths in truth_by_key.items():
        missed = len(truths) - len(matched[(image_id, class_id)])
        class_counts[class_id][2] += missed
        image_counts[image_id][2] += missed

    by_class = {
        class_id: Metrics(tp=values[0], fp=values[1], fn=values[2])
        for class_id, values in class_counts.items()
    }
    overall = Metrics(
        tp=sum(item.tp for item in by_class.values()),
        fp=sum(item.fp for item in by_class.values()),
        fn=sum(item.fn for item in by_class.values()),
    )
    by_image = {
        image_id: Metrics(tp=values[0], fp=values[1], fn=values[2])
        for image_id, values in sorted(image_counts.items())
    }
    return EvaluationReport(overall=overall, by_class=by_class, by_image=by_image)


def threshold_sweep(
    predictions: list[Detection],
    ground_truth: list[ObjectAnnotation],
    thresholds: list[float],
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
            ),
        )
        for threshold in normalized_thresholds
    ]


def _load_json(path: Path | str) -> object:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def load_coco_predictions(path: Path | str) -> list[Detection]:
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
                class_id=_validate_class_id(item["category_id"], f"prediction {index}"),
                score=_validate_score(item["score"], f"prediction {index}"),
                polygon=_bbox_to_polygon(item["bbox"], f"prediction {index}"),
            )
        )
    return predictions


def load_coco_ground_truth(path: Path | str) -> list[ObjectAnnotation]:
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
                class_id=_validate_class_id(item["category_id"], f"ground truth {index}"),
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
        "overall": metrics_dict(report.overall),
        "by_class": {
            str(class_id): metrics_dict(metrics)
            for class_id, metrics in sorted(report.by_class.items())
        },
        "by_image": {
            image_id: metrics_dict(metrics)
            for image_id, metrics in sorted(report.by_image.items())
        },
    }
