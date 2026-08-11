from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import ultralytics
from ultralytics.utils.metrics import DetMetrics, box_iou

from xh_detect.geometry import obb_to_hbb
from xh_detect.taxonomy import Taxonomy
from xh_detect.types import Detection, ObjectAnnotation

ULTRALYTICS_METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


@dataclass(frozen=True)
class UltralyticsEvaluation:
    metrics: dict[str, float]
    per_class: tuple[dict[str, object], ...]
    images: int
    predictions: int
    targets: int
    source_targets: int
    duplicate_targets_removed: int
    max_detections: int
    ultralytics_version: str


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> np.ndarray:
    """Match one image exactly like Ultralytics 8.4.71 BaseValidator."""
    correct = np.zeros((pred_classes.shape[0], iou_thresholds.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    class_iou = (iou * correct_class).cpu().numpy()
    for index, threshold in enumerate(iou_thresholds.cpu().tolist()):
        matches = np.array(np.nonzero(class_iou >= threshold)).T
        if not matches.shape[0]:
            continue
        if matches.shape[0] > 1:
            matches = matches[class_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), index] = True
    return correct


def _boxes(items: Iterable[Detection | ObjectAnnotation]) -> torch.Tensor:
    rows = [obb_to_hbb(item.polygon) for item in items]
    if not rows:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def evaluate_ultralytics(
    predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
    *,
    taxonomy: Taxonomy,
    max_detections: int = 300,
    deduplicate_ground_truth: bool = True,
) -> UltralyticsEvaluation:
    """Compute the four Ultralytics detection metrics from model-agnostic boxes."""
    if isinstance(max_detections, bool) or not isinstance(max_detections, int):
        raise TypeError("max_detections must be an integer")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    if not isinstance(deduplicate_ground_truth, bool):
        raise TypeError("deduplicate_ground_truth must be a boolean")

    predictions_by_image: dict[str, list[Detection]] = defaultdict(list)
    truth_by_image: dict[str, list[ObjectAnnotation]] = defaultdict(list)
    source_target_count = 0
    seen_truth: set[tuple[object, ...]] = set()
    for prediction in predictions:
        if prediction.class_id not in taxonomy.valid_ids:
            raise ValueError(f"unknown prediction class ID: {prediction.class_id}")
        predictions_by_image[prediction.image_id].append(prediction)
    for annotation in ground_truth:
        if annotation.class_id not in taxonomy.valid_ids:
            raise ValueError(f"unknown ground-truth class ID: {annotation.class_id}")
        if not annotation.difficult:
            source_target_count += 1
            truth_key = (
                annotation.image_id,
                annotation.class_id,
                *(coordinate for point in annotation.polygon for coordinate in point),
            )
            if deduplicate_ground_truth and truth_key in seen_truth:
                continue
            seen_truth.add(truth_key)
            truth_by_image[annotation.image_id].append(annotation)

    image_ids = sorted(set(predictions_by_image) | set(truth_by_image))
    metric = DetMetrics(names=dict(taxonomy.names))
    iou_thresholds = torch.linspace(0.5, 0.95, 10)
    prediction_count = 0
    target_count = 0
    for image_id in image_ids:
        image_predictions = sorted(
            predictions_by_image[image_id],
            key=lambda item: item.score,
            reverse=True,
        )[:max_detections]
        image_truth = truth_by_image[image_id]
        pred_classes = torch.tensor(
            [item.class_id for item in image_predictions], dtype=torch.int64
        )
        true_classes = torch.tensor([item.class_id for item in image_truth], dtype=torch.int64)
        if image_predictions and image_truth:
            correct = _match_predictions(
                pred_classes,
                true_classes,
                box_iou(_boxes(image_truth), _boxes(image_predictions)),
                iou_thresholds,
            )
        else:
            correct = np.zeros((len(image_predictions), 10), dtype=bool)
        scores = np.asarray([item.score for item in image_predictions], dtype=np.float64)
        predicted = pred_classes.numpy()
        targets = true_classes.numpy()
        metric.update_stats(
            {
                "tp": correct,
                "conf": scores,
                "pred_cls": predicted,
                "target_cls": targets,
                "target_img": np.unique(targets),
                "im_name": image_id,
            }
        )
        prediction_count += len(image_predictions)
        target_count += len(image_truth)

    if not target_count:
        raise ValueError("ground truth must contain at least one non-difficult target")
    metric.process(plot=False)
    return UltralyticsEvaluation(
        metrics={key: float(metric.results_dict[key]) for key in ULTRALYTICS_METRIC_KEYS},
        per_class=tuple(metric.summary(normalize=True, decimals=8)),
        images=len(image_ids),
        predictions=prediction_count,
        targets=target_count,
        source_targets=source_target_count,
        duplicate_targets_removed=source_target_count - target_count,
        max_detections=max_detections,
        ultralytics_version=ultralytics.__version__,
    )


def ultralytics_evaluation_to_dict(result: UltralyticsEvaluation) -> dict[str, object]:
    per_class = [
        {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in row.items()
        }
        for row in result.per_class
    ]
    return {
        "metrics": result.metrics,
        "per_class": per_class,
        "images": result.images,
        "predictions": result.predictions,
        "targets": result.targets,
        "source_targets": result.source_targets,
        "duplicate_targets_removed": result.duplicate_targets_removed,
        "max_detections": result.max_detections,
        "ultralytics_version": result.ultralytics_version,
    }
