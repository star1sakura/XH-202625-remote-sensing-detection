from __future__ import annotations

import math
from collections.abc import Collection
from numbers import Integral, Real

import numpy as np


def positive_xyxy_mask(bboxes: np.ndarray) -> np.ndarray:
    """Return the valid positive-area rows from an ``xyxy`` box array."""
    bbox_array = np.asarray(bboxes)
    if bbox_array.ndim != 2 or bbox_array.shape[1] != 4:
        raise ValueError("bboxes must have shape (N, 4)")
    if bbox_array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("bboxes must be a real numeric array")
    if not np.isfinite(bbox_array).all():
        raise ValueError("bboxes must be finite")
    return (bbox_array[:, 2] > bbox_array[:, 0]) & (
        bbox_array[:, 3] > bbox_array[:, 1]
    )


def _validate_image_id(value: object) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (Integral, str)):
        raise TypeError("image_id must be an integer or string")
    normalized = int(value) if isinstance(value, Integral) else value
    if normalized == "":
        raise ValueError("image_id must not be empty")
    return normalized


def _validate_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("confidence must be a finite real number")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return confidence


def instances_to_coco_predictions(
    *,
    image_id: int | str,
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    confidence: float,
    valid_class_ids: Collection[int],
) -> list[dict[str, object]]:
    """Convert MMDetection HBB instances to validated COCO result rows."""
    normalized_image_id = _validate_image_id(image_id)
    normalized_confidence = _validate_confidence(confidence)
    valid_ids = set(valid_class_ids)
    if not valid_ids or any(
        isinstance(item, bool) or not isinstance(item, Integral) for item in valid_ids
    ):
        raise ValueError("valid_class_ids must contain integers")
    valid_ids = {int(item) for item in valid_ids}

    bbox_array = np.asarray(bboxes)
    score_array = np.asarray(scores)
    label_array = np.asarray(labels)
    count = len(score_array) if score_array.ndim == 1 else -1
    if bbox_array.shape != (count, 4):
        raise ValueError("bboxes must have shape (N, 4)")
    if label_array.shape != (count,):
        raise ValueError("labels must have shape (N,)")
    if bbox_array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("bboxes must be a real numeric array")
    if score_array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("scores must be a real numeric array")
    if label_array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("labels must be a real numeric array")
    if not np.isfinite(bbox_array).all() or not np.isfinite(score_array).all():
        raise ValueError("bboxes and scores must be finite")

    rows: list[dict[str, object]] = []
    for index in range(count):
        score = float(score_array[index])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score {index} must be in [0, 1]")
        label_value = float(label_array[index])
        if not math.isfinite(label_value) or not label_value.is_integer():
            raise ValueError(f"label {index} must be an integer")
        label = int(label_value)
        if label not in valid_ids:
            raise ValueError(f"label {index} is not present in valid_class_ids")
        x1, y1, x2, y2 = (float(value) for value in bbox_array[index])
        width = x2 - x1
        height = y2 - y1
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"bbox {index} must have positive width and height")
        if score < normalized_confidence:
            continue
        rows.append(
            {
                "image_id": normalized_image_id,
                "category_id": label,
                "bbox": [x1, y1, width, height],
                "score": score,
            }
        )
    return rows
