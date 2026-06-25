from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Protocol, cast

import numpy as np
import torch
from ultralytics import YOLO

from xh_detect.types import BoxPrediction, ImageArray, Polygon4

_INT64_MAX = int(np.iinfo(np.int64).max)


class Detector(Protocol):
    def predict(self, images: list[ImageArray], confidence: float) -> list[list[BoxPrediction]]: ...


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _validate_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("confidence must be a finite real number in [0, 1]")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be a finite real number in [0, 1]")
    return confidence


def _to_numpy_array(
    value: object,
    *,
    result_index: int,
    field_name: str,
    result_type: str = "OBB",
) -> np.ndarray:
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise ValueError(
            f"result {result_index} has invalid {result_type} {field_name} values"
        ) from exc


def _ensure_finite(array: np.ndarray, *, result_index: int, result_type: str = "OBB") -> None:
    try:
        is_finite = np.isfinite(array).all()
    except TypeError as exc:
        raise ValueError(f"result {result_index} contains non-finite {result_type} values") from exc
    if not bool(is_finite):
        raise ValueError(f"result {result_index} contains non-finite {result_type} values")


def _format_array_value(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return repr(value)


def _validate_real_numeric_array(
    array: np.ndarray,
    *,
    result_index: int,
    field_name: str,
    result_type: str = "OBB",
) -> None:
    if array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(
            f"result {result_index} has invalid {result_type} {field_name} values: "
            "expected a real non-boolean numeric array, "
            f"got dtype {array.dtype}"
        )


def _is_real_numeric_scalar(value: object) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value,
        (Real, np.integer, np.floating),
    )


def _validate_class_ids(
    classes: np.ndarray,
    *,
    result_index: int,
    result_type: str = "OBB",
) -> list[int]:
    _validate_real_numeric_array(
        classes,
        result_index=result_index,
        field_name="class",
        result_type=result_type,
    )
    validated: list[int] = []
    for box_index, class_id in enumerate(classes):
        if not _is_real_numeric_scalar(class_id):
            raise ValueError(
                f"result {result_index} has invalid {result_type} class at box {box_index}: "
                "expected a finite non-negative integer, got "
                f"{_format_array_value(class_id)}"
            )

        if isinstance(class_id, (Integral, np.integer)):
            validated_class_id = int(class_id)
            is_valid = 0 <= validated_class_id <= _INT64_MAX
        else:
            numeric_class_id = float(class_id)
            is_valid = (
                math.isfinite(numeric_class_id)
                and numeric_class_id >= 0
                and numeric_class_id.is_integer()
                and numeric_class_id <= _INT64_MAX
            )
            validated_class_id = int(numeric_class_id) if is_valid else 0

        if not is_valid:
            raise ValueError(
                f"result {result_index} has invalid {result_type} class at box {box_index}: "
                "expected a finite non-negative int64 integer, got "
                f"{_format_array_value(class_id)}"
            )
        validated.append(validated_class_id)
    return validated


def _validate_scores(
    scores: np.ndarray,
    *,
    result_index: int,
    result_type: str = "OBB",
) -> list[float]:
    _validate_real_numeric_array(
        scores,
        result_index=result_index,
        field_name="score",
        result_type=result_type,
    )
    validated: list[float] = []
    for box_index, score in enumerate(scores):
        if not _is_real_numeric_scalar(score):
            raise ValueError(
                f"result {result_index} has invalid {result_type} score at box {box_index}: "
                "expected a finite real value in [0, 1], got "
                f"{_format_array_value(score)}"
            )

        numeric_score = float(score)
        if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            raise ValueError(
                f"result {result_index} has invalid {result_type} score at box {box_index}: "
                "expected a finite real value in [0, 1], got "
                f"{_format_array_value(score)}"
            )
        validated.append(numeric_score)
    return validated


def _extract_obb_predictions(result: object, *, result_index: int) -> list[BoxPrediction]:
    obb = getattr(result, "obb", None)
    if obb is None:
        raise ValueError(f"result {result_index} is missing OBB predictions")

    polygons = _to_numpy_array(
        obb.xyxyxyxy.detach().cpu().numpy(),
        result_index=result_index,
        field_name="polygon",
    )
    classes = _to_numpy_array(
        obb.cls.detach().cpu().numpy(),
        result_index=result_index,
        field_name="class",
    ).reshape(-1)
    scores = _to_numpy_array(
        obb.conf.detach().cpu().numpy(),
        result_index=result_index,
        field_name="score",
    ).reshape(-1)

    if polygons.ndim != 3 or polygons.shape[1:] != (4, 2):
        raise ValueError(
            f"result {result_index} has invalid OBB polygon shape: expected (N, 4, 2), "
            f"got {polygons.shape}"
        )

    polygon_count = polygons.shape[0]
    if polygon_count != len(classes) or polygon_count != len(scores):
        raise ValueError(
            "result "
            f"{result_index} has inconsistent OBB lengths: polygons={polygon_count}, "
            f"classes={len(classes)}, scores={len(scores)}"
        )

    _ensure_finite(polygons, result_index=result_index)
    validated_classes = _validate_class_ids(classes, result_index=result_index)
    validated_scores = _validate_scores(scores, result_index=result_index)

    predictions: list[BoxPrediction] = []
    for polygon, class_id, score in zip(
        polygons,
        validated_classes,
        validated_scores,
        strict=True,
    ):
        points = tuple((float(point[0]), float(point[1])) for point in polygon)
        predictions.append(
            BoxPrediction(
                class_id=class_id,
                score=score,
                polygon=cast(Polygon4, points),
            )
        )
    return predictions


def _extract_hbb_predictions(result: object, *, result_index: int) -> list[BoxPrediction]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError(f"result {result_index} is missing HBB boxes")

    coordinates = _to_numpy_array(
        boxes.xyxy.detach().cpu().numpy(),
        result_index=result_index,
        field_name="box",
        result_type="HBB",
    )
    classes = _to_numpy_array(
        boxes.cls.detach().cpu().numpy(),
        result_index=result_index,
        field_name="class",
        result_type="HBB",
    ).reshape(-1)
    scores = _to_numpy_array(
        boxes.conf.detach().cpu().numpy(),
        result_index=result_index,
        field_name="score",
        result_type="HBB",
    ).reshape(-1)

    if coordinates.ndim != 2 or coordinates.shape[1:] != (4,):
        raise ValueError(
            f"result {result_index} has invalid HBB shape: expected (N, 4), got {coordinates.shape}"
        )

    box_count = coordinates.shape[0]
    if box_count != len(classes) or box_count != len(scores):
        raise ValueError(
            "result "
            f"{result_index} has inconsistent HBB lengths: boxes={box_count}, "
            f"classes={len(classes)}, scores={len(scores)}"
        )

    _ensure_finite(coordinates, result_index=result_index, result_type="HBB")
    validated_classes = _validate_class_ids(
        classes,
        result_index=result_index,
        result_type="HBB",
    )
    validated_scores = _validate_scores(scores, result_index=result_index, result_type="HBB")

    predictions: list[BoxPrediction] = []
    for coordinate, class_id, score in zip(
        coordinates,
        validated_classes,
        validated_scores,
        strict=True,
    ):
        xmin, ymin, xmax, ymax = (float(value) for value in coordinate)
        predictions.append(
            BoxPrediction(
                class_id=class_id,
                score=score,
                polygon=((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)),
            )
        )
    return predictions


def _extract_predictions(
    result: object,
    *,
    result_index: int,
    task: str = "obb",
) -> list[BoxPrediction]:
    if task == "detect":
        return _extract_hbb_predictions(result, result_index=result_index)
    if task == "obb":
        return _extract_obb_predictions(result, result_index=result_index)
    raise ValueError(f"unsupported task {task!r}; expected 'detect' or 'obb'")


def _is_cuda_oom(error: RuntimeError) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True

    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def _empty_cuda_cache_if_available() -> None:
    empty_cache = getattr(torch.cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _validate_task(value: object) -> str:
    if not isinstance(value, str) or value not in {"detect", "obb"}:
        raise ValueError("task must be one of: detect, obb")
    return value


class UltralyticsDetector:
    def __init__(
        self,
        model_path: str,
        device: str,
        image_size: int,
        half: bool,
        task: str,
    ) -> None:
        validated_model_path = _validate_non_empty_string(model_path, "model_path")
        self.device = _validate_non_empty_string(device, "device")
        self.image_size = _validate_positive_integer(image_size, "image_size")
        self.half = _validate_bool(half, "half")
        self.task = _validate_task(task)
        self.model = YOLO(validated_model_path)

    def predict(self, images: list[ImageArray], confidence: float) -> list[list[BoxPrediction]]:
        validated_confidence = _validate_confidence(confidence)
        if not images:
            return []
        results = list(
            self.model.predict(
                source=images,
                imgsz=self.image_size,
                conf=validated_confidence,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        )
        if len(results) != len(images):
            raise ValueError(
                f"Ultralytics returned {len(results)} results for {len(images)} input images"
            )
        return [
            _extract_predictions(result, result_index=index, task=self.task)
            for index, result in enumerate(results)
        ]


class UltralyticsOBBDetector(UltralyticsDetector):
    def __init__(
        self,
        model_path: str,
        device: str,
        image_size: int,
        half: bool,
    ) -> None:
        super().__init__(model_path, device, image_size, half, task="obb")


def predict_with_oom_backoff(
    detector: Detector,
    images: Sequence[ImageArray],
    confidence: float,
    initial_batch_size: int,
) -> list[list[BoxPrediction]]:
    validated_confidence = _validate_confidence(confidence)
    validated_batch_size = _validate_positive_integer(initial_batch_size, "initial_batch_size")
    if not images:
        return []

    predictions: list[list[BoxPrediction]] = []
    index = 0
    batch_size = min(validated_batch_size, len(images))

    while index < len(images):
        chunk = list(images[index : index + batch_size])
        try:
            chunk_predictions = detector.predict(chunk, validated_confidence)
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            _empty_cuda_cache_if_available()
            continue

        if len(chunk_predictions) != len(chunk):
            raise ValueError(
                "detector returned "
                f"{len(chunk_predictions)} results for a chunk of {len(chunk)} images"
            )

        predictions.extend(chunk_predictions)
        index += len(chunk)

    return predictions
