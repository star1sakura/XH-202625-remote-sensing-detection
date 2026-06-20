from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from numbers import Integral, Real
from pathlib import Path
from typing import cast

from shapely.errors import GEOSException
from shapely.geometry import Polygon

from xh_detect.geometry import obb_to_hbb
from xh_detect.types import Detection, Polygon4

_COCO_FIELDS = {"image_id", "category_id", "bbox", "score"}


def _is_non_bool_integral(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral)


def _validate_map(image_id_map: Mapping[str, int]) -> dict[str, int]:
    validated: dict[str, int] = {}
    seen_values: set[int] = set()
    for key, value in image_id_map.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("image_id_map keys must be non-empty strings")
        if not _is_non_bool_integral(value):
            raise TypeError("image_id_map values must be non-bool non-negative integers")
        numeric_value = int(value)
        if numeric_value < 0:
            raise ValueError("image_id_map values must be non-bool non-negative integers")
        if numeric_value in seen_values:
            raise ValueError("image_id_map values must be unique")
        seen_values.add(numeric_value)
        validated[key] = numeric_value
    return validated


def _validate_real_number(field_name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be a finite real number")
    return numeric_value


def _validate_detection_polygon(detection: Detection, index: int) -> Polygon4:
    try:
        points = detection.polygon
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"detection {index} (image_id={detection.image_id!r}) has invalid polygon"
        ) from exc

    if len(points) != 4:
        raise ValueError(
            f"detection {index} (image_id={detection.image_id!r}) has invalid polygon: "
            "expected exactly 4 points"
        )

    validated_points: list[tuple[float, float]] = []
    for point_index, point in enumerate(points):
        try:
            x_raw, y_raw = point
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"detection {index} (image_id={detection.image_id!r}) has invalid polygon: "
                f"point {point_index} is not a 2-item coordinate"
            ) from exc

        x = _validate_real_number(
            f"detection {index} (image_id={detection.image_id!r}) polygon[{point_index}][0]",
            x_raw,
        )
        y = _validate_real_number(
            f"detection {index} (image_id={detection.image_id!r}) polygon[{point_index}][1]",
            y_raw,
        )
        validated_points.append((x, y))

    try:
        shape = Polygon(validated_points)
    except GEOSException as exc:
        raise ValueError(
            f"detection {index} (image_id={detection.image_id!r}) has invalid polygon"
        ) from exc

    if shape.is_empty or not shape.is_valid or shape.area <= 0.0:
        raise ValueError(
            f"detection {index} (image_id={detection.image_id!r}) has invalid polygon"
        )

    return cast(Polygon4, tuple(validated_points))


def validate_coco_results(records: list[dict[str, object]]) -> None:
    if not isinstance(records, list):
        raise TypeError("records must be a list of COCO detection records")

    seen: set[tuple[int, int, tuple[float, float, float, float]]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"record {index} must be a mapping")

        keys = set(record)
        missing = _COCO_FIELDS - keys
        extra = keys - _COCO_FIELDS
        if missing or extra:
            parts: list[str] = []
            if missing:
                parts.append(f"missing required fields: {', '.join(sorted(missing))}")
            if extra:
                parts.append(f"unexpected fields: {', '.join(sorted(extra))}")
            raise ValueError(f"record {index} has " + "; ".join(parts))

        image_id = record["image_id"]
        if not _is_non_bool_integral(image_id):
            raise TypeError(f"record {index} field image_id must be a non-bool integer")
        image_id_int = int(image_id)
        if image_id_int < 0:
            raise ValueError(f"record {index} field image_id must be non-negative")

        category_id = record["category_id"]
        if not _is_non_bool_integral(category_id):
            raise TypeError(f"record {index} field category_id must be a non-bool integer")
        category_id_int = int(category_id)
        if category_id_int not in {0, 1, 2}:
            raise ValueError("category_id must be one of 0, 1, or 2")

        bbox = record["bbox"]
        if type(bbox) is not list:
            raise TypeError(f"record {index} field bbox must be a list")
        if len(bbox) != 4:
            raise ValueError(f"record {index} field bbox must contain exactly 4 items")

        normalized_bbox: list[float] = []
        for coord_index, coord in enumerate(bbox):
            numeric_coord = _validate_real_number(
                f"record {index} field bbox[{coord_index}]",
                coord,
            )
            normalized_bbox.append(numeric_coord)

        xmin, ymin, width, height = normalized_bbox
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"record {index} field bbox width and height must be positive")

        score = _validate_real_number(f"record {index} field score", record["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"record {index} field score must be within [0, 1]")

        duplicate_key = (image_id_int, category_id_int, tuple(normalized_bbox))
        if duplicate_key in seen:
            raise ValueError(f"record {index} is a duplicate detection")
        seen.add(duplicate_key)


def _build_record(
    detection: Detection,
    image_id: int,
    polygon: Polygon4,
) -> dict[str, object]:
    xmin, ymin, xmax, ymax = obb_to_hbb(polygon)
    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
    return {
        "image_id": image_id,
        "category_id": detection.class_id,
        "bbox": bbox,
        "score": detection.score,
    }


def _write_json_atomic(destination: Path, payload: list[dict[str, object]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"destination must be a file path: {destination}")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            with suppress(FileNotFoundError):
                temp_path.unlink()
        raise


def export_coco_results(
    detections: Iterable[Detection],
    image_id_map: Mapping[str, int],
    destination: Path,
) -> Path:
    if not isinstance(destination, Path):
        destination = Path(destination)

    validated_map = _validate_map(image_id_map)
    records = []
    for index, detection in enumerate(detections):
        try:
            image_id = validated_map[detection.image_id]
        except KeyError as exc:
            raise ValueError(f"unknown image_id {detection.image_id!r}") from exc
        polygon = _validate_detection_polygon(detection, index)
        records.append(_build_record(detection, image_id, polygon))

    validate_coco_results(records)
    _write_json_atomic(destination, records)
    return destination
