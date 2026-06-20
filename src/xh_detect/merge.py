from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from numbers import Integral, Real
from typing import cast

from xh_detect.geometry import clip_polygon, hbb_iou, obb_to_hbb, polygon_iou
from xh_detect.types import BoxPrediction, Detection, Polygon4, TileMeta


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_non_negative_finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{name} must be a finite real number")
    return numeric_value


def _validate_iou_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("iou_threshold must be a finite real number in [0, 1]")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
        raise ValueError("iou_threshold must be a finite real number in [0, 1]")
    return numeric_value


def _validate_tile_meta(meta: TileMeta, image_width: int, image_height: int) -> None:
    if meta.width <= 0 or meta.height <= 0:
        raise ValueError("meta width and height must be positive")
    if meta.valid_width <= 0 or meta.valid_height <= 0:
        raise ValueError("meta valid_width and valid_height must be positive")
    if meta.valid_width > meta.width or meta.valid_height > meta.height:
        raise ValueError("meta valid dimensions must not exceed tile dimensions")
    if meta.x < 0 or meta.y < 0:
        raise ValueError("meta x and y must be non-negative")
    if meta.x + meta.valid_width > image_width or meta.y + meta.valid_height > image_height:
        raise ValueError("meta valid area must fit inside the image")


def _shift_polygon(polygon: Polygon4, dx: float, dy: float) -> Polygon4:
    shifted = tuple((x + dx, y + dy) for x, y in polygon)
    return cast(Polygon4, shifted)


def project_prediction(
    prediction: BoxPrediction,
    meta: TileMeta,
    image_width: int,
    image_height: int,
) -> Detection:
    image_width = _validate_positive_int("image_width", image_width)
    image_height = _validate_positive_int("image_height", image_height)
    _validate_tile_meta(meta, image_width, image_height)

    shifted = _shift_polygon(prediction.polygon, float(meta.x), float(meta.y))
    clipped = clip_polygon(shifted, image_width + 1, image_height + 1)
    return Detection(
        image_id=meta.image_id,
        class_id=prediction.class_id,
        score=prediction.score,
        polygon=clipped,
    )


def keep_tile_prediction(
    prediction: BoxPrediction,
    meta: TileMeta,
    image_width: int,
    image_height: int,
    margin: float,
) -> bool:
    image_width = _validate_positive_int("image_width", image_width)
    image_height = _validate_positive_int("image_height", image_height)
    margin_value = _validate_non_negative_finite_real("margin", margin)
    _validate_tile_meta(meta, image_width, image_height)

    xmin, ymin, xmax, ymax = obb_to_hbb(prediction.polygon)
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    if not (0.0 <= center_x < meta.valid_width and 0.0 <= center_y < meta.valid_height):
        return False
    if margin_value == 0.0:
        return True

    touches_left_internal = meta.x > 0 and xmin < margin_value
    touches_top_internal = meta.y > 0 and ymin < margin_value
    touches_right_internal = (
        meta.x + meta.valid_width < image_width
        and xmax > float(meta.valid_width) - margin_value
    )
    touches_bottom_internal = (
        meta.y + meta.valid_height < image_height
        and ymax > float(meta.valid_height) - margin_value
    )
    return not (
        touches_left_internal
        or touches_top_internal
        or touches_right_internal
        or touches_bottom_internal
    )


def merge_detections(
    detections: Iterable[Detection],
    iou_threshold: float,
) -> list[Detection]:
    threshold = _validate_iou_threshold(iou_threshold)

    grouped: dict[tuple[str, int], list[tuple[int, Detection]]] = defaultdict(list)
    for original_index, detection in enumerate(detections):
        grouped[(detection.image_id, detection.class_id)].append((original_index, detection))

    kept: list[tuple[int, Detection]] = []
    for group in grouped.values():
        remaining = sorted(group, key=lambda item: (-item[1].score, item[0]))
        while remaining:
            selected_index, selected = remaining.pop(0)
            kept.append((selected_index, selected))
            selected_hbb = obb_to_hbb(selected.polygon)
            survivors: list[tuple[int, Detection]] = []
            for candidate_index, candidate in remaining:
                candidate_hbb = obb_to_hbb(candidate.polygon)
                if hbb_iou(selected_hbb, candidate_hbb) == 0.0:
                    survivors.append((candidate_index, candidate))
                    continue
                overlap = polygon_iou(selected.polygon, candidate.polygon)
                if not (overlap > 0.0 and overlap >= threshold):
                    survivors.append((candidate_index, candidate))
            remaining = survivors

    return [
        detection
        for _, detection in sorted(kept, key=lambda item: (-item[1].score, item[0]))
    ]
