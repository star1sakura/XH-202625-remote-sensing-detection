from __future__ import annotations

import math
from typing import cast

from shapely.errors import GEOSException
from shapely.geometry import Polygon

from xh_detect.types import Polygon4

HBB = tuple[float, float, float, float]


def _normalize_hbb(box: HBB) -> HBB:
    xmin = min(box[0], box[2])
    ymin = min(box[1], box[3])
    xmax = max(box[0], box[2])
    ymax = max(box[1], box[3])
    return xmin, ymin, xmax, ymax


def obb_to_hbb(polygon: Polygon4) -> HBB:
    xs, ys = zip(*polygon, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def clip_polygon(polygon: Polygon4, width: int, height: int) -> Polygon4:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    max_x = float(width - 1)
    max_y = float(height - 1)
    clipped = tuple((min(max(x, 0.0), max_x), min(max(y, 0.0), max_y)) for x, y in polygon)
    return cast(Polygon4, clipped)


def hbb_iou(left: HBB, right: HBB) -> float:
    left = _normalize_hbb(left)
    right = _normalize_hbb(right)
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _polygon_or_none(polygon: Polygon4) -> Polygon | None:
    if not all(math.isfinite(coord) for point in polygon for coord in point):
        return None

    try:
        shape = Polygon(polygon)
    except GEOSException:
        return None

    if not shape.is_valid or shape.is_empty or shape.area <= 0:
        return None
    return shape


def polygon_iou(left: Polygon4, right: Polygon4) -> float:
    left_shape = _polygon_or_none(left)
    right_shape = _polygon_or_none(right)
    if left_shape is None or right_shape is None:
        return 0.0

    try:
        union = left_shape.union(right_shape).area
        if union <= 0:
            return 0.0
        intersection = left_shape.intersection(right_shape).area
    except GEOSException:
        return 0.0

    return intersection / union
