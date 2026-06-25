from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from numbers import Real
from typing import cast

import cv2
import numpy as np

from xh_detect.geometry import obb_to_hbb
from xh_detect.taxonomy import Taxonomy, get_taxonomy
from xh_detect.types import Detection, ImageArray, Polygon4

CLASS_NAMES = {0: "aircraft", 1: "ship", 2: "vehicle"}


def _validate_class_id(class_id: object, taxonomy: Taxonomy) -> int:
    if isinstance(class_id, bool) or not isinstance(class_id, int):
        raise ValueError(
            f"detection class_id must be one of {sorted(taxonomy.valid_ids)} "
            f"for taxonomy {taxonomy.key!r}"
        )
    if class_id not in taxonomy.valid_ids:
        raise ValueError(
            f"detection class_id must be one of {sorted(taxonomy.valid_ids)} "
            f"for taxonomy {taxonomy.key!r}"
        )
    return class_id


def _color(class_id: int) -> tuple[int, int, int]:
    return (
        64 + (class_id * 53) % 192,
        64 + (class_id * 97) % 192,
        64 + (class_id * 193) % 192,
    )


def _validate_polygon(polygon: object) -> Polygon4:
    try:
        points = tuple(tuple(point) for point in polygon)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ValueError("detection polygon must contain four finite points") from exc
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise ValueError("detection polygon must contain four finite points")
    if not all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        for point in points
        for value in point
    ):
        raise ValueError("detection polygon must contain four finite points")
    return cast(
        Polygon4,
        tuple((float(point[0]), float(point[1])) for point in points),
    )


def _validate_image(image: object) -> ImageArray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image height and width must be positive")
    if image.dtype != np.uint8:
        raise TypeError("image dtype must be uint8 for OpenCV visualization")
    return cast(ImageArray, image)


def class_counts(
    detections: Iterable[Detection],
    taxonomy: Taxonomy = get_taxonomy("legacy3"),  # noqa: B008
) -> dict[str, dict[str, int]]:
    fine_counts: Counter[int] = Counter()
    coarse_counts: Counter[str] = Counter()
    for detection in detections:
        class_id = _validate_class_id(detection.class_id, taxonomy)
        fine_counts[class_id] += 1
        coarse_counts[taxonomy.coarse_name(class_id)] += 1
    return {
        "coarse": {name: coarse_counts[name] for name in ("aircraft", "ship", "vehicle")},
        "fine": {
            taxonomy.names[class_id]: fine_counts[class_id]
            for class_id in sorted(taxonomy.valid_ids)
        },
    }


def draw_detections(
    image: ImageArray,
    detections: Iterable[Detection],
    mode: str = "obb",
    taxonomy: Taxonomy = get_taxonomy("legacy3"),  # noqa: B008
) -> ImageArray:
    if mode not in {"obb", "hbb"}:
        raise ValueError("mode must be 'obb' or 'hbb'")
    source = _validate_image(image)
    rendered = source.copy()

    for detection in detections:
        class_id = _validate_class_id(detection.class_id, taxonomy)
        polygon = _validate_polygon(detection.polygon)
        if (
            isinstance(detection.score, bool)
            or not isinstance(detection.score, Real)
            or not math.isfinite(float(detection.score))
        ):
            raise ValueError("detection score must be finite")
        color = _color(class_id)
        if mode == "obb":
            points = np.rint(np.asarray(polygon)).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(rendered, [points], isClosed=True, color=color, thickness=2)
            x, y = (int(round(value)) for value in polygon[0])
        else:
            xmin, ymin, xmax, ymax = obb_to_hbb(polygon)
            x, y = int(round(xmin)), int(round(ymin))
            cv2.rectangle(
                rendered,
                (x, y),
                (int(round(xmax)), int(round(ymax))),
                color,
                2,
            )

        label = f"{taxonomy.names[class_id]} {float(detection.score):.2f}"
        text_x = max(0, min(x, rendered.shape[1] - 1))
        text_y = max(15, min(y, rendered.shape[0] - 1))
        cv2.putText(
            rendered,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    return cast(ImageArray, rendered)
